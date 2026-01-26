import argparse
import json
import time
import threading
import zmq
import numpy as np
import cv2
from utils import load_policy, infer_actions, piper_step_chunk_dual, preprocess_image_for_alignment
from piper_sdk import C_PiperInterface_V2

import sys, os
sys.path.append("/home/tengenx2204/workspace/cxp/robobrain/robobrain")
from camera.camera_client import CameraClient

def _parse_action(msg: str) -> str:
    s = (msg or "").strip()
    try:
        i = s.find("{")
        j = s.rfind("}")
        if i != -1 and j != -1 and j > i:
            obj = json.loads(s[i : j + 1])
            v = obj.get("action", "")
            if isinstance(v, str) and v.strip():
                return v.strip()
    except Exception:
        pass
    return ""


def main():
    parser = argparse.ArgumentParser(description="VLA ZMQ executor with dual-arm Piper")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--config_name", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True, choices=["joint", "ee"])
    parser.add_argument("--sub-host", type=str, default="localhost")
    parser.add_argument("--sub-port", type=int, default=6002)
    parser.add_argument("--sub-topic", type=str, default="vlm")
    parser.add_argument("--chunk_sizes", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--camera_host", type=str, default="localhost")
    parser.add_argument("--camera_port", type=int, default=5555)
    parser.add_argument("--camera_left_serial", type=str, default=None)
    parser.add_argument("--camera_top_serial", type=str, default=None)
    parser.add_argument("--camera_right_serial", type=str, default=None)
    args = parser.parse_args()

    serial_to_slot = {
        "243322070835": "left",
        "243422072346": "top",
        "243422072483": "right",
    }

    cam_clients = []
    if args.camera_left_serial:
        cam_left = CameraClient(host=args.camera_host, port=args.camera_port, serial=args.camera_left_serial)
        cam_left.connect()
        cam_clients.append(("left", cam_left))
        print(f"已订阅左侧相机: {args.camera_left_serial}", flush=True)
    if args.camera_top_serial:
        cam_top = CameraClient(host=args.camera_host, port=args.camera_port, serial=args.camera_top_serial)
        cam_top.connect()
        cam_clients.append(("top", cam_top))
        print(f"已订阅顶部相机: {args.camera_top_serial}", flush=True)
    if args.camera_right_serial:
        cam_right = CameraClient(host=args.camera_host, port=args.camera_port, serial=args.camera_right_serial)
        cam_right.connect()
        cam_clients.append(("right", cam_right))
        print(f"已订阅右侧相机: {args.camera_right_serial}", flush=True)
    agg_client = None
    if len(cam_clients) == 0:
        agg_client = CameraClient(host=args.camera_host, port=args.camera_port, serial=None)
        agg_client.connect()
        print("未指定具体相机序列，已订阅所有相机最新帧。", flush=True)

    piper_right = C_PiperInterface_V2("can_arm1")
    piper_left = C_PiperInterface_V2("can_arm2")
    piper_right.ConnectPort()
    piper_left.ConnectPort()
    while not (piper_right.EnablePiper() and piper_left.EnablePiper()):
        time.sleep(0.01)
    print("Piper双臂已连接并启动。", flush=True)
    piper_right.MotionCtrl_2(0x01, 0x01, 40, 0x00)
    piper_left.MotionCtrl_2(0x01, 0x01, 40, 0x00)
    piper_right.JointCtrl(40036, 3630, -9526, -3140, 17670, -12043)
    piper_left.JointCtrl(-40543, 177, -104, -87029, -5647, 77959)
    piper_right.GripperCtrl(abs(0), 100, 0x01, 0)
    piper_left.GripperCtrl(abs(0), 500, 0x01, 0)
    print("Piper双臂初始化完成。", flush=True)

    policy = load_policy(args.checkpoint_dir, args.config_name)

    current_prompt = ""
    print(f"\n当前 Prompt: {current_prompt}", flush=True)
    print("[caution] 在相机窗口激活时按【空格键】可修改 Prompt", flush=True)

    sub_ctx = zmq.Context()
    sub_socket = sub_ctx.socket(zmq.SUB)
    sub_addr = f"tcp://{args.sub_host}:{args.sub_port}"
    sub_socket.connect(sub_addr)
    sub_socket.setsockopt(zmq.RCVHWM, 10)
    sub_socket.setsockopt_string(zmq.SUBSCRIBE, args.sub_topic)
    print(f"订阅 VLM: {sub_addr} topic={args.sub_topic}", flush=True)

    paused = [False]
    pending_prompt = [""]
    def _sub_worker():
        while True:
            try:
                parts = sub_socket.recv_multipart()
            except Exception:
                time.sleep(0.01)
                continue
            if len(parts) < 2:
                continue
            payload_b = parts[1]
            try:
                data = json.loads(payload_b.decode("utf-8"))
            except Exception:
                continue
            assistant_message = str(data.get("assistant_message") or "").strip()
            act = _parse_action(assistant_message)
            if not act:
                continue
            if paused[0]:
                pending_prompt[0] = act
                print(f"[VLA] 收到新动作(暂存): {act}", flush=True)
            else:
                pending_prompt[0] = ""
                nonlocal current_prompt
                current_prompt = act
                print(f"[VLA] 更新 Prompt: {current_prompt}", flush=True)
    t_sub = threading.Thread(target=_sub_worker, daemon=True)
    t_sub.start()

    t = 0
    steps = int(args.max_steps)
    while t < steps:
        all_device_images = []
        if cam_clients:
            ordered = []
            for name, cli in cam_clients:
                frame = cli.wait_for_frame(timeout=2.0)
                if frame is None:
                    continue
                img = frame["image"]
                aligned_img = preprocess_image_for_alignment(img, quality=90)
                final_img_bgr = cv2.resize(aligned_img, (224, 224), interpolation=cv2.INTER_AREA)
                final_img_rgb = cv2.cvtColor(final_img_bgr, cv2.COLOR_BGR2RGB)
                ordered.append((name, final_img_rgb))
            # enforce left, top, right order if available
            name_order = ["left", "top", "right"]
            for nm in name_order:
                for name, img in ordered:
                    if name == nm:
                        all_device_images.append(img)
                        break
        else:
            start_time = time.time()
            frames_by_serial = agg_client.collect_frames(time_window=0.05)
            if len(frames_by_serial) != 3:
                print("fail to collect all the frames", flush=True)
                all_device_images = []
            else:
                slot_imgs = {"left": None, "top": None, "right": None}
                for serial, f in frames_by_serial.items():
                    slot = serial_to_slot.get(str(serial))
                    if slot is None:
                        continue
                    img = f["image"]
                    aligned_img = preprocess_image_for_alignment(img, quality=90)
                    final_img_bgr = cv2.resize(aligned_img, (224, 224), interpolation=cv2.INTER_AREA)
                    final_img_rgb = cv2.cvtColor(final_img_bgr, cv2.COLOR_BGR2RGB)
                    slot_imgs[slot] = final_img_rgb

                all_device_images = []
                for slot in ["left", "top", "right"]:
                    img = slot_imgs[slot]
                    if img is None:
                        all_device_images = []
                        break
                    all_device_images.append(img)

        for idx, img in enumerate(all_device_images):
            window_name = f"Camera {idx}"
            cv2.imshow(window_name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(1) & 0xFF

        if key == 32:
            print("\n" + "=" * 40, flush=True)
            print("检测到空格键，推理已暂停。", flush=True)
            print(f"当前 Prompt: {current_prompt}", flush=True)
            paused[0] = True
            try:
                new_input = input("请输入新的 Prompt (直接回车保持不变): ").strip()
                if new_input:
                    current_prompt = new_input
                    print(f"Prompt 已更新为: {current_prompt}", flush=True)
                else:
                    print("输入为空，保持原有 Prompt。", flush=True)
            except KeyboardInterrupt:
                print("取消输入...", flush=True)
            paused[0] = False
            if pending_prompt[0]:
                current_prompt = pending_prompt[0]
                pending_prompt[0] = ""
                print(f"应用暂存 Prompt: {current_prompt}", flush=True)
            print("继续推理...", flush=True)
            print("=" * 40 + "\n", flush=True)

        if args.mode == "joint":
            actions_right = piper_right.GetArmJointMsgs().joint_state
            actions_left = piper_left.GetArmJointMsgs().joint_state
            actions_right_arr = np.array([
                actions_right.joint_1 / 1000.0,
                actions_right.joint_2 / 1000.0,
                actions_right.joint_3 / 1000.0,
                actions_right.joint_4 / 1000.0,
                actions_right.joint_5 / 1000.0,
                actions_right.joint_6 / 1000.0,
            ])
            actions_left_arr = np.array([
                actions_left.joint_1 / 1000.0,
                actions_left.joint_2 / 1000.0,
                actions_left.joint_3 / 1000.0,
                actions_left.joint_4 / 1000.0,
                actions_left.joint_5 / 1000.0,
                actions_left.joint_6 / 1000.0,
            ])
        else:
            actions_right = piper_right.GetArmEndPoseMsgs().end_pose
            actions_left = piper_left.GetArmEndPoseMsgs().end_pose
            actions_right_arr = np.array([
                actions_right.X_axis / 1000.0,
                actions_right.Y_axis / 1000.0,
                actions_right.Z_axis / 1000.0,
                actions_right.RX_axis / 1000.0,
                actions_right.RY_axis / 1000.0,
                actions_right.RZ_axis / 1000.0,
            ])
            actions_left_arr = np.array([
                actions_left.X_axis / 1000.0,
                actions_left.Y_axis / 1000.0,
                actions_left.Z_axis / 1000.0,
                actions_left.RX_axis / 1000.0,
                actions_left.RY_axis / 1000.0,
                actions_left.RZ_axis / 1000.0,
            ])

        gripper_right = piper_right.GetArmGripperMsgs().gripper_state.grippers_angle
        gripper_left = piper_left.GetArmGripperMsgs().gripper_state.grippers_angle
        gripper_right_arr = np.array([gripper_right / 1000.0])
        gripper_left_arr = np.array([gripper_left / 1000.0])
        current_observation_state = np.concatenate((
            actions_right_arr,
            gripper_right_arr,
            actions_left_arr,
            gripper_left_arr,
        ))
        # print("当前观测状态（双臂拼接）:", current_observation_state, flush=True)

        if len(all_device_images) < 3:
            continue
        obs = {
            "observation/left_image": all_device_images[0],
            "observation/top_image": all_device_images[1],
            "observation/right_image": all_device_images[2],
            "observation/state": current_observation_state,
            "prompt": current_prompt,
        }
        if current_prompt == "":
            time.sleep(0.5)
            continue
        print(f"current prompt: {current_prompt}", flush=True)
        action_chunk = infer_actions(obs, policy)
        t = piper_step_chunk_dual(piper_right, piper_left, action_chunk, t, mode=args.mode, n_steps=args.chunk_sizes)

    print("推理执行完成。", flush=True)
    try:
        for _, cli in cam_clients:
            cli.disconnect()
        if agg_client:
            agg_client.disconnect()
    except Exception:
        pass


if __name__ == "__main__":
    main()
