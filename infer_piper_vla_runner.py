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
import select
sys.path.append("/home/tengenx2204/workspace/cxp/robobrain/robobrain")
from camera.camera_client import CameraClient

def _parse_vlm_message(msg: str) -> tuple[str, str]:
    s = (msg or "").strip()
    try:
        i = s.find("{")
        j = s.rfind("}")
        if i != -1 and j != -1 and j > i:
            obj = json.loads(s[i : j + 1])
            action = obj.get("action", "")
            direct_action = obj.get("direct action", "") or obj.get("direct_action", "")
            action = action.strip() if isinstance(action, str) else ""
            direct_action = direct_action.strip() if isinstance(direct_action, str) else ""
            return action, direct_action
    except Exception:
        pass
    return "", ""


def main():
    parser = argparse.ArgumentParser(description="VLA ZMQ executor with dual-arm Piper")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--config_name", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True, choices=["joint", "ee"])
    parser.add_argument("--sub-host", type=str, default="localhost")
    parser.add_argument("--sub-port", type=int, default=6002)
    parser.add_argument("--sub-topic", type=str, default="vlm")
    parser.add_argument("--pub-host", type=str, default="0.0.0.0")
    parser.add_argument("--pub-port", type=int, default=6003)
    parser.add_argument("--pub-topic", type=str, default="vla_state")
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
    if len(cam_clients) == 0:
        raise RuntimeError("VLA 需要指定 camera_left_serial/camera_top_serial/camera_right_serial 参数")

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

    def _warmup_vla():
        warmup_prompt = "warmup"
        max_retries = 50
        retry = 0
        while retry < max_retries:
            all_device_images = []
            ordered = []
            for name, cli in cam_clients:
                frame = cli.wait_for_frame(timeout=0.05)
                if frame is None:
                    continue
                img = frame["image"]
                aligned_img = preprocess_image_for_alignment(img, quality=90)
                final_img_bgr = cv2.resize(aligned_img, (224, 224), interpolation=cv2.INTER_AREA)
                final_img_rgb = cv2.cvtColor(final_img_bgr, cv2.COLOR_BGR2RGB)
                ordered.append((name, final_img_rgb))
            name_order = ["left", "top", "right"]
            for nm in name_order:
                for name, img in ordered:
                    if name == nm:
                        all_device_images.append(img)
                        break
            if len(all_device_images) != 3:
                retry += 1
                time.sleep(0.05)
                continue

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

            obs = {
                "observation/left_image": all_device_images[0],
                "observation/top_image": all_device_images[1],
                "observation/right_image": all_device_images[2],
                "observation/state": current_observation_state,
                "prompt": warmup_prompt,
            }

            start_time = time.time()
            infer_actions(obs, policy)
            print(f"VLA 预热推理完成，耗时: {time.time() - start_time:.3f}s", flush=True)
            return True

        print("VLA 预热失败：未获取到全部三路相机帧，跳过预热。", flush=True)
        return False

    print("VLA 正在预热...", flush=True)
    _warmup_vla()
    print("VLA 已准备就绪。", flush=True)

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

    pub_socket = sub_ctx.socket(zmq.PUB)
    pub_addr = f"tcp://{args.pub_host}:{args.pub_port}"
    pub_socket.bind(pub_addr)
    print(f"VLA 状态上报: {pub_addr} topic={args.pub_topic}", flush=True)

    vla_state = ["idle"]
    current_plan_step_id = [None]

    def _publish_state(state: str):
        if state == vla_state[0]:
            return
        vla_state[0] = state
        payload = {"state": state, "timestamp": time.time()}
        try:
            pub_socket.send_multipart(
                [
                    args.pub_topic.encode("utf-8"),
                    json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                ]
            )
        except Exception:
            pass

    def _publish_plan_step_done(step_id, status: str = "done"):
        try:
            sid = int(step_id)
        except Exception:
            return
        payload = {
            "state": vla_state[0],
            "plan_step_id": sid,
            "plan_step_status": status,
            "timestamp": time.time(),
        }
        try:
            pub_socket.send_multipart(
                [
                    args.pub_topic.encode("utf-8"),
                    json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                ]
            )
            print(f"手动标记计划步骤完成: {sid}", flush=True)
        except Exception:
            pass

    stop_event = threading.Event()

    def _sub_worker():
        nonlocal current_prompt
        while True:
            if stop_event.is_set():
                break
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
            assistant_message = data.get("assistant_message")
            if isinstance(assistant_message, str) and assistant_message.strip():
                assistant_message = assistant_message.strip()
                act, direct_action = _parse_vlm_message(assistant_message)
                if not act and not direct_action:
                    continue
                if direct_action:
                    direct_norm = direct_action.lower()
                    if direct_norm == "stop":
                        _publish_state("paused")
                        current_prompt = ""
                        current_plan_step_id[0] = None
                        print("收到紧急停止指令，已暂停执行。", flush=True)
                        continue
                    else:
                        print(f"收到 direct action: {direct_action}", flush=True)
                if not act:
                    continue
                current_prompt = act
                current_plan_step_id[0] = None
                _publish_state("running")
                print(f"更新 Prompt: {current_prompt}", flush=True)
                continue
            act2 = str(data.get("action") or "").strip()
            direct_action2 = str(data.get("direct_action") or data.get("direct action") or "").strip()
            if not act2 and not direct_action2:
                continue
            if direct_action2:
                direct_norm = direct_action2.lower()
                if direct_norm == "stop":
                    _publish_state("paused")
                    current_prompt = ""
                    current_plan_step_id[0] = None
                    print("收到紧急停止指令，已暂停执行。", flush=True)
                    continue
                else:
                    print(f"收到 direct action: {direct_action2}", flush=True)
            if not act2:
                continue
            current_prompt = act2
            plan_step_id = data.get("plan_step_id")
            try:
                current_plan_step_id[0] = int(plan_step_id) if plan_step_id is not None else None
            except Exception:
                current_plan_step_id[0] = None
            _publish_state("running")
            print(f"更新 Prompt: {current_prompt}", flush=True)
    t_sub = threading.Thread(target=_sub_worker, daemon=True)
    t_sub.start()

    t = 0
    steps = int(args.max_steps)
    try:
        while t < steps:
            if stop_event.is_set():
                break
            if vla_state[0] == "paused":
                time.sleep(0.05)
                continue
            all_device_images = []
            ordered = []
            start_time = time.time()
            for name, cli in cam_clients:
                frame = cli.wait_for_frame(timeout=0.05)
                if frame is None:
                    continue
                img = frame["image"]
                aligned_img = preprocess_image_for_alignment(img, quality=90)
                final_img_bgr = cv2.resize(aligned_img, (224, 224), interpolation=cv2.INTER_AREA)
                final_img_rgb = cv2.cvtColor(final_img_bgr, cv2.COLOR_BGR2RGB)
                ordered.append((name, final_img_rgb))
            name_order = ["left", "top", "right"]
            for nm in name_order:
                for name, img in ordered:
                    if name == nm:
                        all_device_images.append(img)
                        break
            if len(all_device_images) != 3:
                print("[VLA] 未获取到全部三路相机帧，本轮跳过控制", flush=True)
                t += 1
                continue

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
                _publish_state("idle")
                time.sleep(0.5)
                continue
            action_chunk = infer_actions(obs, policy)
            t = piper_step_chunk_dual(piper_right, piper_left, action_chunk, t, mode=args.mode, n_steps=args.chunk_sizes)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        print("推理执行完成。", flush=True)
        try:
            for _, cli in cam_clients:
                cli.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
