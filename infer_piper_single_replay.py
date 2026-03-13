import numpy as np
import time 
import argparse
from utils import *
import pyrealsense2 as rs
import os
import glob
import h5py

np.set_printoptions(precision=2)


def main(args, chunk_sizes, prompt, max_steps=5000, len_prev_actions=25):

    # context = rs.context()
    # devices = context.query_devices()
    # serial_numbers = [dev.get_info(rs.camera_info.serial_number) for dev in devices]

    # if not serial_numbers:
    #     print("错误：未找到 RealSense 相机！")
    #     return

    # print(f"找到 {len(serial_numbers)} 台相机:")
    # for serial in serial_numbers:
    #     print(f"  - {serial}")

    # 初始化双臂piper接口
    # piper_right = C_PiperInterface_V2("can_arm1")
    piper_left = C_PiperInterface_V2("can_arm2")
    # piper_right.ConnectPort()
    piper_left.ConnectPort()
    while not (piper_left.EnablePiper()):
        time.sleep(0.01)
    print("Piper左臂已连接并启动。")

    # 双臂初始位置（关节和夹爪）
    # piper_right.MotionCtrl_2(0x01, 0x01, 30, 0x00)
    piper_left.MotionCtrl_2(0x01, 0x01, 40, 0x00)
    # piper_right.JointCtrl(46635, 91114, -81951, -16699, 69789, 47623)
    piper_left.JointCtrl(-30333, 1339, -5378, 781, 11710, -2983) # open and close drawer
    # piper_right.GripperCtrl(abs(0), 100, 0x01, 0)
    piper_left.GripperCtrl(abs(0), 500, 0x01, 0)
    print("Piper臂初始化完成。")

    # 加载模型
    policy = load_policy(args.checkpoint_dir, args.config_name)

    current_prompt = prompt
    print(f"\n当前 Prompt: {current_prompt}")
    print("[caution] 在相机窗口激活时按【空格键】可修改 Prompt")

    t = 50
    history_actions = []
    # len_prev_actions = 25
    while t < max_steps:
        all_device_images = []


        for idx, img in enumerate(all_device_images):
            window_name = f"Camera {idx}"
            # cv2.imshow(window_name, img)
            cv2.imshow(window_name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(1) & 0xFF
        
        if key == 32: # 32 是空格键的 ASCII 码
            print("\n" + "="*40)
            print("检测到空格键，推理已暂停。")
            print(f"当前 Prompt: {current_prompt}")
            
            # 使用 input 会阻塞程序，直到用户在终端输入回车
            # 这正好满足了等待用户输入后再继续的需求
            try:
                new_input = input("请输入新的 Prompt (直接回车保持不变): ").strip()
                if new_input:
                    current_prompt = new_input
                    print(f"Prompt 已更新为: {current_prompt}")
                else:
                    print("输入为空，保持原有 Prompt。")
            except KeyboardInterrupt:
                print("取消输入...")
            
            print("继续推理...")
            print("="*40 + "\n")

        elif key == ord('r'):
            print("重置dan臂位置...")
            time.sleep(1)
            # piper_right.MotionCtrl_2(0x01, 0x01, 40, 0x00)
            piper_left.MotionCtrl_2(0x01, 0x01, 40, 0x00)
            # piper_right.JointCtrl(46635, 91114, -81951, -16699, 69789, 47623)
            piper_left.JointCtrl(-30333, 1339, -5378, 781, 11710, -2983) # open and close drawer
            # piper_right.GripperCtrl(abs(0), 500, 0x01, 0)
            piper_left.GripperCtrl(abs(0), 500, 0x01, 0)
            time.sleep(1)
            print("臂已重置。")
            history_actions = []
            t = 0


        # actions_right = piper_right.GetArmJointMsgs().joint_state
        actions_left = piper_left.GetArmJointMsgs().joint_state

        # actions_right_arr = np.array([
        #     actions_right.joint_1 / 1000.0,
        #     actions_right.joint_2 / 1000.0,
        #     actions_right.joint_3 / 1000.0,
        #     actions_right.joint_4 / 1000.0,
        #     actions_right.joint_5 / 1000.0,
        #     actions_right.joint_6 / 1000.0,
        # ])
        actions_left_arr = np.array([   
            actions_left.joint_1 / 1000.0,
            actions_left.joint_2 / 1000.0,
            actions_left.joint_3 / 1000.0,
            actions_left.joint_4 / 1000.0,
            actions_left.joint_5 / 1000.0,
            actions_left.joint_6 / 1000.0,
        ])

        # gripper_right = piper_right.GetArmGripperMsgs().gripper_state.grippers_angle
        gripper_left = piper_left.GetArmGripperMsgs().gripper_state.grippers_angle
        # gripper_right_arr = np.array([gripper_right / 1000.0])
        gripper_left_arr = np.array([gripper_left / 1000.0])
        if t ==0: 
            # origin_observation_state = np.concatenate((
                # actions_right_arr, gripper_right_arr), axis=0)
            origin_observation_state = np.concatenate((
                actions_left_arr, gripper_left_arr), axis=0)

        # current_observation_state = np.concatenate((
        #     actions_right_arr,
        #     gripper_right_arr,
        # ))
        current_observation_state = np.concatenate((
            actions_left_arr,
            gripper_left_arr,
        ))

        print("当前观测状态:", current_observation_state)


        # 构建观测字典
        replay_dir = args.replay_episode_dir
        print(f"Running in replay mode with directory: {replay_dir}")

        # Load aligned robot states H5 file
        aligned_h5_path = os.path.join(replay_dir, "robot_data_aligned.h5")
        f_robot = h5py.File(aligned_h5_path, 'r')
        joints = f_robot['joints'][:]

        # Load all replay frame image paths
        cam0_images = sorted(glob.glob(os.path.join(replay_dir, "frames", "cam0", "*.jpg")))
        cam1_images = sorted(glob.glob(os.path.join(replay_dir, "frames", "cam1", "*.jpg")))
        # cam2_images = sorted(glob.glob(os.path.join(replay_dir, "frames", "cam2", "*.jpg")))

        num_frames = joints.shape[0]
        print(f"Loaded {num_frames} frames from replay data")

        # if t==50:
        #     piper_right.MotionCtrl_2(0x01, 0x01, 50, 0x00)
        #     actions = [round(x * 1000) for x in joints[t][0:6]]
        #     piper_right.JointCtrl(*actions)


        left_img = cv2.imread(cam0_images[t])
        top_img = cv2.imread(cam1_images[t])
        # right_img = cv2.imread(cam1_images[t])
        left_img = cv2.resize(left_img, (224, 224))
        top_img = cv2.resize(top_img, (224, 224))
        # right_img = cv2.resize(right_img, (224, 224))

        left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
        top_img = cv2.cvtColor(top_img, cv2.COLOR_BGR2RGB)
        # right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)


        if t==0:
            origin_observation_state = joints[0]
            print('origin_observation_state:', origin_observation_state)


        if t < len_prev_actions:
            origin_observation_state = joints[0]
            print('origin_observation_state shape:', origin_observation_state.shape)
            pad_len = len_prev_actions - t
            # Pad observation_state to shape (pad_len, 32)
            pad = np.zeros((pad_len, 32), dtype=np.float32)
            obs_state_padded = np.zeros(32, dtype=np.float32)
            obs_state_padded[:origin_observation_state.shape[0]] = origin_observation_state
            pad[:] = obs_state_padded
            history_action = np.concatenate((pad, np.array(history_actions).reshape(-1, 32)), axis=0)
        else:
            history_action = joints[max(0, t - len_prev_actions):t]

        print('history_action shape:', history_action.shape)
        

        print(f"Replay frame {t} observation state:", joints[t])
        print(f"Replay actually  observation state:", current_observation_state)
        current_observation_state = joints[t]

        obs = {
            'observation/left_image': left_img,
            'observation/top_image': top_img,
            # 'observation/right_image': right_img,
            'observation/state': current_observation_state,
            'prompt': current_prompt,
            # 'prev_actions': history_action,
        }


        action_chunk = infer_actions(obs, policy)
        print(f"推理动作块形状: {action_chunk.shape}, 类型: {type(action_chunk)}")
        history_actions.extend(action_chunk[:chunk_sizes])
        # print(f"历史动作长度: {len(history_actions)}")

        print(f"执行动作块..., t =", t)
        t = piper_step_chunk_single(piper_left, action_chunk, t, mode=args.mode, n_steps=chunk_sizes)
        print(f"当前步骤 t: {t}\n")

    print(f"推理执行完成。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run infer_piper.py with custom directories.")
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Path to the checkpoint directory')
    parser.add_argument('--config_name', type=str, required=True, help='Config name')
    parser.add_argument('--mode', type=str, required=True, choices=['joint', 'ee'], help='Control mode: joint or ee (default: joint)')
    parser.add_argument('--replay_episode_dir', type=str, required=True, help='Directory containing replay episode data')
    args = parser.parse_args()

    chunk_sizes = 20
    history_length = 50
    max_steps = 10000
    prompt = "open and close the drawer"

    main(args, chunk_sizes, prompt, max_steps, history_length)
