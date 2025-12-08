import collections
import os
import cv2
import torch
import numpy as np
import time
from piper_sdk import *
import pyrealsense2 as rs

# Add module import paths
import sys
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'mozihao', 'openpi', 'src')) # <-- pi0 path
sys.path.insert(0, os.path.join(script_dir, 'act_tele'))

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
import time 
import argparse
import time
from scipy.interpolate import interp1d


def load_policy(checkpoint_dir, config_name):
    print(f"Loading OpenPI policy from checkpoint: {checkpoint_dir}")
    config = _config.get_config(config_name)
    policy = _policy_config.create_trained_policy(config, checkpoint_dir)
    print("Policy loaded.")
    return policy


def infer_actions(obs, policy):
    with torch.no_grad():
        action = policy.infer(obs)['actions']
    return action


def piper_step(piper, action):
    joint_0, joint_1, joint_2, joint_3, joint_4, joint_5, joint_6 = action[0], action[1], action[2], action[3], action[4], action[5], action[6]
    start_time = time.time()

    try:
        joint_0 = round(joint_0 * 1000)
        joint_1 = round(joint_1 * 1000)
        joint_2 = round(joint_2 * 1000)
        joint_3 = round(joint_3 * 1000)
        joint_4 = round(joint_4 * 1000)
        joint_5 = round(joint_5 * 1000)
        joint_6 = round(joint_6 * 1000)

        piper.MotionCtrl_2(0x01, 0x01, 10, 0x00)
        piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
        piper.GripperCtrl(abs(joint_6), 100, 0x01, 0)
    except Exception as e:
        raise RuntimeError(f"Piper command failed: {e}")
    
    while piper.GetArmStatus().arm_status.motion_status == 0x01:
        time.sleep(0.0001)  # wait for motion to complete

    elapsed = time.time() - start_time
    print(f'actual fps: {1./elapsed:.4f}')

    fps = 10
    frame_duration = 1.0 / fps
    if elapsed < frame_duration:
        time.sleep(frame_duration - elapsed)


def piper_step_chunk(piper, action_chunk, t, n_steps = 50):
    assert action_chunk.shape[0] >= n_steps
    t = t + n_steps
        
    action_chunk_interp = action_chunk[:n_steps]

    for i in range(action_chunk_interp.shape[0]):
        action = action_chunk_interp[i]
        print(f'Actions: {action}')
        piper_step(piper, action)

    while piper.GetArmStatus().arm_status.motion_status == 0x01:
        time.sleep(0.0001)  # wait for motion to complete

    return t


def wait_key():
    try:
        import msvcrt
        return msvcrt.getch().decode('utf-8').lower()
    except ImportError:
        import sys
        import tty
        import termios

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1).lower()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch
        

def main(args, chunk_sizes = 10):

    max_steps = 1000
    base_save_dir = os.path.join("rollouts", "piper_infer")
    len_prev_actions = args.prev_actions

    # initialize cameras
    context = rs.context()
    devices = context.query_devices()
    serial_numbers = []
    for dev in devices:
        serial = dev.get_info(rs.camera_info.serial_number)
        serial_numbers.append(serial)

    if not serial_numbers:
        print("错误：未找到 RealSense 相机！")
        return

    print(f"找到 {len(serial_numbers)} 台相机:")
    for serial in serial_numbers:
        print(f"  - {serial}")

    pipelines = {}
    configs = {}
    aligns = {}

    width, height, fps = 640, 480, 60

    for serial in serial_numbers:
        pipeline = rs.pipeline(context) # 使用同一个 context
        config = rs.config()
        
        # 告诉 config 只使用这个序列号的相机
        config.enable_device(serial) 
        
        # 配置流
        try:
            config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            
            pipelines[serial] = pipeline
            configs[serial] = config
            
            # 每个 pipeline 都需要自己的 align 对象
            aligns[serial] = rs.align(rs.stream.color) 
        except Exception as e:
            print(f"警告：无法为相机 {serial} 配置流（{width}x{height} @ {fps}）。")
            print(f"  错误: {e}")
            # 如果配置失败，从列表中移除
            if serial in serial_numbers: serial_numbers.remove(serial)

    active_serials = [] # 只保留成功启动的相机
    for serial in serial_numbers:
        print(f"正在启动相机: {serial}...")
        try:
            pipelines[serial].start(configs[serial])
            active_serials.append(serial) # 添加到活动列表
            print(f"相机 {serial} 启动成功。")
        except Exception as e:
            print(f"错误：无法启动相机 {serial}。{e}")
            # 清理失败的 pipeline
            if serial in pipelines: del pipelines[serial]
            if serial in configs: del configs[serial]
            if serial in aligns: del aligns[serial]
        
    if not active_serials:
        print("没有相机成功启动。")
        return

    print(f"\n成功启动 {len(active_serials)} 台相机")

    # initialize piper
    piper = C_PiperInterface_V2("can0")
    piper.ConnectPort()
    while( not piper.EnablePiper()):
        time.sleep(0.01)
    print("Piper connected and enabled.")
    piper.MotionCtrl_2(0x01, 0x01, 70, 0x00)
    # piper.JointCtrl(65100, 90553, -75186, -24983, 70009, 65587)
    piper.JointCtrl(46635, 91114, -81951, -17100, 70009, 47623)
    piper.GripperCtrl(abs(100), 100, 0x01, 0)
    print("Piper moved to initial position.")

    # load model
    policy = load_policy(args.checkpoint_dir, args.config_name)  

    origin_observation_state=0
    for i in range(args.rollouts_num):
        t = 0
        # action_plan = collections.deque()
        action_plan = []
        history_actions = []
        while t < max_steps:

            if  not action_plan:
                # read observations from camera
                all_device_images = []
                for serial in active_serials:
                    pipeline = pipelines[serial]
                    align = aligns[serial]
                    
                    success, frames = pipeline.try_wait_for_frames() 
                    if not success:
                        continue 

                    aligned_frames = align.process(frames)
                    color_frame = aligned_frames.get_color_frame()
                    if not color_frame:
                        continue

                    color_frame = np.asanyarray(color_frame.get_data())
                    all_device_images.append(color_frame)

                # # Show images from all cameras
                for idx, img in enumerate(all_device_images):
                    window_name = f"Camera {idx}"
                    cv2.imshow(window_name, img)
                cv2.waitKey(2)

                # read observations from robot
                joint_states = piper.GetArmJointMsgs().joint_state
                joint_states = np.array((
                    joint_states.joint_1 / 1000.0,
                    joint_states.joint_2 / 1000.0,
                    joint_states.joint_3 / 1000.0,
                    joint_states.joint_4 / 1000.0,
                    joint_states.joint_5 / 1000.0,
                    joint_states.joint_6 / 1000.0,
                ))
                gripper_states = np.array(piper.GetArmGripperMsgs().gripper_state.grippers_angle / 1000.0).reshape(1)
                # print("Joint states:", joint_states.shape, "Gripper state:", gripper_states.shape)
                if t ==0: 
                    origin_observation_state = np.concatenate((joint_states, gripper_states), axis=0)
                # state = np.concatenate((joint_states, gripper_states), axis=0)
                # print("Obs state:", state)

                current_observation_state = np.concatenate((joint_states, gripper_states), axis=0)
                print("Current observation state:", current_observation_state)
                if t == 0:
                    # history_action = np.zeros((1,32))
                    pad = np.zeros((len_prev_actions, 32), dtype=np.float32)
                    obs_state_padded = np.zeros(32, dtype=np.float32)
                    obs_state_padded[:origin_observation_state.shape[0]] = origin_observation_state
                    pad[:] = obs_state_padded
                    history_action = pad
                elif t < len_prev_actions:
                    pad_len = len_prev_actions - t
                    # Pad observation_state to shape (pad_len, 32)
                    pad = np.zeros((pad_len, 32), dtype=np.float32)
                    obs_state_padded = np.zeros(32, dtype=np.float32)
                    obs_state_padded[:origin_observation_state.shape[0]] = origin_observation_state
                    pad[:] = obs_state_padded
                    print(f"pad shape {pad.shape}")
                    history_action = np.concatenate(
                        (pad, np.array(history_actions).reshape(-1, 32)),
                        axis=0
                )
                else:   
                    history_action = np.array(history_actions[-len_prev_actions:]).reshape(-1, 32)

                obs = {
                    'observation/wrist_image': cv2.resize(all_device_images[0], (224, 224)),
                    'observation/image': cv2.resize(all_device_images[1], (224, 224)),
                    'observation/state': current_observation_state,
                    # 'prev_actions': history_action,
                    'prompt': "sequentially touch the yellow, blue and red blocks",
                }

                action_chunk = infer_actions(obs, policy)
                print(f"Inferred action chunk shape: {action_chunk.shape, type(action_chunk)}")
                # action_plan.extend(action_chunk[: chunk_sizes])
                # Pad each action in action_chunk to length 32 before appending to history_actions
                for act in action_chunk[:chunk_sizes]:
                    act_padded = np.zeros(32, dtype=np.float32)
                    act_padded[:len(act)] = act
                    history_actions.append(act_padded)

            # action = action_plan.popleft()


            # print(f"Step {t+1}/{max_steps}, Action: {action}")
            # with open("actions.txt", "a") as f:
            #     f.write(",".join(map(str, action)) + "\n")
            
            # piper_step(piper, action)
            t = piper_step_chunk(piper, action_chunk[:chunk_sizes]/2, t, n_steps=chunk_sizes)

            # t = t+1
        
        print(f"Rollout {i+1}/{args.rollouts_num} completed.")
        time.sleep(5.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run infer_piper.py with custom directories.")
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Path to the checkpoint directory')
    parser.add_argument('--dataset_root', type=str, required=True, help='Path to the dataset root')
    parser.add_argument('--config_name', type=str, default='pi0_rlbench', help='Config name (default: pi0_rlbench)')
    parser.add_argument('--rollouts_num', type=int, default=10, help='Number of rollouts (default: 1)')
    parser.add_argument('--prev_actions', type=int, default=50, help='Number of previous actions to consider (default: 50)')
    args = parser.parse_args()


    chunk_sizes = 5

    main(args, chunk_sizes)