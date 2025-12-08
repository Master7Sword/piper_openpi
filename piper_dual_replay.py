import collections
import os
import cv2
import torch
import numpy as np
import time
from piper_sdk import *
import pyrealsense2 as rs

# 添加模块导入路径...
import sys
import glob
import h5py

# script_dir = os.path.dirname(__file__)
# project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
# sys.path.insert(0, project_root)
# sys.path.insert(0, os.path.join(project_root, 'mozihao', 'openpi', 'src')) 
# sys.path.insert(0, os.path.join(script_dir, 'act_tele'))

import time 
import argparse
from scipy.interpolate import interp1d


def piper_step_dual(piper_right, piper_left, action):
    """
    处理14维action，控制双臂：
    action[0:6]: can_arm1关节
    action[6]: can_arm1夹爪
    action[7:13]: can_arm2关节
    action[13]: can_arm2夹爪
    """
    start_time = time.time()
    try:
        joints_right = [round(x * 1000) for x in action[0:6]]
        gripper_right = round(action[6] * 1000)
        gripper_right = 0 if abs(gripper_right) < 40000 else gripper_right
        joints_left = [round(x * 1000) for x in action[7:13]]
        gripper_left = round(action[13] * 1000)
        gripper_left = 0 if abs(gripper_left) < 30000 else gripper_left

        piper_right.MotionCtrl_2(0x01, 0x01, 100, 0x00)
        piper_left.MotionCtrl_2(0x01, 0x01, 100, 0x00)

        piper_right.JointCtrl(*joints_right)
        piper_left.JointCtrl(*joints_left)

        piper_right.GripperCtrl(abs(gripper_right), 500, 0x01)
        piper_left.GripperCtrl(abs(gripper_left), 500, 0x01)

    except Exception as e:
        raise RuntimeError(f"Piper dual-arm command failed: {e}")
    
    # counter = 0
    # while (piper_right.GetArmStatus().arm_status.motion_status == 0x01 or
    #        piper_left.GetArmStatus().arm_status.motion_status == 0x01):
    #     time.sleep(0.0001)
    #     counter += 1
    #     if counter > 10000:
    #         print("Warning: Piper dual-arm motion taking too long.")
    #         break
    # print(f"counter: {counter}")

    elapsed = time.time() - start_time
    print(f'dual-arm actual fps: {1./elapsed:.4f}')

    fps = 60
    frame_duration = 1.0 / fps
    if elapsed < frame_duration:
        time.sleep(frame_duration - elapsed)


def piper_step_chunk_dual(piper_right, piper_left, action_chunk, t, n_steps=50):
    t = t + 1
    # if action_chunk.shape[0] < n_steps:
    #     orig_steps = action_chunk.shape[0]
    #     x_old = np.linspace(0, 1, orig_steps)
    #     x_new = np.linspace(0, 1, n_steps)
    #     interpolator = interp1d(x_old, action_chunk, axis=0, kind='linear')
    #     action_chunk_interp = interpolator(x_new)
    # elif action_chunk.shape[0] > n_steps:
    #     action_chunk_interp = action_chunk[::5]
    # else:
    #     action_chunk_interp = action_chunk

    action = action_chunk
    # print(f'Dual actions: {action}')
    # time.sleep(1)
    piper_step_dual(piper_right, piper_left, action)

    counter = 0
    temp = time.time()
    while (piper_right.GetArmStatus().arm_status.motion_status == 0x01 or
           piper_left.GetArmStatus().arm_status.motion_status == 0x01):
        time.sleep(0.0001)
        counter += 1
        if counter >= 250:
            print("Warning: Piper dual-arm motion taking too long.")
            break
    print(f"sleep time: {time.time() - temp:.4f}")

    return t


def main(args):

    # Replay mode: use pre-saved aligned data and images instead of live camera/robot
    if args.replay_episode_dir is not None:
        replay_dir = args.replay_episode_dir
        print(f"Running in replay mode with directory: {replay_dir}")

        # Load aligned robot states H5 file
        aligned_h5_path = os.path.join(replay_dir, "robot_data_aligned.h5")
        f_robot = h5py.File(aligned_h5_path, 'r')
        joints = f_robot['joints'][:]

        # Load all replay frame image paths
        cam0_images = sorted(glob.glob(os.path.join(replay_dir, "frames", "cam0", "*.jpg")))
        cam1_images = sorted(glob.glob(os.path.join(replay_dir, "frames", "cam1", "*.jpg")))
        cam2_images = sorted(glob.glob(os.path.join(replay_dir, "frames", "cam2", "*.jpg")))

        num_frames = joints.shape[0]
        print(f"Loaded {num_frames} frames from replay data")

        # Init Piper dual arms
        piper_right = C_PiperInterface_V2("can_arm1")
        piper_left = C_PiperInterface_V2("can_arm2")
        piper_right.ConnectPort()
        piper_left.ConnectPort()
        while not (piper_right.EnablePiper() and piper_left.EnablePiper()):
            time.sleep(0.01)
        print("Piper双臂已连接并启动。")
        piper_right.GripperCtrl(0,1000,0x02, 0)
        piper_right.GripperCtrl(0,1000,0x01, 0)
        piper_left.GripperCtrl(0,1000,0x02, 0)
        piper_left.GripperCtrl(0,1000,0x01, 0)

        # Initialize arms to startup pose
        piper_right.MotionCtrl_2(0x01, 0x01, 50, 0x00)
        piper_left.MotionCtrl_2(0x01, 0x01, 50, 0x00)
        piper_right.JointCtrl(41920, 49997, -74840, -3245, 47584, -2760)
        piper_left.JointCtrl(-24171, 14878, -4253, -27609, -8485, 17650)
        piper_right.GripperCtrl(abs(100), 500, 0x01, 0)
        piper_left.GripperCtrl(abs(100), 500, 0x01, 0)
        print("Piper双臂初始化完成。")

        t = 0
        while t < num_frames:
            # Load synced images from replay frames
            # left_img = cv2.imread(cam0_images[t])
            # top_img = cv2.imread(cam1_images[t])
            # right_img = cv2.imread(cam2_images[t])
            # left_img = cv2.resize(left_img, (224, 224))
            # top_img = cv2.resize(top_img, (224, 224))
            # right_img = cv2.resize(right_img, (224, 224))

            # print(f"Replay frame {t} observation state:", joints[t])

            action_chunk = np.array(joints[t])
            # print(action_chunk.shape)

            t = piper_step_chunk_dual(piper_right, piper_left, action_chunk, t, n_steps=1)

        f_robot.close()
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run infer_piper.py with custom directories.")
    parser.add_argument('--replay_episode_dir', type=str, default=None, help='Path to replay episode directory for offline replay')
    args = parser.parse_args()
    whole_time = time.time()
    main(args)
    print(f"whole time: {time.time() - whole_time:.4f}")
