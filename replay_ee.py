import os
import numpy as np
import time
from piper_sdk import *
import h5py
import time 
import argparse


def piper_step_dual(piper_right, piper_left, end_pose_t):
    """
    处理14维action，控制双臂：
    action[0:6]: can_arm1关节
    action[6]: can_arm1夹爪
    action[7:13]: can_arm2关节
    action[13]: can_arm2夹爪
    """
    start_time = time.time()
    try:
        end_pose_right = [round(x * 1000) for x in end_pose_t[0:6]]
        gripper_right = round(end_pose_t[6] * 1000)
        end_pose_left = [round(x * 1000) for x in end_pose_t[7:13]]
        gripper_left = round(end_pose_t[13] * 1000)

        gripper_right = 0 if abs(gripper_right) < 40000 else gripper_right
        gripper_left = 0 if abs(gripper_left) < 30000 else gripper_left

        piper_right.MotionCtrl_2(0x01, 0x00, 20, 0x00)
        piper_left.MotionCtrl_2(0x01, 0x00, 20, 0x00)

        piper_right.EndPoseCtrl(*end_pose_right)
        piper_left.EndPoseCtrl(*end_pose_left)

        piper_right.GripperCtrl(abs(gripper_right), 500, 0x01)
        piper_left.GripperCtrl(abs(gripper_left), 500, 0x01)

    except Exception as e:
        raise RuntimeError(f"Piper dual-arm command failed: {e}")

    elapsed = time.time() - start_time
    print(f'dual-arm actual fps: {1./elapsed:.4f}')

    fps = 30
    frame_duration = 1.0 / fps
    if elapsed < frame_duration:
        time.sleep(frame_duration - elapsed)


def piper_step_chunk_dual(piper_right, piper_left, end_pose_t, t):
    t = t + 1

    # print(f'Dual actions: {action}')
    # time.sleep(1)
    piper_step_dual(piper_right, piper_left, end_pose_t)

    counter = 0
    temp = time.time()
    while (piper_right.GetArmStatus().arm_status.motion_status == 0x01 or
           piper_left.GetArmStatus().arm_status.motion_status == 0x01):
        time.sleep(0.0001)
        counter += 1
        if counter >= 500:
            print("Warning: Piper dual-arm motion taking too long.")
            break
    print(f"sleep time: {time.time() - temp:.4f}")

    return t


def main(args):
    if args.replay_episode_dir is not None:
        replay_dir = args.replay_episode_dir
        print(f"Running in replay mode with directory: {replay_dir}")

        aligned_h5_path = os.path.join(replay_dir, "robot_data_aligned.h5")
        f_robot = h5py.File(aligned_h5_path, 'r')
        end_pose = f_robot['end_pose'][:]  

        num_frames = end_pose.shape[0]
        print(num_frames)
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
            end_pose_t = np.array(end_pose[t])
            t = piper_step_chunk_dual(piper_right, piper_left, end_pose_t, t, n_steps=1)

        f_robot.close()
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run infer_piper.py with custom directories.")
    parser.add_argument('--replay_episode_dir', type=str, default=None, help='Path to replay episode directory for offline replay')
    args = parser.parse_args()
    whole_time = time.time()
    main(args)
    print(f"whole time: {time.time() - whole_time:.4f}")
