import os
import h5py
import argparse
import numpy as np

from utils import *


def main(args):
    if args.replay_episode_dir is not None:
        replay_dir = args.replay_episode_dir
        print(f"Running in replay mode with directory: {replay_dir}")

        aligned_h5_path = os.path.join(replay_dir, "robot_data_aligned.h5")
        f_robot = h5py.File(aligned_h5_path, 'r')
        joints = f_robot['joints'][:]

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
            action_chunk = np.array(joints[t])
            # print(action_chunk.shape)
            t = piper_step_chunk_dual(piper_right, piper_left, action_chunk, t)

        f_robot.close()
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run infer_piper.py with custom directories.")
    parser.add_argument('--replay_episode_dir', type=str, default=None, help='Path to replay episode directory for offline replay')
    args = parser.parse_args()
    whole_time = time.time()
    main(args)
    print(f"whole time: {time.time() - whole_time:.4f}")
