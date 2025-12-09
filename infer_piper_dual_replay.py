import os
import glob
import h5py
import argparse

from utils import *

import numpy as np
np.set_printoptions(precision=2)


def main(args, chunk_sizes=10):
    if args.replay_episode_dir is not None:
        replay_dir = args.replay_episode_dir
        print(f"Running in replay mode with directory: {replay_dir}")

        # Load aligned robot states H5 file
        aligned_h5_path = os.path.join(replay_dir, "robot_data_aligned.h5")
        f_robot = h5py.File(aligned_h5_path, 'r')
        joints = f_robot['joints'][:]
        end_pose = f_robot['end_pose'][:]  # may be used later if needed

        # Load all replay frame image paths
        cam0_images = sorted(glob.glob(os.path.join(replay_dir, "frames", "cam0", "*.jpg")))
        cam1_images = sorted(glob.glob(os.path.join(replay_dir, "frames", "cam1", "*.jpg")))
        cam2_images = sorted(glob.glob(os.path.join(replay_dir, "frames", "cam2", "*.jpg")))

        num_frames = joints.shape[0]
        print(f"Loaded {num_frames} frames from replay data")

        # Init Piper dual arms
        piper_right = C_PiperInterface_V2("can_arm1")
        piper_left = C_PiperInterface_V2("can_arm2")
        # piper_right = None
        # piper_left = None
        piper_right.ConnectPort()
        piper_left.ConnectPort()
        while not (piper_right.EnablePiper() and piper_left.EnablePiper()):
            time.sleep(0.01)
        print("Piper双臂已连接并启动。")

        # Initialize arms to startup pose
        piper_right.MotionCtrl_2(0x01, 0x01, 50, 0x00)
        piper_left.MotionCtrl_2(0x01, 0x01, 50, 0x00)
        # piper_right.JointCtrl(41913, 50004, -74840, -3245, 47584, -2760)
        # piper_left.JointCtrl(-36499, 299, -3872, -54116, 5961, 54201) # open drawer
        # piper_right.JointCtrl(41920, 49997, -74840, -3245, 47584, -2760)
        # piper_left.JointCtrl(-24171, 14878, -4253, -27609, -8485, 17650) # open & close drawer
        piper_right.JointCtrl(40036, 3630, -9526, -3140, 17670, -12043)
        piper_left.JointCtrl(-40543, 177, -104, -87029, -5647, 77959) #  put item in drawer
        piper_right.GripperCtrl(abs(0), 100, 0x01, 0)
        piper_left.GripperCtrl(abs(0), 100, 0x01, 0)
        print("Piper双臂初始化完成。")

        # Load trained policy model
        policy = load_policy(args.checkpoint_dir, args.config_name)

        for i in range(args.rollouts_num):
            t = 0
            while t < num_frames:
                left_img = cv2.imread(cam0_images[t])
                top_img = cv2.imread(cam1_images[t])
                right_img = cv2.imread(cam2_images[t])
                left_img = cv2.resize(left_img, (224, 224))
                top_img = cv2.resize(top_img, (224, 224))
                right_img = cv2.resize(right_img, (224, 224))

                current_observation_state = joints[t]

                print(f"Replay frame {t} observation state:", current_observation_state)

                obs = {
                    'observation/left_image': left_img,
                    'observation/top_image': top_img,
                    'observation/right_image': right_img,
                    'observation/state': current_observation_state,
                    'prompt': "open drawer, put the yellow block in drawer and close drawer",
                    # 'prompt' : "open drawer then close drawer"
                }

                action_chunk = infer_actions(obs, policy)
                # print(f"Predicted action chunk shape: {action_chunk.shape}")

                t = piper_step_chunk_dual(piper_right, piper_left, action_chunk[:chunk_sizes], t, n_steps=chunk_sizes)

            print(f"Replay rollout {i + 1} completed.")

        f_robot.close()
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run infer_piper.py with custom directories.")
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Path to the checkpoint directory')
    parser.add_argument('--config_name', type=str, required=True, help='Config name')
    parser.add_argument('--rollouts_num', type=int, default=1, help='Number of rollouts (default: 1)')
    parser.add_argument('--prev_actions', type=int, default=50, help='Number of previous actions to consider (default: 50)')
    parser.add_argument('--replay_episode_dir', type=str, default=None, help='Path to replay episode directory for offline replay')
    args = parser.parse_args()

    chunk_sizes = 50

    main(args, chunk_sizes)
