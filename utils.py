import cv2
import torch
import time
from piper_sdk import *

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


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
        joints_left = [round(x * 1000) for x in action[7:13]]
        gripper_left = round(action[13] * 1000)

        gripper_right = 0 if abs(gripper_right) < 40000 else gripper_right
        gripper_left = 0 if abs(gripper_left) < 30000 else gripper_left

        piper_right.MotionCtrl_2(0x01, 0x01, 50, 0x00)
        piper_left.MotionCtrl_2(0x01, 0x01, 50, 0x00)

        piper_right.JointCtrl(*joints_right)
        piper_left.JointCtrl(*joints_left)

        piper_right.GripperCtrl(abs(gripper_right), 500, 0x01, 0)
        piper_left.GripperCtrl(abs(gripper_left), 500, 0x01, 0)
    except Exception as e:
        raise RuntimeError(f"Piper dual-arm command failed: {e}")
    
    counter = 0
    while (piper_right.GetArmStatus().arm_status.motion_status == 0x01 or
           piper_left.GetArmStatus().arm_status.motion_status == 0x01):
        time.sleep(0.0001)
        counter += 1
        if counter > 300:
            print("Warning: Piper dual-arm motion taking too long.")
            break

    elapsed = time.time() - start_time
    print(f'dual-arm actual fps: {1./elapsed:.4f}')

    fps = 60
    frame_duration = 1.0 / fps
    if elapsed < frame_duration:
        time.sleep(frame_duration - elapsed)


def piper_step_chunk_dual(piper_right, piper_left, action_chunk, t, n_steps=50):
    print(action_chunk.shape , n_steps)
    assert action_chunk.shape[0] >= n_steps
    t = t + n_steps
        
    action_chunk = action_chunk[:n_steps]

    for i in range(n_steps):
        action = action_chunk[i]
        # print(f'Dual actions: {action[:14]}')
        # print(f"action {i} right hand: {action[:7]}")
        # print(f"action {i} left hand: {action[7:14]}")
        # time.sleep(1)
        piper_step_dual(piper_right, piper_left, action)

    return t


def preprocess_image_for_alignment(img, quality=90):
    """
    模拟数据采集时的 JPEG 压缩过程，以保持推理输入与训练输入的一致性
    """
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', img, encode_param)
    decimg = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
    return decimg