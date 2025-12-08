#!/usr/bin/env python3
# -*-coding:utf8-*-
# 注意demo无法直接运行，需要pip安装sdk后才能运行
import time
from piper_sdk import *
import numpy as np
from scipy.interpolate import interp1d


def piper_step(piper, action, factor=1000):
    joint_0, joint_1, joint_2, joint_3, joint_4, joint_5, joint_6 = action[0], action[1], action[2], action[3], action[4], action[5], action[6]
    start_time = time.time()

    try:
        joint_0 = round(joint_0 * factor)
        joint_1 = round(joint_1 * factor)
        joint_2 = round(joint_2 * factor)
        joint_3 = round(joint_3 * factor)
        joint_4 = round(joint_4 * factor)
        joint_5 = round(joint_5 * factor)
        joint_6 = round(joint_6 * factor)

        piper.MotionCtrl_2(0x01, 0x01, 5, 0x00)
        piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
        piper.GripperCtrl(abs(joint_6), 100, 0x01, 0)
    except Exception as e:
        raise RuntimeError(f"Piper command failed: {e}")
    
    while piper.GetArmStatus().arm_status.motion_status == 0x01:
        time.sleep(0.0001)  # wait for motion to complete

    elapsed = time.time() - start_time
    print(f'actual fps: {1./elapsed:.4f}')
    fps = 5
    frame_duration = 1.0 / fps
    if elapsed < frame_duration:
        time.sleep(frame_duration - elapsed)



# def moveJ_demo():
#     factor = 57295.7795 #1000*180/3.1415926
#     src = [0,0,0,0,0,0,0]
#     dst = [0.2,0.2,-0.2,0.3,-0.2,0.5,0.08]


#     n_steps = 10*5

#     action_chunk = np.concatenate((np.array(src), np.array(dst)*2), axis=0).reshape(2,7)
#     print(action_chunk.shape)
#     orig_steps = action_chunk.shape[0]
#     x_old = np.linspace(0, 1, orig_steps)
#     x_new = np.linspace(0, 1, n_steps)
#     interpolator = interp1d(x_old, action_chunk, axis=0, kind='linear')
#     action_chunk_interp = interpolator(x_new)  # shape: (1000, 7)
#     print(action_chunk_interp.shape)

#     for i in range(n_steps):
#         action = action_chunk_interp[i]
#         print(action)
#         st = time.time()
#         piper_step(piper, action, factor=factor)
#         print("step fps:", 1.0/(time.time() - st))

#     while piper.GetArmStatus().arm_status.motion_status == 0x01:
#         time.sleep(0.0001)  # wait for motion to complete

#     # time.sleep(0.2)



if __name__ == "__main__":
    # piper = C_PiperInterface_V2("can0")
    piper_right = C_PiperInterface_V2("can_arm1")
    piper_left = C_PiperInterface_V2("can_arm2")
    piper_right.ConnectPort()
    piper_left.ConnectPort()
    while( not piper_right.EnablePiper() or not piper_left.EnablePiper()):
        time.sleep(0.01)

    piper_right.GripperCtrl(0,1000,0x01, 0)
    piper_left.GripperCtrl(0,1000,0x01, 0)
    factor = 57295.7795 #1000*180/3.1415926
    position = [0,0,0,0,0,0,0]
    count = 0
    last_motion_status = 0
    while True:
        count  = count + 1
        print(count)
# 
        piper_right.MotionCtrl_2(0x01, 0x01, 20, 0x00)
        piper_left.MotionCtrl_2(0x01, 0x01, 20, 0x00)
        piper_right.JointCtrl(41920, 49997, -74840, -3245, 47584, -2760)
        piper_left.JointCtrl(-24171, 14878, -4253, -27609, -8485, 17650)
        piper_right.GripperCtrl(abs(100), 1000, 0x01, 0)
        piper_left.GripperCtrl(abs(100), 1000, 0x01, 0)
        time.sleep(0.2)

        while piper_right.GetArmStatus().arm_status.motion_status == 0x01 or piper_left.GetArmStatus().arm_status.motion_status == 0x01:
            time.sleep(0.0001)


        print(piper_right.GetArmStatus().arm_status.motion_status)
        print(piper_left.GetArmStatus().arm_status.motion_status)

        time.sleep(0.05)


    