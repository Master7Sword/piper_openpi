import time
from piper_sdk import *


if __name__ == "__main__":
    piper_right = C_PiperInterface_V2("can_arm1")
    piper_left = C_PiperInterface_V2("can_arm2")
    piper_right.ConnectPort()
    piper_left.ConnectPort()
    while( not piper_right.EnablePiper() or not piper_left.EnablePiper()):
        time.sleep(0.01)

    piper_right.GripperCtrl(0,1000,0x01, 0)
    piper_left.GripperCtrl(0,1000,0x01, 0)
    factor = 57295.7795 # 1000*180/3.1415926
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


    