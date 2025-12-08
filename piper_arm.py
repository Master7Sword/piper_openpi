import time
# Import piper_sdk module
from piper_sdk import *

if __name__ == "__main__":
    piper = C_PiperInterface(can_name="can_arm2",
                                judge_flag=False,
                                can_auto_init=True,
                                dh_is_offset=1,
                                start_sdk_joint_limit=False,
                                start_sdk_gripper_limit=False,
                                logger_level=LogLevel.WARNING,
                                log_to_file=False,
                                log_file_path=None)
    # Enable can send and receive threads
    piper.ConnectPort()
    piper.EnableFkCal()
    while True:
        print(piper.GetArmJointMsgs())
        # print(piper.GetFK()[-1])
        time.sleep(0.005)# 200hz