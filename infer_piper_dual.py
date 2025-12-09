import numpy as np
import time 
import argparse
from utils import *
import pyrealsense2 as rs


def main(args, chunk_sizes=10):
    max_steps = 1000

    # 初始化相机
    context = rs.context()
    devices = context.query_devices()
    serial_numbers = [dev.get_info(rs.camera_info.serial_number) for dev in devices]

    if not serial_numbers:
        print("错误：未找到 RealSense 相机！")
        return

    print(f"找到 {len(serial_numbers)} 台相机:")
    for serial in serial_numbers:
        print(f"  - {serial}")

    pipelines = {}
    configs = {}
    aligns = {}

    width, height, fps = 640, 480, 30
    for serial in serial_numbers:
        pipeline = rs.pipeline(context) 
        config = rs.config()
        config.enable_device(serial) 
        try:
            # config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            pipelines[serial] = pipeline
            configs[serial] = config
            aligns[serial] = rs.align(rs.stream.color) 
        except Exception as e:
            print(f"警告：无法为相机 {serial} 配置流。错误: {e}")
            if serial in serial_numbers:
                serial_numbers.remove(serial)

    active_serials = []
    for serial in serial_numbers:
        print(f"正在启动相机: {serial}...")
        try:
            pipelines[serial].start(configs[serial])
            active_serials.append(serial)
            print(f"相机 {serial} 启动成功。")
        except Exception as e:
            print(f"错误：无法启动相机 {serial}。{e}")
            if serial in pipelines: del pipelines[serial]
            if serial in configs: del configs[serial]
            if serial in aligns: del aligns[serial]

    if not active_serials:
        print("没有相机成功启动。")
        return

    print(f"\n成功启动 {len(active_serials)} 台相机")
    # 对 active_serials 进行排序
    active_serials.sort()

    # 初始化双臂piper接口
    piper_right = C_PiperInterface_V2("can_arm1")
    piper_left = C_PiperInterface_V2("can_arm2")
    piper_right.ConnectPort()
    piper_left.ConnectPort()
    while not (piper_right.EnablePiper() and piper_left.EnablePiper()):
        time.sleep(0.01)
    print("Piper双臂已连接并启动。")

    # 双臂初始位置（关节和夹爪）
    piper_right.MotionCtrl_2(0x01, 0x01, 60, 0x00)
    piper_left.MotionCtrl_2(0x01, 0x01, 60, 0x00)
    # piper_right.JointCtrl(41913, 50004, -74840, -3245, 47584, -2760)
    # piper_left.JointCtrl(-36499, 299, -3872, -54116, 5961, 54201) # open drawer 
    # piper_right.JointCtrl(41920, 49997, -74840, -3245, 47584, -2760)
    # piper_left.JointCtrl(-24171, 14878, -4253, -27609, -8485, 17650) # open & close drawer
    piper_right.JointCtrl(40036, 3630, -9526, -3140, 17670, -12043)
    piper_left.JointCtrl(-40543, 177, -104, -87029, -5647, 77959) #  put item in drawer
    piper_right.GripperCtrl(abs(0), 100, 0x01, 0)
    piper_left.GripperCtrl(abs(0), 100, 0x01, 0)
    print("Piper双臂初始化完成。")

    # 加载模型
    policy = load_policy(args.checkpoint_dir, args.config_name)

    origin_observation_state = 0

    t = 0
    action_plan = []
    history_actions = []
    while t < max_steps:
        if not action_plan:
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
                aligned_img = preprocess_image_for_alignment(color_frame, quality=90)
                final_img = cv2.resize(aligned_img, (224, 224), interpolation=cv2.INTER_AREA)
                all_device_images.append(final_img)
                # all_device_images.append(color_frame)

            for idx, img in enumerate(all_device_images):
                window_name = f"Camera {idx}"
                cv2.imshow(window_name, img)
            cv2.waitKey(2)

            joint_right = piper_right.GetArmJointMsgs().joint_state
            joint_left = piper_left.GetArmJointMsgs().joint_state
            gripper_right = piper_right.GetArmGripperMsgs().gripper_state.grippers_angle
            gripper_left = piper_left.GetArmGripperMsgs().gripper_state.grippers_angle

            joint_right_arr = np.array([
                joint_right.joint_1 / 1000.0,
                joint_right.joint_2 / 1000.0,
                joint_right.joint_3 / 1000.0,
                joint_right.joint_4 / 1000.0,
                joint_right.joint_5 / 1000.0,
                joint_right.joint_6 / 1000.0,
            ])
            joint_left_arr = np.array([
                joint_left.joint_1 / 1000.0,
                joint_left.joint_2 / 1000.0,
                joint_left.joint_3 / 1000.0,
                joint_left.joint_4 / 1000.0,
                joint_left.joint_5 / 1000.0,
                joint_left.joint_6 / 1000.0,
            ])

            gripper_right_arr = np.array([gripper_right / 1000.0])
            gripper_left_arr = np.array([gripper_left / 1000.0])

            current_observation_state = np.concatenate((
                joint_right_arr,
                gripper_right_arr,
                joint_left_arr,
                gripper_left_arr
            ))

            print("当前观测状态（双臂拼接）:", current_observation_state)

            # if t == 0:
            #     origin_observation_state = current_observation_state

            # if t == 0:
            #     pad = np.zeros((len_prev_actions, 32), dtype=np.float32)
            #     obs_state_padded = np.zeros(32, dtype=np.float32)
            #     obs_state_padded[:origin_observation_state.shape[0]] = origin_observation_state
            #     pad[:] = obs_state_padded
            #     history_action = pad
            # elif t < len_prev_actions:
            #     pad_len = len_prev_actions - t
            #     pad = np.zeros((pad_len, 32), dtype=np.float32)
            #     obs_state_padded = np.zeros(32, dtype=np.float32)
            #     obs_state_padded[:origin_observation_state.shape[0]] = origin_observation_state
            #     pad[:] = obs_state_padded
            #     print(f"pad shape {pad.shape}")
            #     history_action = np.concatenate((pad, np.array(history_actions).reshape(-1, 32)),
            #                                    axis=0)
            # else:
            #     history_action = np.array(history_actions[-len_prev_actions:]).reshape(-1, 32)

            obs = {
                'observation/left_image': all_device_images[0],
                'observation/top_image': all_device_images[1],
                'observation/right_image': all_device_images[2],
                'observation/state': current_observation_state,
                # 'prompt': "open drawer then close drawer",
                'prompt': "open drawer, put the yellow block in drawer and close drawer",
            }

            action_chunk = infer_actions(obs, policy)
            # print(f"推理动作块形状: {action_chunk.shape}, 类型: {type(action_chunk)}")

            # for act in action_chunk[:chunk_sizes]:
            #     act_padded = np.zeros(32, dtype=np.float32)
            #     act_padded[:len(act)] = act
            #     history_actions.append(act_padded)

        # 现使用双臂执行步进函数
        t = piper_step_chunk_dual(piper_right, piper_left, action_chunk, t, n_steps=chunk_sizes)

    print(f"推理执行完成。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run infer_piper.py with custom directories.")
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Path to the checkpoint directory')
    parser.add_argument('--config_name', type=str, required=True, help='Config name')
    parser.add_argument('--prev_actions', type=int, default=50, help='Number of previous actions to consider (default: 50)')
    args = parser.parse_args()

    chunk_sizes = 50


    main(args, chunk_sizes)
