import numpy as np
import time 
import argparse
from utils import *
import pyrealsense2 as rs

np.set_printoptions(precision=2)


def main(args, chunk_sizes, prompt, max_steps=5000, len_prev_actions = 25):

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
    active_serials.sort()

    # 初始化双臂piper接口
    piper_right = C_PiperInterface_V2("can_arm1")
    # piper_left = C_PiperInterface_V2("can_arm2")
    piper_right.ConnectPort()
    # piper_left.ConnectPort()
    while not (piper_right.EnablePiper()):
        time.sleep(0.01)
    print("Piper右臂已连接并启动。")

    # 双臂初始位置（关节和夹爪）
    piper_right.MotionCtrl_2(0x01, 0x01, 30, 0x00)
    # piper_left.MotionCtrl_2(0x01, 0x01, 40, 0x00)
    # piper_right.JointCtrl(46635, 91114, -81951, -16699, 69789, 47623)
    # piper_left.JointCtrl(-40543, 177, -104, -87029, -5647, 77959)
    piper_right.JointCtrl(37796, 73905, -54213, -1811, 69789, 34758) # push buttons
    # piper_left.JointCtrl(-30333, 1339, -5378, 781, 11710, -2983) # open and close drawer
    piper_right.GripperCtrl(abs(0), 100, 0x01, 0)
    # piper_left.GripperCtrl(abs(0), 500, 0x01, 0)
    print("Piper臂初始化完成。")

    # 加载模型
    policy = load_policy(args.checkpoint_dir, args.config_name)

    current_prompt = prompt
    print(f"\n当前 Prompt: {current_prompt}")
    print("[caution] 在相机窗口激活时按【空格键】可修改 Prompt")

    t = 0
    history_actions = []
    
    while t < max_steps:
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
            # aligned_img = preprocess_image_for_alignment(color_frame, quality=90)
            # final_img_bgr = cv2.resize(aligned_img, (224, 224), interpolation=cv2.INTER_AREA)
            final_img_bgr = cv2.resize(color_frame, (224, 224), interpolation=cv2.INTER_AREA)
            final_img_rgb = cv2.cvtColor(final_img_bgr, cv2.COLOR_BGR2RGB)
            all_device_images.append(final_img_rgb)
            # all_device_images.append(color_frame)

        for idx, img in enumerate(all_device_images):
            window_name = f"Camera {idx}"
            # cv2.imshow(window_name, img)
            cv2.imshow(window_name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(1) & 0xFF
        
        if key == 32: # 32 是空格键的 ASCII 码
            print("\n" + "="*40)
            print("检测到空格键，推理已暂停。")
            print(f"当前 Prompt: {current_prompt}")
            
            # 使用 input 会阻塞程序，直到用户在终端输入回车
            # 这正好满足了等待用户输入后再继续的需求
            try:
                new_input = input("请输入新的 Prompt (直接回车保持不变): ").strip()
                if new_input:
                    current_prompt = new_input
                    print(f"Prompt 已更新为: {current_prompt}")
                else:
                    print("输入为空，保持原有 Prompt。")
            except KeyboardInterrupt:
                print("取消输入...")
            
            print("继续推理...")
            print("="*40 + "\n")

        elif key == ord('r'):
            import random
            print("重置dan臂位置...")
            time.sleep(1)
            piper_right.MotionCtrl_2(0x01, 0x01, 40, 0x00)
            # piper_left.MotionCtrl_2(0x01, 0x01, 40, 0x00)
            rand_list = [random.randint(-2000, 2000) for _ in range(6)]
            piper_right.JointCtrl(37796 + rand_list[0], 73905 + rand_list[1], -54213 + rand_list[2], -1811 + rand_list[3], 69789 + rand_list[4], 34758 + rand_list[5]) # push buttons
            # piper_left.JointCtrl(-30333 + rand_list[0], 1339 + rand_list[1], -5378 + rand_list[2], 781 + rand_list[3], 11710 + rand_list[4], -2983 + rand_list[5]) # open and close drawer
            piper_right.GripperCtrl(abs(0), 500, 0x01, 0)
            # piper_left.GripperCtrl(abs(0), 500, 0x01, 0)
            time.sleep(1)
            print("臂已重置。")
            history_actions = []
            t = 0

        if args.mode == 'joint':    

            actions_right = piper_right.GetArmJointMsgs().joint_state
            # actions_left = piper_left.GetArmJointMsgs().joint_state

            actions_right_arr = np.array([
                actions_right.joint_1 / 1000.0, 
                actions_right.joint_2 / 1000.0,
                actions_right.joint_3 / 1000.0,
                actions_right.joint_4 / 1000.0,
                actions_right.joint_5 / 1000.0,
                actions_right.joint_6 / 1000.0,
            ])
            # actions_left_arr = np.array([
            #     actions_left.joint_1 / 1000.0,
            #     actions_left.joint_2 / 1000.0,
            #     actions_left.joint_3 / 1000.0,
            #     actions_left.joint_4 / 1000.0,
            #     actions_left.joint_5 / 1000.0,
            #     actions_left.joint_6 / 1000.0,
            # ])
        # elif args.mode == 'ee':

        #     actions_right = piper_right.GetArmEndPoseMsgs().end_pose
        #     # actions_left = piper_left.GetArmEndPoseMsgs().end_pose

        #     actions_right_arr = np.array([
        #         actions_right.X_axis / 1000.0,
        #         actions_right.Y_axis / 1000.0,
        #         actions_right.Z_axis / 1000.0,
        #         actions_right.RX_axis / 1000.0,
        #         actions_right.RY_axis / 1000.0,
        #         actions_right.RZ_axis / 1000.0,
        #     ])
        #     # actions_left_arr = np.array([
        #     #     actions_left.X_axis / 1000.0,
        #     #     actions_left.Y_axis / 1000.0,
        #     #     actions_left.Z_axis / 1000.0,
        #     #     actions_left.RX_axis / 1000.0,
        #     #     actions_left.RY_axis / 1000.0,
        #     #     actions_left.RZ_axis / 1000.0,
        #     # ])

        gripper_right = piper_right.GetArmGripperMsgs().gripper_state.grippers_angle
        # gripper_left = piper_left.GetArmGripperMsgs().gripper_state.grippers_angle
        gripper_right_arr = np.array([gripper_right / 1000.0])
        # gripper_left_arr = np.array([gripper_left / 1000.0])
        if t ==0: 
            origin_observation_state = np.concatenate((
                actions_right_arr, gripper_right_arr), axis=0)
            # origin_observation_state = np.concatenate((
            #     actions_left_arr, gripper_left_arr), axis=0)

        current_observation_state = np.concatenate((
            actions_right_arr,
            gripper_right_arr,
            # actions_left_arr,
            # gripper_left_arr
        ))

        print("当前观测状态:", current_observation_state)

        if t == 0:
            # history_action = np.zeros((1,32))
            pad = np.zeros((len_prev_actions, 32), dtype=np.float32)
            obs_state_padded = np.zeros(32, dtype=np.float32)
            obs_state_padded[:origin_observation_state.shape[0]] = origin_observation_state
            pad[:] = obs_state_padded
            history_action = pad
        elif t < len_prev_actions:
            print('t=', t)
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
            print('t < len_prev_actions')
        else:   
            print('history_actions shape:', np.array(history_actions[-len_prev_actions:]).shape)
            history_action = np.array(history_actions[-len_prev_actions:]).reshape(-1, 32)
            print('t >= len_prev_actions')

        print(f"历史动作形状: {history_action.shape}")

        obs = {
            # 'observation/left_image': all_device_images[0],
            'observation/top_image': all_device_images[1],
            'observation/right_image': all_device_images[2],
            'observation/state': current_observation_state,
            'prompt': current_prompt,
            'prev_actions': history_action,
        }

        action_chunk = infer_actions(obs, policy)
        print(f"推理动作块形状: {action_chunk.shape}, 类型: {type(action_chunk)}")
        history_actions.extend(action_chunk[:chunk_sizes])
        print(f"历史动作长度: {len(history_actions)}")

        print(f"执行动作块..., t=", t)
        t = piper_step_chunk_single(piper_right, action_chunk, t, mode=args.mode, n_steps=chunk_sizes)
        print(f"当前步骤 t: {t}\n")

    print(f"推理执行完成。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run infer_piper.py with custom directories.")
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Path to the checkpoint directory')
    parser.add_argument('--config_name', type=str, required=True, help='Config name')
    parser.add_argument('--mode', type=str, required=True, choices=['joint', 'ee'], help='Control mode: joint or ee (default: joint)')
    args = parser.parse_args()

    chunk_sizes = 20
    max_steps = 10000
    len_prev_actions = chunk_sizes
    prompt = "sequentially touch green and yellow buttons"

    main(args, chunk_sizes, prompt, max_steps, len_prev_actions)
