import os
import time
import h5py
import cv2
import numpy as np
import argparse
import multiprocessing as mp
from multiprocessing import Event
from piper_sdk import C_PiperInterface, LogLevel
import pyrealsense2 as rs

CHUNK_SIZE = 1000  # H5文件每次预分配的行数

def jpeg_compress(img: np.ndarray, quality=90) -> bytes:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', img, encode_param)
    if not result:
        raise ValueError("JPEG compression failed")
    return encimg.tobytes()

def create_episode_folder(base_path, task_name, ep_idx):
    task_folder = os.path.join(base_path, "Data", task_name)
    os.makedirs(task_folder, exist_ok=True)
    path = os.path.join(task_folder, f'episode{ep_idx}')
    os.makedirs(path, exist_ok=True)
    return path

class RobotDataProcess(mp.Process):
    def __init__(self, start_event: Event, stop_event: Event, episode_path: str, 
                 arm1_can="can_arm1", arm2_can="can_arm2"):
        super().__init__()
        self.start_event = start_event
        self.stop_event = stop_event
        self.episode_path = episode_path
        self.arm1_can = arm1_can
        self.arm2_can = arm2_can

    def _init_piper(self, can_name):
        """初始化单个机械臂"""
        piper = C_PiperInterface(can_name=can_name,
                                 judge_flag=False,
                                 can_auto_init=True,
                                 dh_is_offset=1,
                                 start_sdk_joint_limit=False,
                                 start_sdk_gripper_limit=False,
                                 logger_level=LogLevel.WARNING,
                                 log_to_file=False,
                                 log_file_path=None)
        piper.ConnectPort()
        return piper

    def _get_single_arm_data(self, piper_instance):
        """读取单个臂的数据并返回numpy数组"""
        gripper_msg = piper_instance.GetArmGripperMsgs()
        joints_msg = piper_instance.GetArmJointMsgs()
        end_pose_msg = piper_instance.GetArmEndPoseMsgs()

        if joints_msg is None or gripper_msg is None or end_pose_msg is None:
            return None, None

        # 7维: 6个关节 + 1个夹爪
        joints_arr = np.array([
            joints_msg.joint_state.joint_1 / 1000.0,
            joints_msg.joint_state.joint_2 / 1000.0,
            joints_msg.joint_state.joint_3 / 1000.0,
            joints_msg.joint_state.joint_4 / 1000.0,
            joints_msg.joint_state.joint_5 / 1000.0,
            joints_msg.joint_state.joint_6 / 1000.0,
            gripper_msg.gripper_state.grippers_angle / 1000.0,
        ])

        # 7维: XYZ + RPY + 1个夹爪
        end_pose_arr = np.array([
            end_pose_msg.end_pose.X_axis / 1000.0,
            end_pose_msg.end_pose.Y_axis / 1000.0,
            end_pose_msg.end_pose.Z_axis / 1000.0,
            end_pose_msg.end_pose.RX_axis / 1000.0,
            end_pose_msg.end_pose.RY_axis / 1000.0,
            end_pose_msg.end_pose.RZ_axis / 1000.0, 
            gripper_msg.gripper_state.grippers_angle / 1000.0,
        ])
        
        return joints_arr, end_pose_arr

    def run(self):

        print(f"[Robot] Connecting Arm 1 on {self.arm1_can}...")
        piper1 = self._init_piper(self.arm1_can)
        
        print(f"[Robot] Connecting Arm 2 on {self.arm2_can}...")
        piper2 = self._init_piper(self.arm2_can)

        h5_path = os.path.join(self.episode_path, "robot_data.h5")
        robot_file = h5py.File(h5_path, 'w')

        # 关节
        joints_ds = robot_file.create_dataset('joints', shape=(0, 14), maxshape=(None, 14), dtype='f8')
        # 末端
        ee_ds = robot_file.create_dataset('end_pose', shape=(0, 14), maxshape=(None, 14), dtype='f8')
        # 时间戳
        ts_ds = robot_file.create_dataset('timestamps', shape=(0,), maxshape=(None,), dtype='f8')

        idx = 0
        dataset_size = 0
        CHUNK_SIZE = 1000

        print("[Robot] Ready. Waiting for start signal...")

        while not self.start_event.is_set():
            if self.stop_event.is_set():
                robot_file.close()
                return 
            time.sleep(0.01)

        print("[Robot] Recording started.")

        while not self.stop_event.is_set():
            # 1. 获取两臂数据
            j1, ee1 = self._get_single_arm_data(piper1)
            j2, ee2 = self._get_single_arm_data(piper2)

            timestamp = time.time()

            # 2. 完整性检查：如果任一臂数据缺失，丢弃该帧以保持对齐
            if j1 is None or j2 is None:
                continue

            # 3. 数据拼接 
            # 结果 shape: (14,)
            joints_concat = np.concatenate((j1, j2)) 
            # 结果 shape: (14,)
            ee_concat = np.concatenate((ee1, ee2))

            # 4. 扩容H5 dataset
            if idx >= dataset_size:
                dataset_size += CHUNK_SIZE
                joints_ds.resize((dataset_size, 14))
                ee_ds.resize((dataset_size, 14))    
                ts_ds.resize((dataset_size,))

            # 5. 写入数据
            joints_ds[idx, :] = joints_concat
            ee_ds[idx, :] = ee_concat
            ts_ds[idx] = timestamp

            idx += 1

            time.sleep(0.002) 

        # --- 结束清理 ---
        print(f"[Robot] Recording stopped. Saving {idx} frames...")
        joints_ds.resize((idx, 14))
        ee_ds.resize((idx, 14))
        ts_ds.resize((idx,))
        
        robot_file.close()
        print("[Robot] Data saved.")
        

class CameraDataProcess(mp.Process):
    def __init__(self, start_event: Event, stop_event: Event, episode_path: str):
        super().__init__()
        self.start_event = start_event
        self.stop_event = stop_event
        self.episode_path = episode_path

    def run(self):
        # --- 1. 初始化相机资源 ---
        context = rs.context()
        devices = context.query_devices()
        serials = [dev.get_info(rs.camera_info.serial_number) for dev in devices]
        
        if not serials:
            print("[Camera] No RealSense devices found.")
            return

        pipelines = {}
        configs = {}
        aligns = {}
        width, height, fps = 640, 480, 30

        # 配置 Pipeline
        for serial in serials:
            pipeline = rs.pipeline(context)
            config = rs.config()
            config.enable_device(serial)
            try:
                config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            except Exception as e:
                print(f"[Camera] Failed to enable stream for {serial}: {e}")
                continue
            pipelines[serial] = pipeline
            configs[serial] = config
            aligns[serial] = rs.align(rs.stream.color)

        active_serials = []
        # 启动 Pipeline
        for serial in serials:
            if serial not in pipelines: continue
            try:
                pipelines[serial].start(configs[serial])
                active_serials.append(serial)
                print(f"[Camera] {serial} stream started. Warming up...")
            except Exception as e:
                print(f"[Camera] Failed to start pipeline for {serial}: {e}")

        if not active_serials:
            print("[Camera] No active cameras.")
            return
        
        active_serials.sort()  # 保持顺序一致性

        # --- 2. 预热阶段 (Warm-up / Standby) ---
        # 在主进程设置 start_event 之前，持续读取数据以保持 AE/AWB 稳定
        # 同时监听 stop_event 以便允许在预热阶段退出
        while not self.start_event.is_set() and not self.stop_event.is_set():
            for serial in active_serials:
                try:
                    pipelines[serial].try_wait_for_frames(timeout_ms=10)
                except:
                    pass
            time.sleep(0.01) # 避免预热阶段占用过多 CPU

        # 如果在预热阶段就收到了停止信号，则直接清理退出
        if self.stop_event.is_set():
            for serial in active_serials:
                pipelines[serial].stop()
            return

        # --- 3. 录制阶段 (Recording) ---
        print("[Camera] Start Event received. Beginning recording...")
        
        h5_path = os.path.join(self.episode_path, "camera_data.h5")
        cam_file = h5py.File(h5_path, 'w')

        cam_dsets = {}
        ts_dsets = {}
        dset_sizes = {s: 0 for s in active_serials} 
        idx_cams = {s: 0 for s in active_serials}

        # 创建 Dataset
        for serial in active_serials:
            cam_dsets[serial] = cam_file.create_dataset(f'images_{serial}',
                                                        shape=(0,),
                                                        maxshape=(None,),
                                                        dtype=h5py.special_dtype(vlen=np.dtype('uint8')))
            ts_dsets[serial] = cam_file.create_dataset(f'timestamps_{serial}',
                                                       shape=(0,),
                                                       maxshape=(None,),
                                                       dtype='f8')

        while not self.stop_event.is_set():
            for serial in active_serials:
                pipeline = pipelines[serial]
                align = aligns[serial]

                success, frames = pipeline.try_wait_for_frames(timeout_ms=100)
                timestamp = time.time()
                if not success:
                    continue
                
                # 对齐深度和彩色（虽然只存彩色，但align通常能修复一部分视差畸变）
                aligned_frames = align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                if not color_frame:
                    continue

                color_img = np.asanyarray(color_frame.get_data())
                
                try:
                    jpeg_bytes = jpeg_compress(color_img, quality=90)
                except Exception as e:
                    continue

                i = idx_cams[serial]
                
                # Chunk Resize 策略
                if i >= dset_sizes[serial]:
                    dset_sizes[serial] += CHUNK_SIZE
                    cam_dsets[serial].resize((dset_sizes[serial],))
                    ts_dsets[serial].resize((dset_sizes[serial],))

                cam_dsets[serial][i] = np.frombuffer(jpeg_bytes, dtype='uint8')
                ts_dsets[serial][i] = timestamp

                idx_cams[serial] += 1

        # --- 4. 清理与保存 ---
        print("[Camera] Stop Event received. Saving data...")
        for serial in active_serials:
            final_size = idx_cams[serial]
            cam_dsets[serial].resize((final_size,))
            ts_dsets[serial].resize((final_size,))
            if serial in pipelines:
                pipelines[serial].stop()
        
        cam_file.close()
        print("[Camera] Data saved and closed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_episode', type=int, default=0)
    parser.add_argument('--task_name', type=str, required=True, help='Task name')
    args = parser.parse_args()

    # base_path = os.path.abspath(os.path.dirname(__file__))
    base_path = "/home/tengenx2204/workspace/mozihao"
    episode_idx = args.start_episode

    start_event = mp.Event()
    stop_event = mp.Event()

    # 键盘监听
    def wait_key():
        try:
            import msvcrt
            return msvcrt.getch().decode('utf-8').lower()
        except ImportError:
            import sys
            import tty
            import termios
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                ch = sys.stdin.read(1).lower()
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            return ch

    while True:
        episode_path = create_episode_folder(base_path, args.task_name, episode_idx)
        
        start_event.clear()
        stop_event.clear()

        print(f"\n=== Episode {episode_idx} Setup ===")
        print("Initializing Robot and Cameras... (Please wait)")

        # 3. 启动进程 (此时相机开始预热，但未录制)
        robot_proc = RobotDataProcess(start_event, stop_event, episode_path)
        cam_proc = CameraDataProcess(start_event, stop_event, episode_path)
        
        robot_proc.start()
        cam_proc.start()

        # 4. 等待用户输入以开始录制
        print(f"System Ready. Cameras are warming up.")
        print(f"Press 's' to START recording.")
        print(f"Press 'q' to QUIT (during setup phase).")

        k = wait_key()
        
        if k == 's':
            print(f"--> STARTING RECORDING for Episode {episode_idx}...")
            start_event.set() # 触发录制

            print("Recording... Press 'p' to PAUSE/STOP current episode.")
            while True:
                k2 = wait_key()
                if k2 == 'p':
                    print("--> STOPPING RECORDING...")
                    stop_event.set() # 触发停止
                    
                    # 等待进程结束
                    robot_proc.join()
                    cam_proc.join()
                    
                    episode_idx += 1
                    break
                time.sleep(0.1)
        elif k == 'q':
            print("Quitting...")
            stop_event.set() 
            robot_proc.join()
            cam_proc.join()
            break


if __name__ == "__main__":
    main()