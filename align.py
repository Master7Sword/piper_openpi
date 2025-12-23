import h5py
import numpy as np
import os
import argparse
import shutil
from glob import glob
import cv2

def find_nearest_idx(array, value):
    """
    在有序数组 array 中找到最接近 value 的索引。
    """
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or np.abs(value - array[idx-1]) < np.abs(value - array[idx])):
        return idx - 1
    else:
        return idx
    

def post_process_append_hold(data_dict, frames_dir, num_cams, n_repeat=20):
    """
    后处理：将最后一帧的状态和动作复制 n_repeat 份，模拟停留。
    
    Args:
        data_dict (dict): 包含 'timestamps', 'joints', 'end_pose' 等数据的字典。
        frames_dir (str): 保存图片的根目录 (e.g., .../frames)。
        num_cams (int): 相机数量，用于遍历 cam0, cam1...
        n_repeat (int): 需要追加的帧数。
    """
    if(n_repeat <= 0):
        return
    timestamps = data_dict["timestamps"]
    
    # 边界检查：如果数据太少，无法计算 delta_t，直接跳过
    if len(timestamps) < 2:
        print("警告: 帧数不足，跳过末尾填充处理。")
        return

    # 1. 计算时间增量 dt
    # 取最后两帧的时间差作为递增步长
    dt = timestamps[-1] - timestamps[-2]
    
    # 获取需要复制的最后一帧的数据
    last_ts = timestamps[-1]
    last_joints = data_dict["joints"][-1]
    last_ee = data_dict["end_pose"][-1]
    last_j_action = data_dict["joints_actions"][-1]
    last_ee_action = data_dict["end_pose_actions"][-1]

    # 获取当前已有的帧数（用于计算图片文件名）
    current_count = len(timestamps)
    last_img_idx = current_count - 1  # 最后一帧的文件名索引 (e.g., 00199)

    print(f"  - 执行后处理: 追加 {n_repeat} 帧静止数据 (dt={dt:.4f})")

    for i in range(1, n_repeat + 1):
        # --- A. 更新数据字典 ---
        new_ts = last_ts + (dt * i)
        
        data_dict["timestamps"].append(new_ts)
        data_dict["joints"].append(last_joints)
        data_dict["end_pose"].append(last_ee)
        # 动作通常也保持不变（继续保持当前姿态），或者设为零速，这里复用最后一帧动作
        data_dict["joints_actions"].append(last_j_action)
        data_dict["end_pose_actions"].append(last_ee_action)

        # --- B. 复制图像文件 ---
        # 新的索引
        new_img_idx = last_img_idx + i
        
        for cam_id in range(num_cams):
            # 源文件路径 (最后一帧)
            src_path = os.path.join(frames_dir, f"cam{cam_id}", f"{last_img_idx:05d}.jpg")
            # 目标文件路径 (追加帧)
            dst_path = os.path.join(frames_dir, f"cam{cam_id}", f"{new_img_idx:05d}.jpg")
            
            if os.path.exists(src_path):
                # 使用 copy 此时比 cv2 读取再写入快得多且不损失画质
                shutil.copy(src_path, dst_path)


def process_episode(episode_path):
    robot_h5_path = os.path.join(episode_path, "robot_data.h5")
    camera_h5_path = os.path.join(episode_path, "camera_data.h5")

    if not os.path.exists(robot_h5_path) or not os.path.exists(camera_h5_path):
        print(f"跳过 {episode_path}: 缺少数据文件")
        return

    print(f"正在处理: {episode_path} ...")

    try:
        f_cam = h5py.File(camera_h5_path, 'r')
        f_robot = h5py.File(robot_h5_path, 'r')
    except Exception as e:
        print(f"读取H5出错: {e}")
        return

    # 获取相机序列号并排序
    cam_keys = [k for k in f_cam.keys() if k.startswith('images_')]
    serials = sorted([k.split('_')[1] for k in cam_keys])
    
    if not serials:
        print("未找到相机数据")
        return

    # --- 降采样逻辑 ---
    master_serial = serials[0]
    all_timestamps = f_cam[f'timestamps_{master_serial}'][:]
    
    # [::2] 进行隔帧采样，将 60fps 降为 30fps
    # target_timestamps = all_timestamps[::2]
    target_timestamps = all_timestamps
    
    # --- 舍弃最后一帧 ---
    num_valid_frames = len(target_timestamps) - 1
    
    if num_valid_frames < 1:
        print("数据量不足，无法构建 (State, Action) 对")
        return

    # 准备输出目录
    frames_dir = os.path.join(episode_path, "frames")
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    
    for i in range(len(serials)):
        os.makedirs(os.path.join(frames_dir, f"cam{i}"), exist_ok=True)

    # 准备数据容器
    data_dict = {
        "joints": [],
        "end_pose": [],
        "joints_actions": [],    
        "end_pose_actions": [],  
        "timestamps": []
    }

    robot_timestamps = f_robot['timestamps'][:]
    robot_joints = f_robot['joints'][:]
    robot_ee = f_robot['end_pose'][:] 

    print(f"  - 处理中: 原始 {len(all_timestamps)} -> 降采样 {len(target_timestamps)} -> 有效对 {num_valid_frames}")

    # --- 主循环 ---
    for i in range(num_valid_frames):
        curr_ts = target_timestamps[i]
        next_ts = target_timestamps[i+1]

        # --- A. 保存图像 (修改部分：Resize 224x224) ---
        for cam_idx, serial in enumerate(serials):
            if serial == master_serial:
                ts_dset = f_cam[f'timestamps_{serial}'][:]
                curr_img_idx = find_nearest_idx(ts_dset, curr_ts)
            else:
                ts_dset = f_cam[f'timestamps_{serial}'][:]
                curr_img_idx = find_nearest_idx(ts_dset, curr_ts)
            
            # 1. 从 H5 读取原始二进制数据
            raw_data = f_cam[f'images_{serial}'][curr_img_idx]
            
            # 2. 解码图像 (Binary -> Image Array)
            # np.frombuffer 将字节流转换为 numpy 数组，cv2.imdecode 将其解码为图像矩阵
            img_array = cv2.imdecode(np.frombuffer(raw_data, np.uint8), cv2.IMREAD_COLOR)

            if img_array is None:
                # 如果解码失败（空数据或格式错误），创建一个全黑图像防止报错
                print(f"警告: 帧 {i} 相机 {cam_idx} 解码失败，使用全黑图像填充。")
                img_resized = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                # 3. Resize 到 224x224
                img_resized = cv2.resize(img_array, (224, 224), interpolation=cv2.INTER_AREA)

            # 4. 保存图像
            save_path = os.path.join(frames_dir, f"cam{cam_idx}", f"{i:05d}.jpg")
            cv2.imwrite(save_path, img_resized)

        # --- B. 处理机械臂数据 (State & Action) ---
        r_idx_curr = find_nearest_idx(robot_timestamps, curr_ts)
        r_idx_next = find_nearest_idx(robot_timestamps, next_ts)
        
        r_idx_curr = min(max(r_idx_curr, 0), len(robot_timestamps) - 1)
        r_idx_next = min(max(r_idx_next, 0), len(robot_timestamps) - 1)

        data_dict["joints"].append(robot_joints[r_idx_curr])
        data_dict["end_pose"].append(robot_ee[r_idx_curr])
        
        data_dict["joints_actions"].append(robot_joints[r_idx_next])
        data_dict["end_pose_actions"].append(robot_ee[r_idx_next])
        
        data_dict["timestamps"].append(curr_ts)

        if i % 50 == 0:
            print(f"    已处理 {i}/{num_valid_frames} 帧", end='\r')

    # ==========================================
    # --- 后处理: 追加停留帧 (NEW) ---
    # ==========================================
    HOLD_FRAMES = 20  # 追加的停留帧数
    
    post_process_append_hold(
        data_dict=data_dict,
        frames_dir=frames_dir,
        num_cams=len(serials),
        n_repeat=HOLD_FRAMES
    )
    
    # 更新有效总帧数 (这是关键，否则H5的属性会和实际数据长度不符)
    final_num_frames = len(data_dict["timestamps"])

    # 5. 保存 H5
    output_h5_path = os.path.join(episode_path, "robot_data_aligned.h5")
    with h5py.File(output_h5_path, 'w') as f_out:
        f_out.create_dataset('joints', data=np.array(data_dict["joints"]))
        f_out.create_dataset('end_pose', data=np.array(data_dict["end_pose"]))
        f_out.create_dataset('timestamps', data=np.array(data_dict["timestamps"]))
        f_out.create_dataset('joints_actions', data=np.array(data_dict["joints_actions"]))
        f_out.create_dataset('end_pose_actions', data=np.array(data_dict["end_pose_actions"]))
        
        f_out.attrs['num_frames'] = final_num_frames
        f_out.attrs['fps'] = 30
        f_out.attrs['image_size'] = "224x224" 
        f_out.attrs['description'] = "Downsampled to 30fps. Resized to 224x224. Actions are states at t+1."

    f_cam.close()
    f_robot.close()
    print(f"\n  - 完成！有效帧数: {num_valid_frames} (已舍弃末尾帧)")

def main():
    parser = argparse.ArgumentParser(description="对齐机器人与相机数据 (30fps & Action Generation & Resize)")
    parser.add_argument('--data_dir', type=str, required=True, help="包含episode文件夹的根目录")
    parser.add_argument('--start_episode', type=int, default=0)

    args = parser.parse_args()

    episode_dirs = glob(os.path.join(args.data_dir, "episode*"))
    episode_dirs = sorted(episode_dirs, key=lambda x: int(x.split("episode")[-1]) if x.split("episode")[-1].isdigit() else 0)

    episode_dirs = episode_dirs[args.start_episode:]

    if not episode_dirs:
        print(f"在 {args.data_dir} 未找到 'episode*' 文件夹")
        return

    print(f"找到 {len(episode_dirs)} 个 episode，开始处理...")
    
    for ep_dir in episode_dirs:
        process_episode(ep_dir)

    print("\n所有处理任务完成。")

if __name__ == "__main__":
    main()