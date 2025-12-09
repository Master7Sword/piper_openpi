from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from pathlib import Path
from tqdm import tqdm
import shutil
import h5py
import cv2
import re
import numpy as np 

def convert_dataset():

    # --- 1. 定义常量和路径 ---
    original_data_dir = Path("/home/tengenx2204/workspace/mozihao/Data/put_item_in_drawer")
    new_dataset_root = Path("/home/tengenx2204/workspace/mozihao/Data/")
    repo_id = "put_item_in_drawer_lerobot"
    new_dataset_path = new_dataset_root / repo_id

    print(f"源数据目录: {original_data_dir}")
    print(f"将要创建的新数据集目录: {new_dataset_path}")

    # --- 2. 清理旧的数据集 (如果存在) ---
    if new_dataset_path.exists():
        print(f"警告: 目录 {new_dataset_path} 已存在，将被删除以进行全新转换。")
        shutil.rmtree(new_dataset_path)

    # --- 3. 定义features ---
    # 确定图像的尺寸
    try:
        first_demo_path = sorted(list(original_data_dir.glob("episode*")))[0]
        first_image_path = sorted(list((first_demo_path / "frames" / "cam0").glob("*.jpg")))[0]
        first_image = cv2.imread(str(first_image_path))
        if first_image is None:
            raise IOError(f"无法读取图像文件: {first_image_path}")
        img_shape = first_image.shape
        print(f"检测到图像尺寸为: {img_shape}")
    except (IndexError, IOError) as e:
        print(f"错误: 无法自动检测图像尺寸。请检查源数据目录结构。")
        print(e)
        return

    features = {
        "observation.images.left": {"shape": img_shape, "dtype": "image"},
        "observation.images.right": {"shape": img_shape, "dtype": "image"},
        "observation.images.top": {"shape": img_shape, "dtype": "image"},
        "observation.state": {"shape": (32,), "dtype": "float32"},
        "actions": {"shape": (32,), "dtype": "float32"},
    }

    # --- 4. 创建空的 LeRobotDataset ---
    print("正在创建空的 LeRobotDataset...")
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=new_dataset_path, 
        features=features,
        fps=30,
        image_writer_threads=8,
        image_writer_processes=4,
    )
    print("数据集创建成功。")

    # --- 5. 遍历、转换并保存每个 episode ---
    demo_paths = sorted(
        list(original_data_dir.glob("episode*")),
        key=lambda p: int(re.search(r'(\d+)', p.name).group(1))
    )
    print(f"找到 {len(demo_paths)} 个 episode 待转换。")

    for episode_idx, demo_path in enumerate(tqdm(demo_paths, desc="转换 Episodes")):
        # 为每个 episode 显式地创建一个干净的 buffer
        dataset.episode_buffer = dataset.create_episode_buffer(episode_index=episode_idx)

        h5_path = demo_path / "robot_data_aligned.h5"
        frames_dir = demo_path / "frames"

        with h5py.File(h5_path, 'r') as hf:
            actions = hf['joints_actions'][:, ]
            states = hf['joints'][:, ]  
            num_frames = actions.shape[0]

            left_images = sorted(list((frames_dir / "cam0").glob("*.jpg")))
            right_images = sorted(list((frames_dir / "cam2").glob("*.jpg")))
            top_images = sorted(list((frames_dir / "cam1").glob("*.jpg")))  ## pay attention to the order of cameras!!!

            # 验证帧数是否一致
            assert num_frames == len(right_images), f"帧数不匹配: {demo_path}"
            assert num_frames == len(top_images), f"帧数不匹配: {demo_path}, len(top_images)={len(top_images)}, num_frames={num_frames}"
            assert num_frames == len(states), f"帧数不匹配: {demo_path}"

            for i in range(num_frames):       
                # 1. 获取原始数据
                raw_state = states[i]  
                raw_action = actions[i] 

                # 2. 创建 32 维的 float32 全零数组
                padded_state = np.zeros(32, dtype=np.float32)
                padded_action = np.zeros(32, dtype=np.float32)

                # 3. 将原始数据填充到前 N 位 
                dim = len(raw_state)
                padded_state[:dim] = raw_state
                padded_action[:dim] = raw_action

                if episode_idx <= 68:
                    prompt = "put the yellow block into the top drawer"
                elif episode_idx <= 117:
                    prompt = "put the yellow block into the second drawer"
                elif episode_idx <= 166:
                    prompt = "put the yellow block into the third drawer"

                frame = {
                    "task": prompt,  ## 这里记得改！！
                    "observation.state": padded_state,
                    "actions": padded_action,   
                    "observation.images.left": cv2.cvtColor(cv2.imread(str(left_images[i])), cv2.COLOR_BGR2RGB),        
                    "observation.images.right": cv2.cvtColor(cv2.imread(str(right_images[i])), cv2.COLOR_BGR2RGB),
                    "observation.images.top": cv2.cvtColor(cv2.imread(str(top_images[i])), cv2.COLOR_BGR2RGB),
                }
                dataset.add_frame(frame)

        dataset.save_episode()

    print("所有 episodes 转换并保存完毕。")

if __name__ == "__main__":
    convert_dataset()