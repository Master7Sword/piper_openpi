# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project integrates Physical Intelligence's OpenPI vision-language-action (VLA) models with the Piper dual-arm robotic manipulator. The workflow consists of:
1. Collecting teleoperation demonstrations with the Piper robot and RealSense cameras
2. Converting collected data to LeRobot format
3. Fine-tuning π₀.₅ models on custom tasks
4. Running real-time inference on the physical robot

The repository contains both the OpenPI source code (in `openpi/`) and custom Piper-specific scripts for data collection, processing, and inference.

## Environment Setup

This project uses `uv` for Python dependency management:

```bash
# Install dependencies (inside openpi/ directory)
cd openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# Apply transformers patches for PyTorch support (if needed)
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
```

**Important**: The project has a virtual environment at `openpi/.venv` for the OpenPI package. Commands should be run from the appropriate directory.

## Hardware Setup Commands

### Activate Robot Arms
Before collecting data or running inference, activate the Piper dual-arm system:

```bash
bash can_multi_activate.sh
```

Expected output should show successful CAN bus initialization for both arms (can_arm1 and can_arm2).

### Test Robot Connection
```bash
python piper_test.py
```

### Test Camera Connection
```bash
python multi-cam.py
# Or use RealSense viewer
realsense-viewer
```

## Data Collection & Processing Pipeline

### 1. Collect Teleoperation Data
```bash
python piper_data_collect.py --task_name <task_name> --start_episode <N> --fps 30
```

- Data is saved to `/home/tengenx2204/workspace/mozihao/Data/<task_name>/`
- Creates separate HDF5 files for robot data and camera data
- Use `--start_episode` to resume from a specific episode number
- `--fps` controls camera capture frequency

### 2. Align Robot and Camera Data
```bash
python align.py --data_dir /home/tengenx2204/workspace/mozihao/Data/<task_name> --start_episode <N>
```

This script:
- Synchronizes camera frames with robot states using timestamps
- Downsamples images to 224x224 and saves as JPEG in `/frames` subdirectories
- Records next state as current action (standard RL convention)
- Optional: Line 112 can downsample by 50% if uncommented

### 3. Convert to LeRobot Format
```bash
python convert.py
```

**Important**: Edit `convert_dataset()` function to set:
- `original_data_dir`: Path to aligned dataset
- `repo_id`: Output directory name for converted dataset

### 4. Replay/Verify Data
```bash
# Replay collected episode
python piper_dual_replay.py --replay_episode_dir /path/to/episode

# Replay with end-effector control mode
python replay_ee.py --replay_episode_dir /path/to/episode
```

## Training

### Compute Normalization Statistics
Must run before training:

```bash
uv run scripts/compute_norm_stats.py --config-name <config_name>
```

Examples: `pi05_put_item_in_drawer`, `pi05_open_drawer_lora`, `pi05_libero`

### Run Fine-Tuning

JAX training (default):
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py <config_name> --exp-name=<experiment_name> --overwrite
```

PyTorch training:
```bash
# Single GPU
uv run scripts/train_pytorch.py <config_name> --exp_name <run_name> --save_interval <N>

# Multi-GPU (single node)
uv run torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <run_name>

# Resume training
uv run scripts/train_pytorch.py <config_name> --exp_name <run_name> --resume
```

- Checkpoints saved to `openpi/checkpoints/<config_name>/<exp_name>/`
- Training progress logged to Weights & Biases
- Use `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` to allow JAX to use 90% of GPU memory

## Inference

### Real-Time Robot Inference
```bash
python infer_piper_dual.py \
    --checkpoint_dir <path_to_checkpoint> \
    --config_name <config_name> \
    --mode joint
```

Examples:
```bash
# Put item in drawer task
python infer_piper_dual.py \
    --checkpoint_dir /home/tengenx2204/workspace/mozihao/piper_openpi/openpi/checkpoints/pi05_open_drawer_full/29999 \
    --config_name pi05_put_item_in_drawer \
    --mode joint

# Pick block task
python infer_piper_dual.py \
    --checkpoint_dir /home/tengenx2204/workspace/mozihao/piper_openpi/openpi/checkpoints/pi05_pick_block/29999 \
    --config_name pi05_pick_block \
    --mode joint
```

**Important Safety Notes**:
- Robot initializes to home position ~5 seconds after script starts
- Model loading takes ~30 seconds
- Kill the process (Ctrl+C) immediately if collision or task completion occurs
- Ensure primary (teleoperation) arm is unpowered during inference

**Mode Options**:
- `joint`: Joint space control (default, more stable)
- `ee`: End-effector/Cartesian control

### Inference on Recorded Episodes
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python infer_piper_dual_replay.py \
    --checkpoint_dir <path_to_checkpoint> \
    --replay_episode_dir <path_to_episode> \
    --config_name <config_name> \
    --mode joint
```

### Policy Server (Remote Inference)
Start a policy server:
```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=<config_name> \
    --policy.dir=<checkpoint_dir>
```

Server listens on port 8000 by default. See `openpi/docs/remote_inference.md` for client implementation details.

## Architecture

### Action Space
The Piper dual-arm system uses a 14-dimensional action space:
- `[0:6]`: Right arm joint angles (can_arm1)
- `[6]`: Right gripper position
- `[7:13]`: Left arm joint angles (can_arm2)
- `[13]`: Left gripper position

Actions are in radians, converted to milliradians (×1000) for robot commands.

### Data Flow
1. **Collection**: `piper_data_collect.py` uses multiprocessing for asynchronous camera and robot data capture
2. **Alignment**: `align.py` synchronizes streams via binary search on timestamps
3. **Conversion**: `convert.py` creates LeRobot dataset with image observations, proprioceptive state, and actions
4. **Training**: OpenPI trains transformer-based VLA models with vision-language inputs
5. **Inference**: `infer_piper_dual.py` runs closed-loop control at ~60 Hz

### OpenPI Package Structure
- `src/openpi/models/`: JAX model implementations (π₀, π₀-FAST, π₀.₅)
- `src/openpi/models_pytorch/`: PyTorch model implementations
- `src/openpi/policies/`: Policy wrappers and input/output transforms
- `src/openpi/training/`: Training configs, data loaders, optimizers
- `src/openpi/serving/`: Policy server implementation
- `examples/`: Platform-specific examples (DROID, ALOHA, LIBERO, UR5)

### Key Configuration Files
Training configs are defined in `openpi/src/openpi/training/config.py`:
- Dataset configs (e.g., `LeRobotLiberoDataConfig`)
- Policy I/O transforms (e.g., `LiberoInputs`, `LiberoOutputs`)
- Training hyperparameters (e.g., `pi05_libero`, `pi05_droid`)

## Task-Specific Details

### Initial Robot Positions (infer_piper_dual.py:~80)
Uncomment the appropriate initialization for your task:
- **Open drawer**: Lines 76-77 or 78-79
- **Put item in drawer**: Lines 80-81
- **Pick block**: Lines 82-83

### Prompts (infer_piper_dual.py:~64)
Modify the `prompt` field in the observation dictionary:
- Use `yellow`, `red` for in-distribution objects
- Use `blue`, `green`, `car`, `pig`, `cow`, etc. for OOD generalization (less stable)
- Specify drawer level: `top`, `second`, `third`

Example: `"put the yellow block into the second drawer"`

## Common Issues

### Memory Errors During Training
- Set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` or higher
- Use `--fsdp-devices <N>` for multi-GPU memory distribution
- Consider disabling EMA in config

### Robot Not Moving After Inference Start
- Kill and restart the script
- Check that can_multi_activate.sh completed successfully
- Verify teleoperation arm is unpowered

### Data Alignment Issues
- Ensure camera and robot timestamps are synchronized
- Check that all episodes have matching camera and robot HDF5 files
- Verify image frames are saved in correct `/frames` subdirectory

### Model Loading Errors
- Run `scripts/compute_norm_stats.py` before training
- Verify checkpoint path points to correct iteration directory (e.g., `29999/`)
- For PyTorch, ensure transformers patches were applied

## Testing Commands

```bash
# Test robot-only
python piper_test.py

# Test cameras-only
python multi-cam.py

# Verify dataset
python piper_dual_replay.py --replay_episode_dir <path>

# Test model without robot (OpenPI)
cd openpi
uv run examples/simple_client/README.md  # See for instructions
```

## File Descriptions

### Root Directory Scripts
- `can_multi_activate.sh`: CAN bus activation for dual arms (requires sudo)
- `piper_data_collect.py`: Teleoperation data collection with multiprocessing
- `align.py`: Temporal alignment and JPEG conversion
- `convert.py`: LeRobot dataset conversion
- `infer_piper_dual.py`: Real-time closed-loop inference
- `infer_piper_dual_replay.py`: Offline inference on recorded episodes
- `piper_dual_replay.py`: Replay collected demonstrations
- `utils.py`: Policy loading, action inference, robot control utilities
- `temp.txt`: Command history and episode annotations

### OpenPI Directory (openpi/)
Standard OpenPI repository structure. See `openpi/README.md` for comprehensive documentation on:
- Model architectures and checkpoints
- Fine-tuning workflows
- Remote inference setup
- Docker installation
- Platform-specific examples
