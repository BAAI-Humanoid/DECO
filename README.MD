# DECO: Decoupled Multimodal Diffusion Transformer for Bimanual Dexterous Manipulation with a Plugin Tactile Adapter
<p>
  <a href="https://arxiv.org/pdf/2602.05513">
    <img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv">
  </a>
  <a href="https://baai-humanoid.github.io/DECO-webpage/">
    <img src="https://img.shields.io/badge/Project-Website-blue">
  </a>
  <a href="https://huggingface.co/datasets/BAAI-Humanoid/DECO-50">
    <img src="https://img.shields.io/badge/HuggingFace-Data-yellow?logo=huggingface">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-Apache--2.0-green">
  </a>
</p>

# Quick start Example 
## 1. Installation
The code is tested on python3.10 and pytorch2.6, cuda12.4.
```
pip install -r requirements.txt
```

## 2. Prepare for Dataset
### Using the DECO-50 Dataset
- Download the dataset
```bash
hf download BAAI-Humanoid/DECO-50 --repo-type dataset --local-dir ./
```
- Batch unzip files

<details>
<summary><b>Click to expand full command for unzip</b></summary>

```bash
#!/usr/bin/env bash
set -euo pipefail

SRC_ROOT="./BAAI-Humanoid/DECO-50"
DST_ROOT="./BAAI-Humanoid/DECO-50-unpack"

PARALLEL_JOBS=8


mkdir -p "$DST_ROOT"

unpack_one() {
    tar_file="$1"
    rel="${tar_file#$SRC_ROOT/}"
    rel_dir="${rel%.tar.gz}"

    out_dir="$DST_ROOT/$rel_dir"

    if [[ -d "$out_dir" ]]; then
        echo "Skip existing $rel_dir"
        return
    fi

    mkdir -p "$(dirname "$out_dir")"
    echo "Unpacking $rel_dir"

    tar -xzf "$tar_file" -C "$(dirname "$out_dir")"
}

export -f unpack_one
export SRC_ROOT DST_ROOT

find "$SRC_ROOT" -type f -name "*.tar.gz" -print0 \
| xargs -0 -n 1 -P "$PARALLEL_JOBS" bash -c 'unpack_one "$@"' _

echo "All episodes unpacked."
```
</details>

- Merge the dataset
```bash
# This will combine all sub-tasks under a task into a single directory.
python utils/merge.py
```
- Computing dataset statistics
```bash
python utils/cal_mean_std.py --data-path ./Deco-50/task1_merged --save-path ./Deco-50/task1_statistics
```
- Transfer the computed statistics to the corresponding fields in [`config/`](https://github.com/1343744768/IL_training_codebase/blob/master/config/deco.yaml#L25).
- Alternatively, you may use the precomputed statistics provided with the DECO-50 dataset.
### Using Your Custom Dataset. 
- If tactile data is unavailable, please modify [`dataset.py`](https://github.com/1343744768/IL_training_codebase/blob/master/dataset.py#L98) by assigning dummy values to `tac1` and `tac2` before returning them. Additionally, set [`use_tactile`](https://github.com/1343744768/IL_training_codebase/blob/master/config/deco.yaml#L8) to `False`.
- Your own dataset should follow the structure below:
```
Dataset/
├── episode_000000/
│   ├── colors/
│   │   ├── 000_color_0.jpg        # left cam
│   │   ├── 000_color_1.jpg        # right cam
│   │   └── ...
│   ├── tactiles/
│   │   ├── 000_left_ee_tactile.npy    # left tactile sensor
│   │   ├── 000_right_ee_tactile.npy   # right tactile sensor
│   │   └── ...
│   └── data.pkl
├── episode_000001/
└── ...
```
- `data.pkl` should contain the following keys:
```python
index: int,           # Index of the data sample
left_action: list,    # Left arm or hand action sequence
right_action: list,   # Right arm or hand action sequence
head_action: list,    # Active head action sequence
left_obs: list,       # Left arm or hand observation sequence
right_obs: list,      # Right arm or hand observation sequence
head_obs: list,       # Active head observation sequence
condition_indx: int,  # Index of the task condition
```

## 3. Hyperparameters for models
- Configuration files for different models (e.g., ACT, DP, DECO) are provided in the [`config`](https://github.com/1343744768/IL_training_codebase/tree/master/config) directory.
- Below we describe the key configuration parameters for DECO.
<details>
<summary><b>Click to expand full YAML configuration for DECO</b></summary>

```yaml
model_name: deco              # Importlib module name for model definition
model:
  action_dim: 28              # Action dimension
  chunk_size: 32              # Temporal chunk size
  obs_state: True             # Whether to use proprioceptive observation states
  use_task_condition: False   # Whether to enable one-hot task conditioning
  num_tasks: 10               # Number of tasks for one-hot embedding
  use_tactile: False          # Whether to use tactile observations
  plugin: False               # Enable plugin tactile adapter for vision-based pretrained policy. If True, specify `pretrain_model_path`
  num_attn_blocks: 6          # Number of transformer attention blocks
  inf_step: 5                 # Number of flow-matching inference steps
  heads: 8                    # Number of attention heads
  dim: 512                    # Model hidden dimension
  rope_axes_dim: [256, 256]   # Rotary positional embedding dimensions for [height, width]
  img_pretrain: /root/.cache/torch/hub/checkpoints/resnet34-b627a593.pth  # ImageNet-pretrained backbone path
  freeze_backbone: False      # Whether to freeze the image encoder
  pretrain_model_path: False  # Path to vision-based pretrained policy
  adapter_model_path: False   # Path to plugin tactile adapter (for deployment)

data:
  chunk_size: 32              # Temporal chunk size
  tac_left_max: 3486.0        # Maximum value of left tactile sensor
  tac_right_max: 4050.0       # Maximum value of right tactile sensor
  norm_type: 'mean_std'       # Normalization type: mean_std or min_max
  observation_mean: [...]     # Observation mean (computed via utils/cal_mean_std.py)
  observation_std: [...]      # Observation standard deviation
  observation_min: [...]      # Observation minimum (for min-max normalization)
  observation_max: [...]      # Observation maximum (for min-max normalization)
  action_mean: [...]          # Action mean
  action_std: [...]           # Action standard deviation
  action_min: [...]           # Action minimum
  action_max: [...]           # Action maximum

img:
  img_size: [256, 256]        # [height, width]. Resize input images to this resolution
  img_mean: [0.485, 0.456, 0.406]  # ImageNet mean
  img_std: [0.229, 0.224, 0.225]   # ImageNet standard deviation
```
</details>

## 4. Training
We train our models on 8 NVIDIA A100 GPUs (40GB each), with 200 epochs per task.

- **Multi-GPU training**
```bash
torchrun --nproc_per_node 8 \
  train.py \
  --distributed True \
  --amp True \
  --device_id '0,1,2,3,4,5,6,7' \
  --config ./config/deco.yaml \
  --data ./Deco-50/task1_merged/ \
  --lr 1e-4 \
  --lr_f 5e-6 \
  --batch_size 1024 \
  --num-workers 16 \
  --epochs 200 \
  --logs ./logs/deco_task1/ \
  --save_period 10
```
- **Single-GPU training**
```bash
python train.py
  --distributed False \
  --amp False \
  --device_id '0' \
  --config ./config/deco.yaml \
  --data ./Deco-50/task1_merged \
  --lr 1e-4 \
  --lr_f 5e-6 \
  --batch_size 128 \
  --num-workers 16 \
  --epochs 500 \
  --logs ./logs/deco_task1 \
  --save_period 50
```

## 5. Deploy

### Hardware Setup
- 3D print the active camera in `assets/active_camera`
- Using [Dynamixel Wizard](https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_wizard2/) to set motor yaw with ID-1 and motor pitch with ID-2. Plug motors into active camera and mount it on H1-2
- Connect PC and H1-2.

### Software Setup
- Follow the guide in [xr_teleoperate](https://github.com/unitreerobotics/xr_teleoperate) to install requirements for deployment.
- Install [Dynamixel SDK](https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_sdk/overview/)

### Launch Server on Robot (same as xr_teleoperate)
- Connect to unitree H1_2
```shell
# robot ip
ssh unitree@192.168.123.167
```
```shell
# launch imager server
cd teleimager
conda activate teleimager
python -m teleimager.image_server
```
```shell
# launch hand server
cd inspire_hand_ws
python inspire_hand_sdk/example/Headless_driver_double.py
```

### Deploy on Host
- Modify the checkpoint in deploy_h1.py
```shell
# using ifconfig to check the network-interface connected with robot. eg, enx9c69d30201e2
cd deploy
python deploy_h1.py --network-interface=enx9c69d30201e2
```

# Citation
If you find this project useful for your research, please consider citing our paper:

```bibtex
@article{li2026deco,
  title={DECO: Decoupled Multimodal Diffusion Transformer for Bimanual Dexterous Manipulation with a Plugin Tactile Adapter},
  author={Xukun Li and Yu Sun and Lei Zhang and Bosheng Huang and Yibo Peng and Yuan Meng and Haojun Jiang and Shaoxuan Xie and Guacai Yao and Alois Knoll and Zhenshan Bing and Xinlong Wang and Zhenguo Sun},
  journal={arXiv preprint arXiv:2602.05513},
  year={2026}
}
```
# License
This project is released under the Apache 2.0 license.

## Contact
We are hiring!!! Full-time researchers, engineers, interns and PhD students are all open for recruitment. If you are interested in working with us on whole-body mobile manipulation for humanoid robots, please contact hitsunzhenguo@gmail.com.
