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
  <a href="https://huggingface.co/BAAI-Humanoid/DECO_sim">
    <img src="https://img.shields.io/badge/HuggingFace-Model-red?logo=huggingface">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-Apache--2.0-green">
  </a>
</p>

# Quick Start Example
## 1. Installation
The code is tested on python3.10 and pytorch2.6, cuda12.4.
```bash
pip install -r requirements.txt
```

## 2. Hyperparameters
- Configuration files for DECO are provided in the `config/` directory.

```yaml
model:
  action_dim: 7               # DOF of action
  chunk_size: 16              # chunk size
  n_view: 2                   # Number of camera views
  obs_state: True             # Whether to use proprioceptive states
  obs_dim: 8                  # DOF of proprioception states
  use_task_condition: True    # Whether to enable one-hot task embedding
  num_tasks: 40               # Number of tasks for one-hot embedding
  use_tactile: False          # Whether to use tactile observations
  plugin: False               # Enable plugin tactile adapter for vision-based pretrained policy
  num_attn_blocks: 6          # Number of transformer attention blocks
  inf_step: 5                 # Number of flow-matching inference steps
  heads: 8                    # Number of attention heads
  dim: 512                    # Model hidden dimension
  rope_axes_dim: [256, 256]   # Rotary positional embedding dimensions for [height, width], same to img_size
  pretrain_model_path: False  # Path to vision-based pretrained policy checkpoint (for deployment)
  adapter_model_path: False   # Path to plugin tactile adapter (for deployment)

data:
  chunk_size: 16              # chunksize
  n_view: 2                   # Number of camera views
  tac_left_max: 3486.0        # Maximum value of left tactile sensor
  tac_right_max: 4050.0       # Maximum value of right tactile sensor
  norm_type: 'mean_std'       # Normalization type: mean_std or min_max
  observation_mean: [...]     # Observation mean (computed via utils/cal_mean_std.py)
  observation_std: [...]      # Observation standard deviation
  observation_min: [...]      # Observation minimum 
  observation_max: [...]      # Observation maximum
  action_mean: [...]          # Action mean
  action_std: [...]           # Action standard deviation
  action_min: [...]           # Action minimum
  action_max: [...]           # Action maximum
  img_size: [256, 256]        # [height, width]. Resize input images to this resolution
  img_mean: [0.485, 0.456, 0.406]  # ImageNet mean
  img_std: [0.229, 0.224, 0.225]   # ImageNet standard deviation
```

## 3. Dataset Preparation

### Prepare LIBERO Dataset

- Download the LIBERO dataset from HuggingFace:
```bash
hf download HuggingFaceVLA/libero --repo-type dataset --local-dir ./
```

- Convert the LIBERO data format:
```bash
cd utils
python convert_libero.py \
  --data_path /path/to/libero/data/chunk-000 \
  --save_path /path/to/output 

python task2json.py \
  --input /path/to/libero/meta/tasks.parquet \
  --output /path/to/tasks.json
```

- The processed data structure:
```
datasets/
├── episode_0000/
│   ├── img1/              # External camera
│   │   └── frame_*.png
│   ├── img2/              # Wrist camera
│   │   └── frame_*.png
│   └── episode_info.pkl   # Contains state, action, task index
└── tasks.json              # prompt → index mapping
```

### Prepare DexJoCo Dataset

- Download the official DexJoCo dataset:
```bash
hf download DexJoCo/DexJoCo-Datasets-LeRobot \
  --repo-type dataset \
  --local-dir /path/to/DexJoCo-Datasets-LeRobot
```

- Convert both `rand_obj` and `rand_full` data to the DECO format:
```bash
python utils/convert_dexjoco.py \
  --data_path /path/to/DexJoCo-Datasets-LeRobot \
  --save_path /path/to/deco_dexjoco \
  --workers 32
```

The converter requires `pyarrow` and the `ffmpeg` system command. They can be installed with:
```bash
pip install pyarrow==24.0.0
sudo apt-get install ffmpeg
```

The converted episodes are saved under `/path/to/deco_dexjoco/data`. Each episode contains `img1`, `img2`, optional `img3`, and `episode_info.pkl`.

### Prepare UniVTAC Dataset
- Download the UniVTAC dataset from HuggingFace:
```bash
hf download HuggingFaceVLA/UniVTAC \
  --repo-type dataset \
  --local-dir ./datasets/UniVTAC/
```

- Convert the UniVTAC data format:
```bash
python utils/convert.py \
  --data_path ./datasets/UniVTAC \
  --save_path ./datasets/UniVTAC_converted
```


## 4. Training

Use `--dataset` to select the dataset type: `libero` (default), `dexjoco`, or `univtac`.
Three simulation datasets were trained on 8 × A100 GPUs. The initial learning rate was set to 1 × 10⁻⁴ and decayed to 1 × 10⁻⁶, with a total batch size of 1024. LIBERO was trained for 300 epochs, while the other two simulation datasets were trained for 200 epochs.

- **Multi-GPU training**
```bash
torchrun --nproc_per_node 8 \
  train.py \
  --distributed True \
  --amp True \
  --device_id '0,1,2,3,4,5,6,7' \
  --config ./config/deco_libero_80m.yaml \
  --data /path/to/libero/data \
  --dataset libero \
  --lr 1e-4 \
  --lr_f 1e-6 \
  --batch_size 1024 \
  --num-workers 16 \
  --epochs 300 \
  --logs ./logs/libero_deco/ \
  --save_period 50
```
- **Single-GPU training**
```bash
python train.py \
  --distributed False \
  --amp False \
  --device_id '0' \
  --config ./config/deco_libero_80m.yaml \
  --data /path/to/libero/data \
  --dataset libero \
  --lr 1e-4 \
  --lr_f 1e-6 \
  --batch_size 128 \
  --num-workers 16 \
  --epochs 300 \
  --logs ./logs/libero_deco/ \
  --save_period 50
```

## 5. Evaluation

### Eval LIBERO

- Install LIBERO:
```bash
# NOTE: mujoco version must be 3.3.2
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
# Install the required dependencies and set up the environment according to the official instructions.
cd LIBERO && pip install -e .
```

- Patch LIBERO for high PyTorch versions (`LIBERO/libero/libero/benchmark/__init__.py` line 164):
```python
init_states = torch.load(init_states_path, weights_only=False)
```
- Download weight from huggingface
```bash
hf download BAAI-Humanoid/DECO_sim --repo-type model --local-dir ./sim_ckpt/
```
- Configure `eval_libero.py`: set `yaml_path`, `task_suite_name`, `task_config` and the checkpoint path in [`yaml`](https://github.com/1343744768/IL_training_codebase/blob/main/config/deco_libero_80m.yaml#L16).
- Run evaluation:
```bash
export MUJOCO_GL=egl
python eval_libero.py
```

- LIBERO Results: Success rates (50 trials per task):

| Model | Spatial | Object | Goal | Long | **Average** |
|:-----:|:--------------:|:-------------:|:-----------:|:---------:|:-----------:|
| DECO | 94.2% | 99.4% | 96.8% | 86.4% | **94.2%** |

### Eval DexJoCo

DexJoCo evaluation requires a separate environment because it runs the MuJoCo simulator online. Create the evaluation environment and install the official DexJoCo package:
```bash
git clone https://github.com/brave-eai/dexjoco.git
# Install the required dependencies and set up the environment according to the official instructions.
pip install -e ./dexjoco/dexjoco
```
Download weight from huggingface
```bash
hf download BAAI-Humanoid/DECO_sim --repo-type model --local-dir ./sim_ckpt/
```
Run DexJoCo evaluation with headless EGL rendering:
```bash
export MUJOCO_GL=egl

python eval_dexjoco.py \
  --config ./config/deco_dexjoco_80m.yaml \
  --checkpoint /path/to/best.pth \
  --dexjoco-root /path/to/dexjoco \
  --regime rand_obj \
  --tasks all \
  --save-video
```

The command evaluates all 11 tasks under `rand_obj`. Change `--regime` to `rand_full` for full visual randomization, and omit `--save-video` if rollout videos are not needed.

- DexJoCo Results: Success rates (50 trials per task). `/B` denotes a bimanual task.

| Model | Hammer Nail | Click Mouse | Pick Bucket | Pinch Tongs | Fold Glasses | Water Plant | Unlock iPad /B | Hanoi /B | Assembly /B | Microwave /B | Photograph /B | **Average** |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| DECO (rand-obj) | 68 | 90 | 80 | 46 | 50 | 88 | 40 | 18 | 0 | 64 | 76 | **56.4** |
| DECO (rand-full) | 66 | 82 | 88 | 52 | 52 | 80 | 40 | 24 | 16 | 72 | 66 | **58** |

### Eval UniVTAC
- Install UniVTAC:
> We test our policy on UniVTAC by using the official installation script at commit 05bcd3e
> https://github.com/univtac/UniVTAC

```bash
# manually create a conda environment
conda create -n UniVTAC python=3.10 -y
conda env update -n UniVTAC --file ./third_party/TacEx/source/tacex_uipc/libuipc/conda/env.yaml
conda activate UniVTAC
# install vcpkg
mkdir -p ~/Toolchain && cd ~/Toolchain
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.sh -disableMetrics
# set toolchain environment variable
export CMAKE_TOOLCHAIN_FILE="$HOME/Toolchain/vcpkg/scripts/buildsystems/vcpkg.cmake"

# install using official installation script
cd ~
git clone https://github.com/univtac/UniVTAC.git
cd UniVTAC
bash scripts/install.sh
```

Test two of our policies on UniVTAC:
- Download policy and modify config files:
Download from HuggingFace:
```shell
cd DECO
hf download BAAI-Humanoid/DECO_sim --repo-type model --local-dir ./sim_ckpt/
```
If you use a different path to download the policies, please modify the config files accordingly.
```yaml
#DECO/config/deco_univtac_80m.yaml
model:
  #...
  pretrain_model_path: sim_ckpt/univtac.pth
  adapter_model_path: False

#DECO/config/deco.p_univtac_80m.yaml
model:
  #...
  pretrain_model_path: sim_ckpt/univtac-p.pth
  adapter_model_path: sim_ckpt/univtac-p.pth
```

- Test our policies:
You can evaluate these two policies on all tasks by running the following command:
```shell
cd DECO
python run_evals.py --univtac_path ~/UniVTAC
```
or you can test a specific policy on a specific task:
```shell
cd DECO
python eval_univtac.py \                                    
  --univtac_path ~/UniVTAC \
  --model-config config/deco_univtac_80m.yaml \
  --task_name lift_can
```

- UniVTAC Results: Success rates (100 trials per task):

| Model | grasp_classify | lift_can | lift_bottle | pull_out_key | put_bottle_in_shelf | insert_hole | insert_tube | insert_HDMI | **Average** |
|:-----:|:--------------:|:-------------:|:-----------:|:---------:|:---------:|:---------:|:---------:|:---------:|:-----------:|
| DECO (vision) | 96.0% | 99.0% | 100.0% | 96.0% | 90.0% | 63.0% | 88.0% | 25.0% | **82.1%** |
| DECO.p (+tactile) | 99.0% | 99.0% | 100.0% | 96.0% | 93.0% | 59.0% | 88.0% | 23.0% | **82.1%** |

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
