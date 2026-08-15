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

# Quick Start
## 1. Installation
The code has been tested with Python 3.12, PyTorch 2.11, and CUDA 13.0. We strongly recommend installing torchcodec by following [this page](https://github.com/meta-pytorch/torchcodec); otherwise, the dataloader will be slow when using the PyAV decoder with a large batch size during training.

```bash
apt update
apt install ffmpeg
conda create -n deco python=3.12
conda activate deco
pip install -r requirements.txt
```

## 2. Configuration
- Configuration files for DECO are provided in the `config/` directory.

```yaml
model:
  action_dim: 14              # DOF of the action (bimanual: 7 per arm)
  chunk_size: 32              # Action chunk size
  n_view: 3                   # Number of camera views
  obs_state: False            # Whether to use proprioceptive states
  obs_dim: 14                 # DOF of proprioception states
  use_prompt: True            # Whether to condition the policy on language prompts (encoded offline)
  use_tactile: False          # Whether to use tactile observations
  plugin: False               # Enable the plugin tactile adapter for a vision-based pretrained policy
  plugin_rank: 32             # Rank of the plugin tactile adapter
  num_attn_blocks: 6          # Number of transformer attention blocks
  inf_step: 5                 # Number of flow-matching inference steps
  heads: 8                    # Number of attention heads
  dim: 512                    # Model hidden dimension
  rope_axes_dim: [192, 256]   # Rotary positional embedding dimensions for [height, width], matching img_size
  pretrain_model_path: False  # Path to the vision-based pretrained policy checkpoint (for deployment)
  adapter_model_path: False   # Path to the plugin tactile adapter (for deployment)

data:
  chunk_size: 32              # Action chunk size
  n_view: 3                   # Number of camera views
  tac_left_max: 1000.0        # Maximum value of the left tactile sensor
  tac_right_max: 1000.0       # Maximum value of the right tactile sensor
  norm_type: 'mean_std'       # Normalization type: mean_std or min_max
  observation_mean: [...]     # Observation mean (computed via utils/cal_mean_std.py)
  observation_std: [...]      # Observation standard deviation
  observation_min: [...]      # Observation minimum
  observation_max: [...]      # Observation maximum
  action_mean: [...]          # Action mean
  action_std: [...]           # Action standard deviation
  action_min: [...]           # Action minimum
  action_max: [...]           # Action maximum
  img_size: [192, 256]        # [height, width]. Input images are resized to this resolution
  img_mean: [0.485, 0.456, 0.406]  # ImageNet mean
  img_std: [0.229, 0.224, 0.225]   # ImageNet standard deviation
```


## 3. Dataset Preparation

### Prepare Robotwin Dataset

- Download the Robotwin dataset from HuggingFace:
```bash
hf download StarVLA/RoboTwin-Randomized --repo-type dataset --local-dir ./
```

- Merge the clean and randomized data:
```bash
for dir in Clean/*/; do
    name=$(basename "$dir")
    mv "$dir" "Randomized/${name}_clean"
done
```
- Download the T5 model and encode prompts offline:
```bash
hf download google-t5/t5-base --local-dir ./
python prompt_encode.py # change the path first
```

- The merged data structure is as follows:
```text
RoboTwin/
├── Randomized/   # LeRobot v2.1 data format combined clean and randomized data
│     adjust_bottle/            # Randomized data
│        data/chunk-000
│        meta/
│        prompt/
│        videos/chunk-000
│     adjust_bottle_clean/      # Clean data
│     ...
│     turn_switch/
│     turn_switch_clean/

# train.py --data should be the path to the Randomized directory
# e.g., train.py --data ./dataset/robotwin/Randomized
```




## 4. Training

- NOTE: If your machine is short on memory, disable `persistent_workers` and set `prefetch_factor=1` with `num_workers=8`; otherwise you risk having DataLoader workers OOM-killed during training.
- **Multi-GPU training**
```bash
torchrun --nproc_per_node 8 \
  train.py \
  --distributed True \
  --amp True \
  --device_id '0,1,2,3,4,5,6,7' \
  --config ./config/DECO_robotwin.yaml \
  --data /path/to/robotwin/randz \
  --lr 1e-4 \
  --lr_f 1e-6 \
  --batch-size 1024 \
  --num-workers 16 \
  --epochs 40 \
  --logs ./logs/ \
  --save_period 10
```
- **Single-GPU training**
```bash
python train.py \
  --distributed False \
  --amp False \
  --device_id '0' \
  --config ./config/DECO_robotwin.yaml \
  --data /path/to/robotwin/data \
  --lr 1e-4 \
  --lr_f 1e-6 \
  --batch-size 128 \
  --num-workers 16 \
  --epochs 40 \
  --logs ./logs/ \
  --save_period 10
```

## 5. Evaluation
- Download our model checkpoint (trained for 40 epochs on the clean + randomized data with the default settings):
```bash
hf download BAAI-Humanoid/DECO_sim --repo-type model --local-dir ./
```
- Clone the official repository and install the RoboTwin dependencies following the instructions in the repository:
```bash
git clone https://github.com/RoboTwin-Platform/RoboTwin.git
```
- RoboTwin was recently updated to support XPolicyLab. For convenience, we support both the new and the legacy evaluation setups.

### RoboTwin latest version
- Move the `DECO/` directory under `robotwin_eval/` to `RoboTwin/XPolicyLab/policy/`.
```
mv robotwin_eval/DECO RoboTwin/XPolicyLab/policy/
```
- Update the path in `robotwin_eval/eval.sh`, then run `bash robotwin_eval/eval.sh`.

### RoboTwin legacy version
- Move the `DECO_o/` directory under `robotwin_eval/` to `RoboTwin/policy/` and rename it to `DECO`.
```
mv robotwin_eval/DECO_o RoboTwin/policy/DECO
```
- Update the paths in `deploy_policy.py`.
- Move `robotwin_eval/eval_all_tasks.py` to the `RoboTwin/script/` directory, then run:
```bash
mv robotwin_eval/eval_all_tasks.py RoboTwin/script/
cd RoboTwin
# ~7 GB memory usage per GPU
python script/eval_all_tasks.py --gpus 0,1,2,3,4,5,6,7 --policy DECO --max-tasks-per-gpu 2
``` 

## 6. Results
DECO achieves these results with a model of only about **80M parameters**. Success rates (%) of DECO on the 50 RoboTwin dual-arm tasks under the **Seen** and **Unseen** settings, with 100 trials per task.

| Task | Seen Clean | Seen Randomized | Seen Avg. | Unseen Clean | Unseen Randomized | Unseen Avg. |
|------|-----------:|----------------:|----------:|-------------:|------------------:|------------:|
| adjust_bottle | 100.00 | 100.00 | 100.00 | 99.00 | 99.00 | 99.00 |
| beat_block_hammer | 96.00 | 86.00 | 91.00 | 98.00 | 90.00 | 94.00 |
| blocks_ranking_rgb | 72.00 | 65.00 | 68.50 | 75.00 | 65.00 | 70.00 |
| blocks_ranking_size | 44.00 | 38.00 | 41.00 | 42.00 | 40.00 | 41.00 |
| click_alarmclock | 95.00 | 88.00 | 91.50 | 93.00 | 92.00 | 92.50 |
| click_bell | 95.00 | 95.00 | 95.00 | 94.00 | 90.00 | 92.00 |
| dump_bin_bigbin | 96.00 | 95.00 | 95.50 | 98.00 | 97.00 | 97.50 |
| grab_roller | 98.00 | 99.00 | 98.50 | 100.00 | 99.00 | 99.50 |
| handover_block | 97.00 | 86.00 | 91.50 | 97.00 | 87.00 | 92.00 |
| handover_mic | 100.00 | 96.00 | 98.00 | 99.00 | 97.00 | 98.00 |
| hanging_mug | 43.00 | 26.00 | 34.50 | 30.00 | 33.00 | 31.50 |
| lift_pot | 100.00 | 100.00 | 100.00 | 100.00 | 99.00 | 99.50 |
| move_can_pot | 85.00 | 83.00 | 84.00 | 77.00 | 79.00 | 78.00 |
| move_pillbottle_pad | 93.00 | 92.00 | 92.50 | 91.00 | 88.00 | 89.50 |
| move_playingcard_away | 100.00 | 96.00 | 98.00 | 98.00 | 99.00 | 98.50 |
| move_stapler_pad | 85.00 | 87.00 | 86.00 | 87.00 | 85.00 | 86.00 |
| open_laptop | 96.00 | 100.00 | 98.00 | 96.00 | 100.00 | 98.00 |
| open_microwave | 88.00 | 84.00 | 86.00 | 85.00 | 86.00 | 85.50 |
| pick_diverse_bottles | 84.00 | 77.00 | 80.50 | 85.00 | 74.00 | 79.50 |
| pick_dual_bottles | 90.00 | 92.00 | 91.00 | 93.00 | 93.00 | 93.00 |
| place_a2b_left | 63.00 | 59.00 | 61.00 | 62.00 | 58.00 | 60.00 |
| place_a2b_right | 53.00 | 47.00 | 50.00 | 49.00 | 50.00 | 49.50 |
| place_bread_basket | 92.00 | 81.00 | 86.50 | 92.00 | 86.00 | 89.00 |
| place_bread_skillet | 91.00 | 81.00 | 86.00 | 91.00 | 83.00 | 87.00 |
| place_burger_fries | 97.00 | 100.00 | 98.50 | 99.00 | 98.00 | 98.50 |
| place_can_basket | 79.00 | 71.00 | 75.00 | 78.00 | 74.00 | 76.00 |
| place_cans_plasticbox | 100.00 | 99.00 | 99.50 | 100.00 | 99.00 | 99.50 |
| place_container_plate | 99.00 | 97.00 | 98.00 | 95.00 | 97.00 | 96.00 |
| place_dual_shoes | 85.00 | 81.00 | 83.00 | 79.00 | 76.00 | 77.50 |
| place_empty_cup | 99.00 | 100.00 | 99.50 | 97.00 | 98.00 | 97.50 |
| place_fan | 92.00 | 93.00 | 92.50 | 93.00 | 99.00 | 96.00 |
| place_mouse_pad | 83.00 | 75.00 | 79.00 | 85.00 | 74.00 | 79.50 |
| place_object_basket | 89.00 | 89.00 | 89.00 | 86.00 | 83.00 | 84.50 |
| place_object_scale | 77.00 | 68.00 | 72.50 | 77.00 | 67.00 | 72.00 |
| place_object_stand | 90.00 | 87.00 | 88.50 | 96.00 | 85.00 | 90.50 |
| place_phone_stand | 83.00 | 85.00 | 84.00 | 77.00 | 83.00 | 80.00 |
| place_shoe | 98.00 | 98.00 | 98.00 | 98.00 | 100.00 | 99.00 |
| press_stapler | 82.00 | 84.00 | 83.00 | 81.00 | 87.00 | 84.00 |
| put_bottles_dustbin | 87.00 | 88.00 | 87.50 | 83.00 | 86.00 | 84.50 |
| put_object_cabinet | 75.00 | 84.00 | 79.50 | 84.00 | 79.00 | 81.50 |
| rotate_qrcode | 85.00 | 93.00 | 89.00 | 85.00 | 87.00 | 86.00 |
| scan_object | 88.00 | 89.00 | 88.50 | 92.00 | 90.00 | 91.00 |
| shake_bottle | 100.00 | 98.00 | 99.00 | 100.00 | 98.00 | 99.00 |
| shake_bottle_horizontally | 100.00 | 99.00 | 99.50 | 100.00 | 99.00 | 99.50 |
| stack_blocks_three | 85.00 | 83.00 | 84.00 | 85.00 | 81.00 | 83.00 |
| stack_blocks_two | 96.00 | 99.00 | 97.50 | 97.00 | 98.00 | 97.50 |
| stack_bowls_three | 83.00 | 83.00 | 83.00 | 81.00 | 78.00 | 79.50 |
| stack_bowls_two | 96.00 | 95.00 | 95.50 | 99.00 | 98.00 | 98.50 |
| stamp_seal | 61.00 | 75.00 | 68.00 | 69.00 | 71.00 | 70.00 |
| turn_switch | 63.00 | 72.00 | 67.50 | 69.00 | 64.00 | 66.50 |
| **Overall** | **86.56** | **84.76** | **85.66** | **86.32** | **84.36** | **85.34** |


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
