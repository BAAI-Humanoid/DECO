#!/bin/bash
# `.` is POSIX so this also works under `sh` (dash); `source` is bash/zsh only.
. /root/miniconda3/etc/profile.d/conda.sh
conda activate robotwin
# Make the `conda` *command* available to the scheduler's child processes.
# Use condabin (contains only `conda`, no `python`) so robotwin's python stays first.
export PATH="/root/miniconda3/condabin:$PATH"

# Model paths (single source of truth; consumed by XPolicyLab/policy/DECO/model.py)
export T5_MODEL_PATH=/share/yusun/models/t5_base
export DECO_MODEL_PATH=/share/yusun/models/deco/best.pth

cd /share/yusun/RoboTwin
bash scripts/eval_policy.sh multitask \
   --config env_cfg/eval/all_tasks.yml \
   --policy-name DECO --ckpt-name best \
   --env-cfg-type arx_x5 \
   --task-config demo_randomized \
   --policy-conda-env robotwin --eval-env-conda-env robotwin
# change task-config to demo_clean/demo_randomized