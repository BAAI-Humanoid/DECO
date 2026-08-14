import os
import torch
import pandas as pd
from models.tokenizer import HFEmbedder
from tqdm import tqdm

data_path = './dataset/robotwin/Randomized'
t5_model_path = './models/t5_base'
token_max_length = 64
prompt_type = 'jsonl' # jsonl or parquet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t5 = HFEmbedder(t5_model_path, max_length=token_max_length, torch_dtype=torch.bfloat16).to(device)


for task in tqdm(os.listdir(data_path)):
    task_path = os.path.join(data_path, task)
    if not os.path.isdir(task_path):
        continue

    prompt_file = os.path.join(task_path, 'meta', f'tasks.{prompt_type}')
    save_dir = os.path.join(task_path, 'prompt')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if prompt_type == 'parquet':
        prompts = pd.read_parquet(prompt_file)
    elif prompt_type == 'jsonl':
        prompts = pd.read_json(prompt_file, lines=True)
    for i in range(len(prompts)):
        task_idx = prompts.iloc[i][prompts.columns[0]]
        prompt = prompts.iloc[i][prompts.columns[1]]
        with torch.inference_mode():
            outputs, mask = t5(prompt)
            outputs = outputs.squeeze().detach().cpu()
            mask = mask.squeeze().detach().cpu()
            save_dict = {'tokens': outputs, 'mask': mask}
        torch.save(save_dict, os.path.join(save_dir, f'{str(task_idx).zfill(4)}.pt'))
    

