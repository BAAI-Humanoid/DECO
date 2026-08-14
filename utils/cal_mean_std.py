import os
import json
import yaml
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import v2 as transforms
from torch.utils.data import DataLoader, Dataset


"""
Calculate mean and standard deviation for the dataset
1. For images, observations, and actions, calculate mean and standard deviation
2. For tactile data, calculate min and max due to the presence of many zeros
"""

def list_representer(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)


class my_Dataset(Dataset):
    def __init__(self, data_dir):
        self.transform = transforms.Compose([
                    transforms.Resize((315, 560)),  # height, width 
                    transforms.ToTensor()])
        self.data_dir = data_dir
        self.img_path_list = []
        for episode in os.listdir(self.data_dir):
            img_path = os.path.join(self.data_dir, episode, 'colors')
            for img in os.listdir(img_path):
                if not img.endswith('.jpg'):
                    continue
                self.img_path_list.append(os.path.join(episode, 'colors', img))


    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.img_path_list[idx])
        img = Image.open(img_path)
        img = self.transform(img)
        return img



def cal_img_mean_std(data_path):
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_images = 0
    data = my_Dataset(data_dir=data_path)
    loader = DataLoader(data, batch_size=32, shuffle=False, num_workers=4, pin_memory=False, drop_last=False)
    print(len(data))

    for images in tqdm(loader):
        batch_samples = images.size(0)
        images = images.view(batch_samples, 3, -1)

        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += batch_samples

    mean /= total_images
    std /= total_images

    print("Image Mean:", mean)
    print("Image Std:", std)
    return mean, std



def _welford_update(mean, M2, n, batch_data):
    """Welford's online algorithm: update mean & M2 with a batch of data."""
    batch_mean = batch_data.mean(0)
    batch_var = batch_data.var(0)
    total_n = n + batch_data.size(0)
    delta = batch_mean - mean
    mean_new = mean + delta * batch_data.size(0) / total_n
    M2_new = M2 + batch_var * batch_data.size(0) + delta ** 2 * n * batch_data.size(0) / total_n
    return mean_new, M2_new, total_n


def cal_all_statistics(data_path):
    """Calculate all statistics (tactile, state, action) in a single pass."""
    episodes = os.listdir(data_path)

    # --- tactile ---
    tac_min_left, tac_max_left = float('inf'), float('-inf')
    tac_min_right, tac_max_right = float('inf'), float('-inf')

    # --- state obs (left/right arm+ee: 13 each; head: 2) ---
    s_mean_left = torch.zeros(13)
    s_M2_left = torch.zeros(13)
    s_mean_right = torch.zeros(13)
    s_M2_right = torch.zeros(13)
    s_mean_head = torch.zeros(2)
    s_M2_head = torch.zeros(2)
    s_n = 0
    s_min_left = torch.full((13,), float('inf'))
    s_max_left = torch.full((13,), float('-inf'))
    s_min_right = torch.full((13,), float('inf'))
    s_max_right = torch.full((13,), float('-inf'))
    s_min_head = torch.full((2,), float('inf'))
    s_max_head = torch.full((2,), float('-inf'))

    # --- action obs ---
    a_mean_left = torch.zeros(13)
    a_M2_left = torch.zeros(13)
    a_mean_right = torch.zeros(13)
    a_M2_right = torch.zeros(13)
    a_mean_head = torch.zeros(2)
    a_M2_head = torch.zeros(2)
    a_n = 0
    a_min_left = torch.full((13,), float('inf'))
    a_max_left = torch.full((13,), float('-inf'))
    a_min_right = torch.full((13,), float('inf'))
    a_max_right = torch.full((13,), float('-inf'))
    a_min_head = torch.full((2,), float('inf'))
    a_max_head = torch.full((2,), float('-inf'))

    for episode in tqdm(episodes, desc='Processing episodes'):
        # --- tactile ---
        tactile_path = os.path.join(data_path, episode, 'tactiles')
        if os.path.isdir(tactile_path):
            for tactile in os.listdir(tactile_path):
                if not tactile.endswith('.npy'):
                    continue
                npy = np.load(os.path.join(tactile_path, tactile))
                if 'left' in tactile:
                    tac_min_left = min(tac_min_left, npy.min().item())
                    tac_max_left = max(tac_max_left, npy.max().item())
                else:
                    tac_min_right = min(tac_min_right, npy.min().item())
                    tac_max_right = max(tac_max_right, npy.max().item())

        # --- json data (states & actions) ---
        json_path = os.path.join(data_path, episode, 'data.json')
        if not os.path.isfile(json_path):
            continue
        data = json.load(open(json_path, 'r'))
        records = data['data']
        N = len(records)

        # ---- states ----
        s_left_list, s_right_list, s_head_list = [], [], []
        # ---- actions ----
        a_left_list, a_right_list, a_head_list = [], [], []

        for rec in records:
            # states
            st = rec['states']
            s_left_list.append(st['left_arm']['qpos'] + st['left_ee']['qpos'])
            s_right_list.append(st['right_arm']['qpos'] + st['right_ee']['qpos'])
            s_head_list.append(st['head']['qpos'])
            # actions
            ac = rec['actions']
            a_left_list.append(ac['left_arm']['qpos'] + ac['left_ee']['qpos'])
            a_right_list.append(ac['right_arm']['qpos'] + ac['right_ee']['qpos'])
            a_head_list.append(ac['head']['qpos'])

        # batch tensors (states)
        s_left_t = torch.tensor(s_left_list, dtype=torch.float32)
        s_right_t = torch.tensor(s_right_list, dtype=torch.float32)
        s_head_t = torch.tensor(s_head_list, dtype=torch.float32)

        s_mean_left, s_M2_left, _ = _welford_update(s_mean_left, s_M2_left, s_n, s_left_t)
        s_mean_right, s_M2_right, _ = _welford_update(s_mean_right, s_M2_right, s_n, s_right_t)
        s_mean_head, s_M2_head, _ = _welford_update(s_mean_head, s_M2_head, s_n, s_head_t)
        s_n += N

        s_min_left = torch.minimum(s_min_left, s_left_t.min(dim=0).values)
        s_max_left = torch.maximum(s_max_left, s_left_t.max(dim=0).values)
        s_min_right = torch.minimum(s_min_right, s_right_t.min(dim=0).values)
        s_max_right = torch.maximum(s_max_right, s_right_t.max(dim=0).values)
        s_min_head = torch.minimum(s_min_head, s_head_t.min(dim=0).values)
        s_max_head = torch.maximum(s_max_head, s_head_t.max(dim=0).values)

        # batch tensors (actions)
        a_left_t = torch.tensor(a_left_list, dtype=torch.float32)
        a_right_t = torch.tensor(a_right_list, dtype=torch.float32)
        a_head_t = torch.tensor(a_head_list, dtype=torch.float32)

        a_mean_left, a_M2_left, _ = _welford_update(a_mean_left, a_M2_left, a_n, a_left_t)
        a_mean_right, a_M2_right, _ = _welford_update(a_mean_right, a_M2_right, a_n, a_right_t)
        a_mean_head, a_M2_head, _ = _welford_update(a_mean_head, a_M2_head, a_n, a_head_t)
        a_n += N

        a_min_left = torch.minimum(a_min_left, a_left_t.min(dim=0).values)
        a_max_left = torch.maximum(a_max_left, a_left_t.max(dim=0).values)
        a_min_right = torch.minimum(a_min_right, a_right_t.min(dim=0).values)
        a_max_right = torch.maximum(a_max_right, a_right_t.max(dim=0).values)
        a_min_head = torch.minimum(a_min_head, a_head_t.min(dim=0).values)
        a_max_head = torch.maximum(a_max_head, a_head_t.max(dim=0).values)

    # --- finalize ---
    s_std_left = torch.sqrt(s_M2_left / s_n)
    s_std_right = torch.sqrt(s_M2_right / s_n)
    s_std_head = torch.sqrt(s_M2_head / s_n)

    a_std_left = torch.sqrt(a_M2_left / a_n)
    a_std_right = torch.sqrt(a_M2_right / a_n)
    a_std_head = torch.sqrt(a_M2_head / a_n)

    # concatenate left+right for arm+ee, then append head
    state_mean = torch.cat([s_mean_left, s_mean_right, s_mean_head])
    state_std = torch.cat([s_std_left, s_std_right, s_std_head])
    action_mean = torch.cat([a_mean_left, a_mean_right, a_mean_head])
    action_std = torch.cat([a_std_left, a_std_right, a_std_head])

    state_min_left_right = s_min_left.tolist() + s_min_right.tolist() + s_min_head.tolist()
    state_max_left_right = s_max_left.tolist() + s_max_right.tolist() + s_max_head.tolist()
    action_min_left_right = a_min_left.tolist() + a_min_right.tolist() + a_min_head.tolist()
    action_max_left_right = a_max_left.tolist() + a_max_right.tolist() + a_max_head.tolist()

    # --- print summary ---
    print(f'Tactile Left  min={tac_min_left:.4f}  max={tac_max_left:.4f}')
    print(f'Tactile Right min={tac_min_right:.4f}  max={tac_max_right:.4f}')
    print(f'State Mean:   {state_mean}')
    print(f'State Std:    {state_std}')
    print(f'Action Mean:  {action_mean}')
    print(f'Action Std:   {action_std}')

    return {
        'tac_left_max': tac_max_left,
        'tac_right_max': tac_max_right,
        'observation_mean': state_mean,
        'observation_std': state_std,
        'observation_min': state_min_left_right,
        'observation_max': state_max_left_right,
        'action_mean': action_mean,
        'action_std': action_std,
        'action_min': action_min_left_right,
        'action_max': action_max_left_right,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='/ssd/yusun/datasets/deco50/total_data/')
    parser.add_argument('--save-path', type=str, default='/ssd/yusun/datasets/deco50/')
    args = parser.parse_args()

    # img_mean, img_std = cal_img_mean_std(args.data_path)
    stats = cal_all_statistics(args.data_path)

    yaml.add_representer(list, list_representer)
    with open(os.path.join(args.save_path, 'data_statistics.yaml'), 'w') as f:
        all_data = {'data': {
            'tac_left_max': stats['tac_left_max'],
            'tac_right_max': stats['tac_right_max'],
            'observation_mean': stats['observation_mean'].tolist(),
            'observation_std': stats['observation_std'].tolist(),
            'observation_min': stats['observation_min'],
            'observation_max': stats['observation_max'],
            'action_mean': stats['action_mean'].tolist(),
            'action_std': stats['action_std'].tolist(),
            'action_min': stats['action_min'],
            'action_max': stats['action_max'],
        }}
        # all_data['img'] = {
        #     'img_mean': img_mean.tolist(),
        #     'img_std': img_std.tolist(),
        # }
        """
        We recommend using the following values for image normalization:
        ImageNet standard values
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        """
        yaml.dump(all_data, f)