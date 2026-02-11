import os
import json
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms


class letterbox():
    """
    Resize image by keeping aspect ratio, 
    then pad to make it h x w.
    """
    def __init__(self, size=256, fill=128):
        self.size = size
        self.fill = fill  # padding color, 0 for black

    def __call__(self, img: Image.Image):
        w, h = img.size

        # 计算缩放比例
        scale = self.size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)

        # 缩放图像
        img = img.resize((new_w, new_h), Image.BILINEAR)

        # 计算padding
        pad_w = self.size - new_w
        pad_h = self.size - new_h
        left = pad_w // 2
        top = pad_h // 2
        right = pad_w - left
        bottom = pad_h - top

        img = transforms.functional.pad(img, (left, top, right, bottom), fill=self.fill)
        return img
    
    
class my_Dataset(Dataset):
    def __init__(self, data_dir, train=True, transform=None, **args):
        self.img_list = []
        self.root = data_dir
        self.transform = transform
        self.norm_type = args['norm_type']
        self.chunksize = args['chunk_size']
        self.label_dict = {} 

        self.obs_mean = torch.tensor(args['observation_mean'])
        self.obs_std = torch.tensor(args['observation_std']).clamp_min(1e-8)
        self.obs_min = torch.tensor(args['observation_min'])
        self.obs_max = torch.tensor(args['observation_max'])


        self.action_mean = torch.tensor(args['action_mean'])
        self.action_std = torch.tensor(args['action_std']).clamp_min(1e-8)
        self.action_min = torch.tensor(args['action_min'])
        self.action_max = torch.tensor(args['action_max'])

        
        self.tac_left_max = args['tac_left_max']
        self.tac_right_max = args['tac_right_max']

        episode_list = os.listdir(data_dir)
        total_episodes = len(episode_list)
        random.seed(42)
        random.shuffle(episode_list)
        if train:
            episode_list = episode_list[:int(total_episodes * 0.9)]
        else:
            episode_list = episode_list[int(total_episodes * 0.9):]
        
        for episode in episode_list:
            episode_path = os.path.join(data_dir, episode, 'colors')
            self.label_dict[episode] = pd.read_pickle(os.path.join(data_dir, episode, 'data.pkl'))
            for img in os.listdir(episode_path):
                if img.endswith('0.jpg'):
                    self.img_list.append(os.path.join(episode, 'colors', img))
        

    def __getitem__(self, index):
        relat_path = self.img_list[index]
        episode_id, img_name = relat_path.split(os.sep)[0], relat_path.split(os.sep)[-1]
        img_idx = int(img_name.split('_color')[0])

        img1_path = os.path.join(self.root, relat_path)
        img2_path = img1_path.replace('_0.jpg', '_1.jpg')

        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)

        tac1_path = img1_path.replace('_color_0.jpg', '_left_ee_tactile.npy').replace('colors', 'tactiles')
        tac2_path = img2_path.replace('_color_1.jpg', '_right_ee_tactile.npy').replace('colors', 'tactiles')

        tact1 = np.load(tac1_path) / self.tac_left_max
        tact2 = np.load(tac2_path) / self.tac_right_max

        tact1 = torch.from_numpy(tact1).float()
        tact2 = torch.from_numpy(tact2).float()

        seed = random.randint(0, 1000000000)
        if self.transform is not None: # same transform for both images
            self.seed_all(seed)
            img1 = self.transform(img1)
            self.seed_all(seed)
            img2 = self.transform(img2)

        df_slice = self.label_dict[episode_id].iloc[img_idx:img_idx + self.chunksize]
        obs_state = df_slice.iloc[0]['left_obs'] + df_slice.iloc[0]['right_obs'] + df_slice.iloc[0]['head_obs']
        obs_state = torch.tensor(obs_state)

        action_keys = ['left_action', 'right_action', 'head_action']
        action_list = [np.stack(df_slice[key].values, axis=0) for key in action_keys]
        action = np.concatenate(action_list, axis=1)
        action = torch.from_numpy(action).float()

        task_idx = df_slice.iloc[0]['condition_indx'] if 'condition_indx' in df_slice.columns else -1
        task_idx = torch.tensor(task_idx, dtype=torch.long)

        action_len = action.size(0)
        padd_len = self.chunksize - action_len
        last_action = action[-1].unsqueeze(0).repeat(padd_len, 1) if padd_len > 0 else torch.tensor([], dtype=action.dtype)
        action_padd = torch.cat([action, last_action], dim=0)
        mask = torch.arange(self.chunksize) < action_len


        if self.norm_type == 'min_max':
            obs_state = (obs_state - self.obs_min) / (self.obs_max - self.obs_min)
            action_padd = (action_padd - self.action_min[None, :]) / (self.action_max[None, :] - self.action_min[None, :])
        else:
            obs_state = (obs_state - self.obs_mean) / self.obs_std
            action_padd = (action_padd - self.action_mean[None, :]) / self.action_std[None, :]

        return img1, img2, tact1, tact2, obs_state, action_padd, mask, task_idx

    def __len__(self):
        return len(self.img_list)

    def seed_all(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

if __name__ == "__main__":
    import yaml
    config = yaml.safe_load(open('./IL_training_codebase/config/act.yaml', 'r'))

    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomApply([transforms.ColorJitter(brightness=(0.7, 1.3), contrast=(0.8, 1.2), saturation=(0.8, 1.2))], p=0.5),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(random.choice([3, 5, 7])), sigma=random.uniform(0.1, 2))], p=0.5),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(
            mean=[0.3574, 0.3694, 0.3745], 
            std=[0.2222, 0.2093, 0.1849])
        ])
    data = my_Dataset(data_dir='/root/toy_datasets/data_0107_6', train=False, transform=test_transform, **config['data'])
    train_loader = torch.utils.data.DataLoader(data, batch_size=8, shuffle=True, num_workers=4, pin_memory=False, drop_last=False)
    from tqdm import tqdm
    for index, (img1, img2, tac1, tac2, obs, action, mask, task_idx) in tqdm(enumerate(train_loader), total=len(train_loader)):
        print(img1.shape, img2.shape, tac1.shape, tac2.shape, obs.shape, action.shape, mask.shape, task_idx.shape)
        quit()