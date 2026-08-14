import os
import yaml
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms

class univtac_Dataset(Dataset):
    def __init__(self, data_dir, train=True, transform=None, **args):
        self.img_list = []
        self.root = data_dir
        self.transform = transform
        self.norm_type = args['norm_type']
        self.chunksize = args['chunk_size']
        self.n_view = args.get('n_view', 2)
        self.lable_dict = {} 

        self.obs_mean = torch.tensor(args['observation_mean'])
        self.obs_std = torch.tensor(args['observation_std']).clamp_min(1e-8)
        self.obs_min = torch.tensor(args['observation_min'])
        self.obs_max = torch.tensor(args['observation_max'])

        self.action_mean = torch.tensor(args['action_mean'])
        self.action_std = torch.tensor(args['action_std']).clamp_min(1e-8)
        self.action_min = torch.tensor(args['action_min'])
        self.action_max = torch.tensor(args['action_max'])

        episode_list = os.listdir(data_dir)
        total_episodes = len(episode_list)
        random.seed(42)
        random.shuffle(episode_list)
        if train:
            episode_list = episode_list[:int(total_episodes)]
        else:
            episode_list = episode_list[int(total_episodes * 0.9):]  # not bug, just sample some data to observe loss
        
        for episode in episode_list:
            episode_path = os.path.join(data_dir, episode, 'img1')
            self.lable_dict[episode] = os.path.join(episode, 'episode_info.pkl')
            for img in os.listdir(episode_path):
                self.img_list.append(os.path.join(episode, 'img1', img))

        ## tac img transform for univtac
        self.tac_transform = transforms.Compose([
            transforms.Resize(args['tac_img_size']),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(
                mean=args['tac_img_mean'],
                std=args['tac_img_std'])
        ])

    def __getitem__(self, index):
        relat_path = self.img_list[index]
        episode_id, img_name = relat_path.split(os.sep)[0], relat_path.split(os.sep)[-1]
        img_idx = int(img_name.split('frame_')[1].split('.')[0])

        img1_path = os.path.join(self.root, relat_path)
        img2_path = img1_path.replace('img1', 'img2')
        tac1_path = img1_path.replace('img1', 'tac1')
        tac2_path = img1_path.replace('img1', 'tac2')

        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
        tac1 = Image.open(tac1_path)
        tac2 = Image.open(tac2_path)

        seed = random.randint(0, 1000000000)
        if self.transform is not None: # same transform for both images
            self.seed_all(seed)
            img1 = self.transform(img1)
            self.seed_all(seed)
            img2 = self.transform(img2)
            self.seed_all(seed)
            tac1 = self.tac_transform(tac1)
            self.seed_all(seed)
            tac2 = self.tac_transform(tac2)

        # Stack views: [n_view, 3, H, W]
        imgs = torch.stack([img1, img2], dim=0)
        tacs = torch.stack([tac1, tac2], dim=0)
        
        label_pkl = pd.read_pickle(os.path.join(self.root, self.lable_dict[episode_id]))
        task_condition = torch.tensor(label_pkl.iloc[img_idx]['task'], dtype=torch.long)
        obs_state = torch.tensor(label_pkl.iloc[img_idx]['state']).float()
        action = np.stack(label_pkl.iloc[img_idx:]['action'], axis=0)
        action = torch.from_numpy(action).float()

        if action.size(0) < self.chunksize:      # padding action2chunksize and add mask
            padd_len = self.chunksize - action.size(0)
            last_action = action[-1].unsqueeze(0).repeat(padd_len, 1)  # [padd_len, 26]
            action_padd = torch.cat([action, last_action], dim=0)  # [self.chunksize, 26]
            mask = torch.cat([torch.ones(action.size(0), dtype=torch.bool), torch.zeros(padd_len, dtype=torch.bool)], dim=0)
        else:
            action_padd = action[:self.chunksize]  # clamp to chunksize and add mask 
            mask = torch.ones(self.chunksize, dtype=torch.bool)
        
        if self.norm_type == 'min_max':
            obs_state = 2 * (obs_state - self.obs_min) / (self.obs_max - self.obs_min) - 1
            action_padd = 2 * (action_padd - self.action_min[None, :]) / (self.action_max[None, :] - self.action_min[None, :]) - 1
        else:
            obs_state = (obs_state - self.obs_mean) / self.obs_std
            action_padd = (action_padd - self.action_mean[None, :]) / self.action_std[None, :]

        return imgs, tacs, obs_state, action_padd, mask, task_condition

    def __len__(self):
        return len(self.img_list)

    def seed_all(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

class libero_Dataset(Dataset):
    def __init__(self, data_dir, train=True, transform=None, **args):
        self.img_list = []
        self.root = data_dir
        self.transform = transform
        self.norm_type = args['norm_type']
        self.chunksize = args['chunk_size']
        self.n_view = args.get('n_view', 2)
        self.lable_dict = {} 

        self.obs_mean = torch.tensor(args['observation_mean'])
        self.obs_std = torch.tensor(args['observation_std']).clamp_min(1e-8)
        self.obs_min = torch.tensor(args['observation_min'])
        self.obs_max = torch.tensor(args['observation_max'])


        self.action_mean = torch.tensor(args['action_mean'])
        self.action_std = torch.tensor(args['action_std']).clamp_min(1e-8)
        self.action_min = torch.tensor(args['action_min'])
        self.action_max = torch.tensor(args['action_max'])

        episode_list = os.listdir(data_dir)
        total_episodes = len(episode_list)
        random.seed(42)
        random.shuffle(episode_list)
        if train:
            episode_list = episode_list[:int(total_episodes)]
        else:
            episode_list = episode_list[int(total_episodes * 0.9):]  # not bug, just sample some data to observe loss
        
        for episode in episode_list:
            episode_path = os.path.join(data_dir, episode, 'img1')
            self.lable_dict[episode] = os.path.join(episode, 'episode_info.pkl')
            for img in os.listdir(episode_path):
                self.img_list.append(os.path.join(episode, 'img1', img))
        

    def __getitem__(self, index):
        relat_path = self.img_list[index]
        episode_id, img_name = relat_path.split(os.sep)[0], relat_path.split(os.sep)[-1]
        img_idx = int(img_name.split('frame_')[1].split('.')[0])

        img1_path = os.path.join(self.root, relat_path)
        img2_path = img1_path.replace('img1', 'img2')

        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)

        seed = random.randint(0, 1000000000)
        if self.transform is not None: # same transform for both images
            self.seed_all(seed)
            img1 = self.transform(img1)
            self.seed_all(seed)
            img2 = self.transform(img2)

        # Stack views: [n_view, 3, H, W]
        imgs = torch.stack([img1, img2], dim=0)
        tacs = torch.stack([torch.tensor([-1.]), torch.tensor([-1.])], dim=0)
        
        label_pkl = pd.read_pickle(os.path.join(self.root, self.lable_dict[episode_id]))
        task_condition = torch.tensor(label_pkl.iloc[img_idx]['task'], dtype=torch.long)
        obs_state = torch.tensor(label_pkl.iloc[img_idx]['state']).float()
        action = np.stack(label_pkl.iloc[img_idx:]['action'], axis=0)
        action = torch.from_numpy(action).float()

        if action.size(0) < self.chunksize:      # padding action2chunksize and add mask
            padd_len = self.chunksize - action.size(0)
            last_action = action[-1].unsqueeze(0).repeat(padd_len, 1)  # [padd_len, 26]
            action_padd = torch.cat([action, last_action], dim=0)  # [self.chunksize, 26]
            mask = torch.cat([torch.ones(action.size(0), dtype=torch.bool), torch.zeros(padd_len, dtype=torch.bool)], dim=0)
        else:
            action_padd = action[:self.chunksize]  # clamp to chunksize and add mask 
            mask = torch.ones(self.chunksize, dtype=torch.bool)
        
        if self.norm_type == 'min_max':
            obs_state = 2 * (obs_state - self.obs_min) / (self.obs_max - self.obs_min) - 1
            action_padd = 2 * (action_padd - self.action_min[None, :]) / (self.action_max[None, :] - self.action_min[None, :]) - 1
        else:
            obs_state = (obs_state - self.obs_mean) / self.obs_std
            action_padd = (action_padd - self.action_mean[None, :]) / self.action_std[None, :]

        return imgs, tacs, obs_state, action_padd, mask, task_condition

    def __len__(self):
        return len(self.img_list)

    def seed_all(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


class dexjoco_Dataset(Dataset):
    def __init__(self, data_dir, train=True, transform=None, **args):
        self.img_list = []
        self.root = data_dir
        self.transform = transform
        self.norm_type = args['norm_type']
        self.chunksize = args['chunk_size']
        self.n_view = args.get('n_view', 3)
        self.label_dict = {}

        self.obs_mean = torch.tensor(args['observation_mean'])
        self.obs_std = torch.tensor(args['observation_std']).clamp_min(1e-8)
        self.obs_min = torch.tensor(args['observation_min'])
        self.obs_max = torch.tensor(args['observation_max'])

        self.action_mean = torch.tensor(args['action_mean'])
        self.action_std = torch.tensor(args['action_std']).clamp_min(1e-8)
        self.action_min = torch.tensor(args['action_min'])
        self.action_max = torch.tensor(args['action_max'])

        episode_list = os.listdir(data_dir)
        total_episodes = len(episode_list)
        random.seed(42)
        random.shuffle(episode_list)
        if train:
            episode_list = episode_list[:int(total_episodes)]
        else:
            episode_list = episode_list[int(total_episodes * 0.9):]  # not bug, just sample some data to observe loss

        for episode in episode_list:
            episode_path = os.path.join(data_dir, episode, 'img1')
            self.label_dict[episode] = os.path.join(episode, 'episode_info.pkl')
            for img in os.listdir(episode_path):
                self.img_list.append(os.path.join(episode, 'img1', img))

    def __getitem__(self, index):
        relat_path = self.img_list[index]
        episode_id, img_name = relat_path.split(os.sep)[0], relat_path.split(os.sep)[-1]
        img_idx = int(img_name.split('frame_')[1].split('.')[0])

        img1_path = os.path.join(self.root, relat_path)
        img2_path = img1_path.replace('img1', 'img2')
        img3_path = img1_path.replace('img1', 'img3')

        label_pkl = pd.read_pickle(os.path.join(self.root, self.label_dict[episode_id]))
        is_bimanual = bool(label_pkl.iloc[img_idx]['is_bimanual'])

        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        if is_bimanual:
            img3 = Image.open(img3_path).convert('RGB')
        else:
            img3 = img2.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        seed = random.randint(0, 1000000000)
        if self.transform is not None: # same transform for all images
            self.seed_all(seed)
            img1 = self.transform(img1)
            self.seed_all(seed)
            img2 = self.transform(img2)
            self.seed_all(seed)
            img3 = self.transform(img3)

        imgs = torch.stack([img1, img2, img3], dim=0)
        tacs = torch.stack([torch.tensor([-1.]) for _ in range(self.n_view)], dim=0)

        task_condition = torch.tensor(label_pkl.iloc[img_idx]['task'], dtype=torch.long)
        obs_state = torch.tensor(label_pkl.iloc[img_idx]['state']).float()
        action_slice = label_pkl.iloc[img_idx:img_idx + self.chunksize]
        action = np.stack(action_slice['action'], axis=0)
        action = torch.from_numpy(action).float()

        action_len = action.size(0)
        if action_len < self.chunksize:  
            padd_len = self.chunksize - action_len
            last_action = action[-1].unsqueeze(0).repeat(padd_len, 1)
            action = torch.cat([action, last_action], dim=0)
        time_mask = torch.arange(self.chunksize) < action_len

        unified_obs = torch.zeros(46, dtype=torch.float32)
        unified_action = torch.zeros(self.chunksize, 44, dtype=torch.float32)
        obs_mask = torch.zeros(46, dtype=torch.bool)
        action_mask = torch.zeros(44, dtype=torch.bool)
        if is_bimanual:
            unified_obs[:] = obs_state[:46]
            unified_action[:] = action[:, :44]
            obs_mask[:] = True
            action_mask[:] = True
        else:
            unified_obs[:7] = obs_state[:7]
            unified_obs[14:30] = obs_state[7:23]
            unified_action[:, :22] = action[:, :22]
            obs_mask[:7] = True
            obs_mask[14:30] = True
            action_mask[:22] = True

        obs_state = torch.zeros_like(unified_obs)
        action_padd = torch.zeros_like(unified_action)
        if self.norm_type == 'min_max':
            obs_denom = (self.obs_max - self.obs_min).clamp_min(1e-8)
            action_denom = (self.action_max - self.action_min).clamp_min(1e-8)
            obs_state[obs_mask] = 2 * (unified_obs[obs_mask] - self.obs_min[obs_mask]) / obs_denom[obs_mask] - 1
            action_padd[:, action_mask] = (
                2 * (unified_action[:, action_mask] - self.action_min[action_mask])
                / action_denom[action_mask] - 1
            )
        else:
            obs_state[obs_mask] = (
                unified_obs[obs_mask] - self.obs_mean[obs_mask]
            ) / self.obs_std[obs_mask]
            action_padd[:, action_mask] = (
                unified_action[:, action_mask] - self.action_mean[action_mask]
            ) / self.action_std[action_mask]

        mask = time_mask[:, None] & action_mask[None, :]
        return imgs, tacs, obs_state, action_padd, mask, task_condition

    def __len__(self):
        return len(self.img_list)

    def seed_all(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

if __name__ == "__main__":
    data_path = './datasets/univtac_converted/data'
    json_path = './config/deco_univtac_80m.yaml'
    with open(json_path, 'r') as f:
        config = yaml.safe_load(f)
    test_transform = transforms.Compose([
        transforms.Resize(config['data']['img_size']),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(
            mean=config['data']['img_mean'],
            std=config['data']['img_std'])
        ])
    dataset = univtac_Dataset(data_path, True, test_transform, **config['data'])
    print(len(dataset))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)

    from tqdm import tqdm
    for imgs, tacs, obs_state, action_padd, mask, task_condition in tqdm(dataloader, total=len(dataloader)):
        print(imgs.shape, tacs.shape, obs_state.shape, action_padd.shape, mask.shape, task_condition.shape)
        quit()