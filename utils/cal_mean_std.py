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



def cal_tactile_min_max(data_path):
    min_left = 100
    max_left = 0
    min_right = 100
    max_right = 0
    for episode in tqdm(os.listdir(data_path)):
        tactile_path = os.path.join(data_path, episode, 'tactiles')
        for tactile in os.listdir(tactile_path):
            if not tactile.endswith('.npy'):
                continue
            if "left" in tactile:
                try:
                    npy = np.load(os.path.join(tactile_path, tactile))
                except:
                    print("Error loading: ", os.path.join(tactile_path, tactile))
                    raise
                if npy.min() < min_left:
                    min_left = npy.min()
                if npy.max() > max_left:
                    max_left = npy.max()
            else:
                npy = np.load(os.path.join(tactile_path, tactile))
                if npy.min() < min_right:
                    min_right = npy.min()
                if npy.max() > max_right:
                    max_right = npy.max()
    print("Min_left:", min_left)
    print("Max_left:", max_left)
    print("Min_right:", min_right)
    print("Max_right:", max_right)
    return min_left, max_left, min_right, max_right


def cal_head_mean_std(data_path, mode='states'):
    mean = torch.zeros(2)
    M2 = torch.zeros(2)
    n = 0

    print('Cal Mode: ', mode)
    head_min = torch.ones(2) * 2 * np.pi   
    head_max = torch.ones(2) * -2 * np.pi  
    for episode in tqdm(os.listdir(data_path)):
        json_path = os.path.join(data_path, episode, 'data.json')
        data = json.load(open(json_path, 'r'))
        batch_data = len(data['data'])
        head_tensor = torch.zeros(batch_data, 2)
        for i in range(batch_data):
            head = data['data'][i][mode]['head']['qpos']
            head_tensor[i] = torch.tensor(head)
        
        total_n = n + batch_data

        batch_mean = head_tensor.mean(0)
        batch_var = head_tensor.var(0)
        delta = batch_mean - mean
        mean += delta * batch_data / total_n
        M2 = M2 + batch_var * batch_data + delta**2 * n * batch_data / total_n

        batch_min = head_tensor.min(dim=0).values  
        head_min = torch.minimum(head_min, batch_min)  
        batch_max = head_tensor.max(dim=0).values  
        head_max = torch.maximum(head_max, batch_max)  

        n = total_n
    var = M2 / n
    std = torch.sqrt(var)
    print('Head Mean: ', mean)
    print('Head Std: ', std)
    print('Head Min: ', head_min)
    print('Head Max: ', head_max)
    return mean, std, head_min, head_max


def cal_obs_mean_std(data_path, mode='states'):
    mean_left = torch.zeros(13)
    M2_left = torch.zeros(13)
    mean_right = torch.zeros(13)
    M2_right = torch.zeros(13)

    print('Cal Mode: ', mode)

    left_min = torch.ones(13) * 2 * np.pi
    right_min = torch.ones(13) * 2 * np.pi
    left_max = torch.ones(13) * -2 * np.pi
    right_max = torch.ones(13) * -2 * np.pi
    n = 0
    for episode in tqdm(os.listdir(data_path)):
        json_path = os.path.join(data_path, episode, 'data.json')
        data = json.load(open(json_path, 'r'))
        batch_data = len(data['data'])
        left_tensor = torch.zeros(batch_data, 13)
        right_tensor = torch.zeros(batch_data, 13)
        for i in range(batch_data):
            left_arm = data['data'][i][mode]['left_arm']['qpos']
            left_ee = data['data'][i][mode]['left_ee']['qpos']
            right_arm = data['data'][i][mode]['right_arm']['qpos']
            right_ee = data['data'][i][mode]['right_ee']['qpos']
            
            left = torch.tensor(left_arm + left_ee)
            right = torch.tensor(right_arm + right_ee)
            left_tensor[i] = left
            right_tensor[i] = right 
            
        total_n = n + batch_data

        batch_mean_left = left_tensor.mean(0)
        batch_var_left = left_tensor.var(0)
        delta_left = batch_mean_left - mean_left
        mean_left += delta_left * batch_data / total_n
        M2_left = M2_left + batch_var_left * batch_data + delta_left**2 * n * batch_data / total_n


        batch_mean_right = right_tensor.mean(0)
        batch_var_right = right_tensor.var(0)
        delta_right = batch_mean_right - mean_right
        mean_right += delta_right * batch_data / total_n
        M2_right = M2_right + batch_var_right * batch_data + delta_right**2 * n * batch_data / total_n

        batch_min_left = left_tensor.min(dim=0).values
        left_min = torch.minimum(left_min, batch_min_left)
        batch_max_left = left_tensor.max(dim=0).values
        left_max = torch.maximum(left_max, batch_max_left)

        batch_min_right = right_tensor.min(dim=0).values
        right_min = torch.minimum(right_min, batch_min_right)
        batch_max_right = right_tensor.max(dim=0).values
        right_max = torch.maximum(right_max, batch_max_right)

        n = total_n

    var_left = M2_left / n
    var_right = M2_right / n
    std_left = torch.sqrt(var_left)
    std_right = torch.sqrt(var_right)

    mean_arm_and_hand = torch.cat((mean_left, mean_right), 0)
    std_arm_and_hand = torch.cat((std_left, std_right), 0)
    print('Mean: ', mean_arm_and_hand)
    print('Std: ', std_arm_and_hand)
    print('Min: ', left_min, right_min)
    print('Max: ', left_max, right_max)
    return mean_arm_and_hand, std_arm_and_hand, left_min, left_max, right_min, right_max


if __name__ == '__main__':
    arser = argparse.ArgumentParser()
    arser.add_argument('--data-path', type=str, default='/root/toy_datasets/task1/')
    arser.add_argument('--save-path', type=str, default='./')
    args = arser.parse_args()
    data_path = args.data_path
    
    # img_mean, img_std = cal_img_mean_std(data_path)
    tactile_min_left, tactile_max_left, tactile_min_right, tactile_max_right = cal_tactile_min_max(data_path)
    state_mean_arm_and_hand, state_std_arm_and_hand, state_min_left, state_max_left, state_min_right, state_max_right = cal_obs_mean_std(data_path, mode='states')
    state_mean_head, state_std_head, state_min_head, state_max_head = cal_head_mean_std(data_path, mode='states')
    action_mean_arm_and_hand, action_std_arm_and_hand, action_min_left, action_max_left, action_min_right, action_max_right = cal_obs_mean_std(data_path, mode='actions')
    action_mean_head, action_std_head, action_min_head, action_max_head = cal_head_mean_std(data_path, mode='actions')

    state_mean = torch.cat((state_mean_arm_and_hand, state_mean_head), 0)
    state_std = torch.cat((state_std_arm_and_hand, state_std_head), 0)
    action_mean = torch.cat((action_mean_arm_and_hand, action_mean_head), 0)
    action_std = torch.cat((action_std_arm_and_hand, action_std_head), 0)

    yaml.add_representer(list, list_representer)
    with open(os.path.join(args.save_path, 'data_statistics.yaml'), 'w') as f:
        all_data = {}
        all_data['data'] = {
            'tac_left_max': tactile_max_left.item(),
            'tac_right_max': tactile_max_right.item(),
            'observation_mean': state_mean.tolist(),
            'observation_std': state_std.tolist(),
            'observation_min': state_min_left.tolist() + state_min_right.tolist() + state_min_head.tolist(),
            'observation_max': state_max_left.tolist() + state_max_right.tolist() + state_max_head.tolist(),
            'action_mean': action_mean.tolist(),
            'action_std': action_std.tolist(),
            'action_min': action_min_left.tolist() + action_min_right.tolist() + action_min_head.tolist(),
            'action_max': action_max_left.tolist() + action_max_right.tolist() + action_max_head.tolist(),
        }

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