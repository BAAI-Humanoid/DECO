import torch
import importlib
from PIL import Image
from torchvision.transforms import v2 as transforms


class letterbox():
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
    
def preprocess(img1, img2, obs, tac1, tac2, yaml_config, letterbox_flag=False):
    # norm obs
    obs = torch.tensor(obs, dtype=torch.float32)
    norm_type = yaml_config['data']['norm_type']
    if norm_type == 'mean_std':
        obs_mean = torch.tensor(yaml_config['data']['observation_mean'])
        obs_std = torch.tensor(yaml_config['data']['observation_std']).clamp_min(1e-8)
        obs = (obs - obs_mean) / obs_std
    else:
        obs_min = torch.tensor(yaml_config['data']['observation_min'])
        obs_max = torch.tensor(yaml_config['data']['observation_max'])
        obs = (obs - obs_min) / (obs_max - obs_min)
        obs = obs.clamp(0, 1.0)
    obs = obs.unsqueeze(0) # (b, 28)

    # norm tactile
    tac_max_l = yaml_config['data']['tac_left_max']
    tac_max_r = yaml_config['data']['tac_right_max']
    tac1 = torch.from_numpy(tac1 / tac_max_l).float().clamp(0, 1.0)
    tac2 = torch.from_numpy(tac2 / tac_max_r).float().clamp(0, 1.0)
    tac1, tac2 = tac1.unsqueeze(0), tac2.unsqueeze(0)

    # preprocess image
    img_config = yaml_config['img']
    if letterbox_flag:
        resize = letterbox(img_config['img_size'][0])
    else:
        resize = transforms.Resize(img_config['img_size'])
    test_transform = transforms.Compose([
        resize,
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(
            mean=img_config['img_mean'],
            std=img_config['img_std'])
    ]) 
    img1 = Image.fromarray(img1)
    img1 = test_transform(img1).unsqueeze(0) # pil2tensor and unsqueeze (b, 3, h, w)
    img2 = Image.fromarray(img2)
    img2 = test_transform(img2).unsqueeze(0) # pil2tensor and

    return img1, img2, obs, tac1, tac2


def postprocess(action, yaml_config):
    norm_type = yaml_config['data']['norm_type']
    if norm_type == 'mean_std':
        action_mean = torch.tensor(yaml_config['data']['action_mean'])
        action_std = torch.tensor(yaml_config['data']['action_std']).clamp_min(1e-8)
        action = action * action_std[None, :] + action_mean[None, :]
    else:
        action_min = torch.tensor(yaml_config['data']['action_min'])
        action_max = torch.tensor(yaml_config['data']['action_max'])
        action = action * (action_max - action_min)[None, :] + action_min[None, :]
    return action


def predict_action(model, device, yaml_config, img1, img2, obs, task_idx=0, tac1=None, tac2=None):
    task_idx = torch.tensor(task_idx, dtype=torch.long).unsqueeze(0).to(device)
    if yaml_config['img']['img_size'] == [256, 256]:
        letterbox = True
    else:
        letterbox = False
    with torch.no_grad():
        img1, img2, obs, tac1, tac2 = preprocess(img1, img2, obs, tac1, tac2, yaml_config, letterbox_flag=letterbox)
        img1, img2, obs, tac1, tac2 = img1.to(device), img2.to(device), obs.to(device), tac1.to(device), tac2.to(device)
        action = model(img1, img2, obs=obs, act=None, task_idx=task_idx, tac1=tac1, tac2=tac2, action_mask=None, training=False)
        action = action.cpu().squeeze(0) # (1, chunksize, dim) --> (chunksize, dim)
        action = postprocess(action, yaml_config)  # (chunksize, dim)
        
    return action


def modeling(yaml_config): 
    model_name = yaml_config['model_name']
    importmodule = importlib.import_module(f"models.{model_name}")
    model = importmodule.modeling(**yaml_config['model']) 
    return model

class ACTTemporalEnsembler:
    def __init__(self, temporal_ensemble_coeff: float, chunk_size: int) -> None:

        self.chunk_size = chunk_size
        self.ensemble_weights = torch.exp(-temporal_ensemble_coeff * torch.arange(chunk_size))
        self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0)
        self.reset()

    def reset(self):
        """Resets the online computation variables."""
        self.ensembled_actions = None
        # (chunk_size,) count of how many actions are in the ensemble for each time step in the sequence.
        self.ensembled_actions_count = None

    def update(self, actions: torch.Tensor) -> torch.Tensor:
        self.ensemble_weights = self.ensemble_weights.to(device=actions.device)
        self.ensemble_weights_cumsum = self.ensemble_weights_cumsum.to(device=actions.device)
        if self.ensembled_actions is None:
            # Initializes `self._ensembled_action` to the sequence of actions predicted during the first
            # time step of the episode.
            self.ensembled_actions = actions.clone()
            # Note: The last dimension is unsqueeze to make sure we can broadcast properly for tensor
            # operations later.
            self.ensembled_actions_count = torch.ones(
                (self.chunk_size, 1), dtype=torch.long, device=self.ensembled_actions.device
            )
        else:
            # self.ensembled_actions will have shape (batch_size, chunk_size - 1, action_dim). Compute
            # the online update for those entries.
            self.ensembled_actions *= self.ensemble_weights_cumsum[self.ensembled_actions_count - 1]
            self.ensembled_actions += actions[:, :-1] * self.ensemble_weights[self.ensembled_actions_count]
            self.ensembled_actions /= self.ensemble_weights_cumsum[self.ensembled_actions_count]
            self.ensembled_actions_count = torch.clamp(self.ensembled_actions_count + 1, max=self.chunk_size)
            # The last action, which has no prior online average, needs to get concatenated onto the end.
            self.ensembled_actions = torch.cat([self.ensembled_actions, actions[:, -1:]], dim=1)
            self.ensembled_actions_count = torch.cat(
                [self.ensembled_actions_count, torch.ones_like(self.ensembled_actions_count[-1:])]
            )
        # "Consume" the first action.
        action, self.ensembled_actions, self.ensembled_actions_count = (
            self.ensembled_actions[:, 0],
            self.ensembled_actions[:, 1:],
            self.ensembled_actions_count[1:],
        )
        return action

