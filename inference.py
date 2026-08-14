import torch
from PIL import Image
from torchvision.transforms import v2 as transforms


def preprocess(imgs, obs, tacs, yaml_config):
    """
    Args:
        imgs: list of numpy arrays, each [H, W, C]
        obs: numpy array [obs_dim]
        tacs: list of numpy arrays or None
        yaml_config: dict
    Returns:
        imgs: [1, n_view, 3, H, W]
        obs: [1, obs_dim]
        tacs: [1, n_view, ...] or None
    """
    n_view = yaml_config['model'].get('n_view', len(imgs))
    assert len(imgs) == n_view, f"Expected {n_view} images, got {len(imgs)}"

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
        obs = 2 * (obs - obs_min) / (obs_max - obs_min) - 1
    obs = obs.unsqueeze(0)  # [1, obs_dim]

    # preprocess image
    data_config = yaml_config['data']
    test_transform = transforms.Compose([
        transforms.Resize(data_config['img_size']),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(
            mean=data_config['img_mean'],
            std=data_config['img_std'])
    ])
    processed_imgs = []
    for img in imgs:
        img = Image.fromarray(img)
        img = test_transform(img).unsqueeze(0)  # [1, 3, H, W]
        processed_imgs.append(img) # [i, 1, 3, H, W]
    imgs = torch.stack(processed_imgs, dim=1)  # [1, n_view, 3, H, W]

    # norm tactile
    if tacs is not None:
        tac_transform = transforms.Compose([
            transforms.Resize(data_config['tac_img_size']),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(
                mean=data_config['tac_img_mean'],
                std=data_config['tac_img_std'])
        ])
        processed_tacs = []
        for tac in tacs:
            tac = Image.fromarray(tac)
            tac = tac_transform(tac).unsqueeze(0)  # [1, 3, H, W]
            processed_tacs.append(tac) # [i, 1, 3, H, W]
        tacs = torch.stack(processed_tacs, dim=1)  # [1, n_view, 3, H, W]
    else:
        tacs = None

    return imgs, obs, tacs


def postprocess(action, yaml_config):
    norm_type = yaml_config['data']['norm_type']
    if norm_type == 'mean_std':
        action_mean = torch.tensor(yaml_config['data']['action_mean'])
        action_std = torch.tensor(yaml_config['data']['action_std']).clamp_min(1e-8)
        action = action * action_std[None, :] + action_mean[None, :]
    else:
        action_min = torch.tensor(yaml_config['data']['action_min'])
        action_max = torch.tensor(yaml_config['data']['action_max'])
        action = (action + 1) / 2 * (action_max - action_min)[None, :] + action_min[None, :]
    return action


def predict_action(model, device, yaml_config, imgs, obs, task_idx=0, tacs=None):
    """
    Args:
        model: DECO model
        device: torch.device
        yaml_config: dict
        imgs: list of numpy arrays, each [H, W, C]
        obs: numpy array [obs_dim]
        task_idx: int
        tacs: list of numpy arrays or None
    Returns:
        action: [chunksize, act_dim]
    """
    task_idx = torch.tensor(task_idx, dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        imgs, obs, tacs = preprocess(imgs, obs, tacs, yaml_config)
        imgs, obs = imgs.to(device), obs.to(device)
        tacs = tacs.to(device) if tacs is not None else None
        action = model(imgs, obs=obs, act=None, task_idx=task_idx, tacs=tacs, action_mask=None, training=False)
        action = action.cpu().squeeze(0) # (1, chunksize, dim) --> (chunksize, dim)
        action = postprocess(action, yaml_config)  # (chunksize, dim)
        
    return action


def modeling(yaml_config): 
    from models import modeling as _modeling
    model = _modeling(**yaml_config['model']) 
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

