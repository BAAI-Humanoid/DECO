
import torch
import math
import torchvision
import numpy as np
from typing import List
import torch.nn.functional as F
from torch import nn, Tensor
from collections.abc import Callable
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


class DiffusionModel(nn.Module):
    def __init__(
            self, 
            act_dim=28,
            chunk_size=16,
            obs_state=True,
            use_task_condition=False,
            num_tasks=10,
            use_tactile=False,
            img_size=[224, 224], 
            dim_list=[512, 1024, 2048],
            time_dim=128, 
            kernel_size=5, 
            n_groups=8, 
            spatial_keypoints=32, 
            noise_scheduler_type='DDPM',
            train_step=100,
            infer_step=100,
            beta_schedule="squaredcos_cap_v2",
            replace_groupnorm_backbone=False
            ):
        super().__init__()

        assert img_size[0] % 32 == 0 and img_size[1] % 32 == 0, "Image size must be divisible by 32"
        self.train_step = train_step
        self.num_inference_steps = infer_step
        self.chunk_size = chunk_size
        self.act_dim = act_dim
        self.obs_state = obs_state
        self.use_task_condition = use_task_condition
        self.use_tactile = use_tactile
        self.rgb_encoder = ImageEncoder(img_size=img_size, spatial_kps=spatial_keypoints, replace_groupnorm_backbone=replace_groupnorm_backbone)
        global_cond_dim = spatial_keypoints * 4 

        if self.use_task_condition:
            self.task_encoder = nn.Embedding(num_tasks, time_dim)
            global_cond_dim = global_cond_dim + time_dim

        if self.obs_state:
            global_cond_dim = global_cond_dim + act_dim 
        
        if self.use_tactile:
            global_cond_dim = global_cond_dim + time_dim  # assuming tactile feature dim is time_dim
            self.tactile_encoder = nn.Sequential(
                                    nn.Linear(1062*2, 1024),
                                    nn.Mish(),
                                    nn.Linear(1024, time_dim)
                                    )
            
            
        self.unet = ConditionUnet1d(dim_list=dim_list, act_dim=act_dim, time_dim=time_dim, global_cond_dim=global_cond_dim, kernel_size=kernel_size, n_groups=n_groups)
        if noise_scheduler_type == 'DDPM':
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=train_step,
                beta_schedule=beta_schedule,
                clip_sample=False
            )
        else:
            self.noise_scheduler = DDIMScheduler(
                num_train_timesteps=train_step,
                beta_schedule=beta_schedule,
                clip_sample=False
            )
    
    def forward(self, img1, img2, obs=None, act=None, task_idx=None, tac1=None, tac2=None, action_mask=None, training=True):
        rgb_feat1 = self.rgb_encoder(img1)  # b, spatial_keypoints*2
        rgb_feat2 = self.rgb_encoder(img2)  # b, spatial_keypoints*2
        condition = torch.cat([rgb_feat1, rgb_feat2], dim=-1)

        if self.obs_state:
            condition = torch.cat([obs, condition], dim=-1)

        if self.use_task_condition:
            task_embed = self.task_encoder(task_idx) # b, time_dim
            condition = torch.cat([task_embed, condition], dim=-1)  # b, spatial_keypoints*4+act_dim+time_dim
        
        if self.use_tactile:
            tactile = torch.cat([tac1, tac2], dim=-1)  # b, 1062*2
            tactile = self.tactile_encoder(tactile)  # b, time_dim
            condition = torch.cat([tactile, condition], dim=-1)

        if training:
            timestep = torch.randint(low=0, high=self.train_step, size=(act.shape[0],), device=act.device).long()
            x_t, noise = self.add_noise(act, timestep)
            pred = self.unet(x_t, timestep, condition=condition)
            return pred, noise

        else:
            sample = torch.randn((img1.shape[0], self.chunk_size, self.act_dim), device=img1.device)
            self.noise_scheduler.set_timesteps(self.num_inference_steps)
            for t in self.noise_scheduler.timesteps:
                model_output = self.unet(
                    sample,
                    torch.full(sample.shape[:1], t, dtype=torch.long, device=sample.device),
                    condition=condition,
                )
                # Compute previous image: x_t -> x_t-1
                sample = self.noise_scheduler.step(model_output, t, sample).prev_sample

            return sample
    
    def add_noise(self, x_0: Tensor, t: Tensor | int):
        noise = torch.randn(x_0.shape, device=x_0.device)
        x_t = self.noise_scheduler.add_noise(x_0, noise, t)
        return x_t, noise


class ConditionUnet1d(nn.Module):
    def __init__(self, dim_list=[512, 1024, 2048], act_dim=28, time_dim=128, global_cond_dim=128, kernel_size=5, n_groups=8):
        super().__init__()
        self.time_step_encoder = nn.Sequential(
            PosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.Mish(),
            nn.Linear(time_dim * 4, time_dim),
        )
        cond_dim = time_dim + global_cond_dim
        kwargs = {
            "cond_dim": cond_dim,
            "kernel_size": kernel_size,
            "n_groups": n_groups,
            }
        in_out = [(act_dim, dim_list[0])] + list(
            zip(dim_list[:-1], dim_list[1:], strict=True)
        )
        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(
                nn.ModuleList(
                    [
                        CondResidualBlock1d(dim_in, dim_out, **kwargs),
                        CondResidualBlock1d(dim_out, dim_out, **kwargs),
                        # Downsample as long as it is not the last block.
                        nn.Conv1d(dim_out, dim_out, 3, 2, 1) if not is_last else nn.Identity(),
                    ]))

        self.mid_modules = nn.ModuleList(
            [
                CondResidualBlock1d(dim_list[-1], dim_list[-1], **kwargs),
                CondResidualBlock1d(dim_list[-1], dim_list[-1], **kwargs),
            ])

        self.up_modules = nn.ModuleList([])
        for ind, (dim_out, dim_in) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(
                nn.ModuleList(
                    [
                        CondResidualBlock1d(dim_in * 2, dim_out, **kwargs),
                        CondResidualBlock1d(dim_out, dim_out, **kwargs),
                        nn.ConvTranspose1d(dim_out, dim_out, 4, 2, 1) if not is_last else nn.Identity(),
                    ]))

        self.final_conv = nn.Sequential(
            Conv1dBlock(dim_list[0], dim_list[0], kernel_size=kernel_size),
            nn.Conv1d(dim_list[0], act_dim, 1),
        )
    
    def forward(self, x: Tensor, t: Tensor | int, condition: Tensor | None):
        """
        Args:
            x: (b, seq, dim) input noise action.
            t: (b,) time step.
            condition: (b, dim) global conditioning vector.
        """
        time_emb = self.time_step_encoder(t)  # (b,) ---> (b, dim)
        condition = torch.cat([time_emb, condition], dim=1)

        x = x.transpose(1, 2) # (b, dim, seq)
        encoder_skip_features: list[Tensor] = []
        for conv1, conv2, downsample in self.down_modules:
            x = conv1(x, condition)
            x = conv2(x, condition)
            encoder_skip_features.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, condition)

        for conv1, conv2, upsample in self.up_modules:
            x = torch.cat((x, encoder_skip_features.pop()), dim=1)
            x = conv1(x, condition)
            x = conv2(x, condition)
            x = upsample(x)
        out = self.final_conv(x)  # (b, dim, seq)
        out = out.transpose(1, 2)  # (b, seq, dim)

        return out



class SpatialSoftmax(nn.Module):
    def __init__(self, input_shape, num_kp=32):
        super().__init__()

        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape

        if num_kp is not None:
            self.nets = torch.nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._out_c = num_kp
        else:
            self.nets = None
            self._out_c = self._in_c

        # we could use torch.linspace directly but that seems to behave slightly differently than numpy
        # and causes a small degradation in pc_success of pre-trained models.
        pos_x, pos_y = np.meshgrid(np.linspace(-1.0, 1.0, self._in_w), np.linspace(-1.0, 1.0, self._in_h))
        pos_x = torch.from_numpy(pos_x.reshape(self._in_h * self._in_w, 1)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self._in_h * self._in_w, 1)).float()
        # register as buffer so it's moved to the correct device.
        self.register_buffer("pos_grid", torch.cat([pos_x, pos_y], dim=1))

    def forward(self, features: Tensor) -> Tensor:
        """
        Args:
            features: (B, C, H, W) input feature maps.
        Returns:
            (B, K, 2) image-space coordinates of keypoints.
        """
        if self.nets is not None:
            features = self.nets(features)

        # [B, K, H, W] -> [B * K, H * W] where K is number of keypoints
        features = features.reshape(-1, self._in_h * self._in_w)
        # 2d softmax normalization
        attention = F.softmax(features, dim=-1)
        # [B * K, H * W] x [H * W, 2] -> [B * K, 2] for spatial coordinate mean in x and y dimensions
        expected_xy = attention @ self.pos_grid
        # reshape to [B, K, 2]
        feature_keypoints = expected_xy.view(-1, self._out_c, 2)

        return feature_keypoints


class ImageEncoder(nn.Module):
    """Encodes an RGB image into a 1D feature vector.

    Includes the ability to normalize and crop the image first.
    """

    def __init__(self, img_size: List[int], spatial_kps: int = 32, replace_groupnorm_backbone: bool = False):
        super().__init__()
        if replace_groupnorm_backbone:
            backbone_model = getattr(torchvision.models, "resnet18")(
                weights=None, # "ResNet18_Weights.IMAGENET1K_V1",
            )
            # Note: This assumes that the layer4 feature map is children()[-3]
            # TODO(alexander-soare): Use a safer alternative.
            self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))
            self.backbone = self._replace_submodules(
                    root_module=self.backbone,
                    predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                    func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features),
                )
        else:
            backbone_model = getattr(torchvision.models, "resnet18")(weights="ResNet18_Weights.IMAGENET1K_V1")
            self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))

        feature_map_shape = (512, img_size[0] // 32, img_size[1] // 32)
        self.pool = SpatialSoftmax(feature_map_shape, num_kp=spatial_kps)
        self.feature_dim = spatial_kps * 2
        self.out = nn.Linear(self.feature_dim, self.feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, H, W) image tensor with pixel values in [0, 1].
        Returns:
            (B, D) image feature.
        """
        # Extract backbone feature.
        x = torch.flatten(self.pool(self.backbone(x)), start_dim=1)
        # Final linear layer with non-linearity.
        x = self.relu(self.out(x))
        return x

    def _replace_submodules(
        self, root_module: nn.Module, predicate: Callable[[nn.Module], bool], func: Callable[[nn.Module], nn.Module]
    ) -> nn.Module:
        """
        Args:
            root_module: The module for which the submodules need to be replaced
            predicate: Takes a module as an argument and must return True if the that module is to be replaced.
            func: Takes a module as an argument and returns a new module to replace it with.
        Returns:
            The root module with its submodules replaced.
        """
        if predicate(root_module):
            return func(root_module)

        replace_list = [k.split(".") for k, m in root_module.named_modules(remove_duplicate=True) if predicate(m)]
        for *parents, k in replace_list:
            parent_module = root_module
            if len(parents) > 0:
                parent_module = root_module.get_submodule(".".join(parents))
            if isinstance(parent_module, nn.Sequential):
                src_module = parent_module[int(k)]
            else:
                src_module = getattr(parent_module, k)
            tgt_module = func(src_module)
            if isinstance(parent_module, nn.Sequential):
                parent_module[int(k)] = tgt_module
            else:
                setattr(parent_module, k, tgt_module)
        # verify that all BN are replaced
        assert not any(predicate(m) for _, m in root_module.named_modules(remove_duplicate=True))
        return root_module


class PosEmb(nn.Module):
    """1D sinusoidal positional embeddings as in Attention is All You Need."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Conv1dBlock(nn.Module):
    """Conv1d --> GroupNorm --> Mish"""

    def __init__(self, inp_channels, out_channels, kernel_size=5, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class CondResidualBlock1d(nn.Module):
    """ResNet style 1D convolutional block with FiLM modulation for conditioning."""

    def __init__(self, in_channels: int, out_channels: int, cond_dim: int, kernel_size: int = 5, n_groups: int = 8):
        super().__init__()

        self.out_channels = out_channels
        self.conv1 = Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups)

        # FiLM modulation (https://huggingface.co/papers/1709.07871) outputs per-channel bias and (maybe) scale.
        cond_channels = out_channels * 2 
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, cond_channels))

        self.conv2 = Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups)
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """
        Args:
            x: (B, in_channels, T)
            cond: (B, cond_dim)
        Returns:
            (B, out_channels, T)
        """
        out = self.conv1(x)

        cond_embed = self.cond_encoder(cond).unsqueeze(-1)
        scale = cond_embed[:, : self.out_channels]
        bias = cond_embed[:, self.out_channels :]
        out = scale * out + bias

        out = self.conv2(out)
        out = out + self.residual_conv(x)
        return out


def modeling(
    action_dim, 
    chunk_size, 
    obs_state,
    use_task_condition=False,
    num_tasks=0,
    use_tactile=False,
    img_size=[224, 224], 
    dim_list=[512, 1024, 2048], 
    time_dim=128, 
    kernel_size=5, 
    n_groups=8, 
    spatial_keypoints=32, 
    noise_scheduler_type="DDIM", 
    train_step=100, 
    infer_step=10, 
    beta_schedule="squaredcos_cap_v2", 
    replace_groupnorm_backbone=False,
    pretrain_model_path=None):

    dp = DiffusionModel(
            act_dim=action_dim,
            chunk_size=chunk_size, 
            obs_state=obs_state, 
            use_task_condition=use_task_condition,
            num_tasks=num_tasks,
            use_tactile=use_tactile,
            img_size=img_size, 
            dim_list=dim_list,
            time_dim=time_dim, 
            kernel_size=kernel_size, 
            n_groups=n_groups, 
            spatial_keypoints=spatial_keypoints, 
            noise_scheduler_type=noise_scheduler_type,
            train_step=train_step,
            infer_step=infer_step,
            beta_schedule=beta_schedule,
            replace_groupnorm_backbone=replace_groupnorm_backbone
        )
    if pretrain_model_path:
        print("loading pretrained weights from {}".format(pretrain_model_path))
        model_dict = torch.load(pretrain_model_path, map_location='cpu')
        dp.load_state_dict(model_dict, strict=True)

    return dp


if __name__ == "__main__":
    model = modeling(img_size=(256, 256), infer_step=1, use_task_condition=True, num_tasks=100)
    img1 = torch.randn(1, 3, 256, 256)
    img2= torch.randn(1, 3, 256, 256)
    obs = torch.randn(1, 28)
    action = torch.randn(1, 16, 28)
    task_idx = torch.randint(low=0, high=100, size=(1,))
    pred, noise = model(img1, img2, obs=obs, act=action, task_idx=task_idx, training=True)
    print(pred.shape, noise.shape)

    sample = model(img1, img2, obs=obs, task_idx=task_idx, training=False)
    print(sample.shape)