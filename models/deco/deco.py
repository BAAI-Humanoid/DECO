import math
import torch
import einops
import numpy as np
from torch import Tensor, nn
from torch.nn import functional as F
from models.deco.img_encoder import ResNet34
from models.deco.denoise_schedular import get_schedule
from models.deco.rope import apply_rotary_emb, RotaryPosEmbed


class DECO(nn.Module):
    def __init__(self, act_dim, chunk_size, obs_state=True, use_tactile=False, plugin=False, plugin_rank=32, use_task_condition=False, num_tasks=10, inf_step=10, img_pretrain=False, num_attn_blocks=6, heads=8, dim=512, rope_axes_dim=[256, 256], freeze_backbone=True):
        super().__init__()
        head_dim = dim // heads
        self.head_dim = head_dim
        self.chunk_size = chunk_size
        self.act_dim = act_dim
        self.obs_state = obs_state
        self.use_tactile = use_tactile
        self.use_task_condition = use_task_condition
        self.inference_step = inf_step
        self.rope = RotaryPosEmbed(head_dim, rope_axes_dim)  # initial mrope embedding
        self.img_encoder = ResNet34()  # img encoder: resnet from ImageNet pretrained model
        self.img_head = nn.Conv2d(512, dim, kernel_size=3, padding=1)
        
        self.pos_idx_embedd = nn.Embedding(2, dim) # img embedding to distinguish img1 and img2
        if self.obs_state:
            self.obs_encoder = nn.Sequential(
                nn.Linear(act_dim, dim),
                nn.Mish(),
                nn.Linear(dim, dim)
            )
        if self.use_tactile:
            # calculate the mean tactile data of each regions(17 regions in inspire hand) in forward pass
            # left: (batch, 1062) --> (batch, 17), right: (batch, 1062) --> (batch, 17)
            self.init_tac_regions()

            # gated tactile
            self.gated = nn.Linear(68, 68, bias=False)

            # positional embedding for tactile regions
            self.pos_tac_embedd = nn.Sequential(
                            timeEmb(dim),
                            nn.Linear(dim, dim * 4),
                            nn.Mish(),
                            nn.Linear(dim * 4, dim)
                        )

            # learnable of each regions in forward pass
            # (b, 1062 + 1062) --> (b, 17 + 17)
            self.tactile_encoder = nn.Sequential(
                nn.Linear(1062*2, 512),
                nn.Mish(),
                nn.Linear(512, 34),
            )
        
        if self.use_task_condition:
            self.task_encoder = nn.Embedding(num_tasks, dim)

        self.time_embedd = nn.Sequential(
            timeEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim)
        )
        self.action_embedd = nn.Parameter(torch.zeros(1, chunk_size, dim))  # learnable action positional embedding
        self.action_encoder = nn.Sequential(
            nn.Linear(act_dim, dim),
            nn.Mish(),
            nn.Linear(dim, dim)
        )

        self.mmattn = nn.ModuleList([MMAttention(heads, dim, use_tactile, plugin, plugin_rank) for _ in range(num_attn_blocks)])  # joint attention blocks
        self.linear = nn.Linear(dim, act_dim) # final action prediction head

        if not plugin:
            self.initialize_weights()

        if img_pretrain:  # load weights after initialization 
            model_dict = torch.load(img_pretrain, map_location='cpu')
            pretrain_dict = {k: v for k, v in model_dict.items() if k in self.img_encoder.state_dict().keys() and np.shape(model_dict[k]) == np.shape(v)}
            self.img_encoder.load_state_dict(pretrain_dict, strict=True)
            if freeze_backbone:
                self.freeze()

    def forward(self, img1, img2, obs=None, act=None, task_idx=None, tac1=None, tac2=None, action_mask=None, training=True):
        """
        Args:
            img1: [B, C, H, W]
            img2: [B, C, H, W]
            obs: [B, 28]
            act: [B, chunk, 28]
            task_idx: [B, ]
            tac1: [B, 1062]
            tac2: [B, 1062]
            training: bool
        """
        # image encoding 
        feat, image_rotary_emb = self.img_encoding(img1, img2) # feat:[B, 2*img_seq_len, dim]
        if self.use_tactile:
            # calculate the mean tactile data of each regions(17 regions in inspire hand) in forward pass
            tac1_avg = torch.stack([tac1[:, s:e].mean(dim=1) for s, e in self.tactile_data_index.values()], dim=1)  # (batch, 17)
            tac2_avg = torch.stack([tac2[:, s:e].mean(dim=1) for s, e in self.tactile_data_index.values()], dim=1)  # (batch, 17)
            
            # learnable tactile encoder
            tactile_emb = self.tactile_encoder(torch.cat([tac1, tac2], dim=-1))  # (b, 1062 + 1062) --> (b, 17+17)
            # gated tactile
            tactile = torch.cat([tac1_avg, tac2_avg, tactile_emb], dim=-1)
            tactile = tactile * torch.sigmoid(self.gated(tactile))  # (b, 68)
            # positional embedding for tactile regions
            tactile = self.pos_tac_embedd(tactile)  # (b, 68) --> (b, 68, dim)
        else:
            tactile = None

        if self.obs_state:
            obs = self.obs_encoder(obs)  # (b, act_dim) --> (b, dim)
        if self.use_task_condition: # onehot condition for different sub-tasks
            task_emb = self.task_encoder(task_idx) # (b, ) --> (b, dim)

        if training:
            t = torch.sigmoid(torch.randn((act.shape[0],), device=act.device))  # t in [0, 1] (b, )
            act, noise = self.add_noise(act, t)
            t = self.time_embedd(t)  # time embedding, (b, ) --> (b, dim)
            if self.obs_state:
                t = t + obs
            if self.use_task_condition:
                t = t + task_emb
            feat, act = self.atten_forward(feat, act, image_rotary_emb=image_rotary_emb, t=t, tactile=tactile)
            return act, noise

        else:
            sample = torch.randn(img1.shape[0], self.chunk_size, self.act_dim).to(img1.device)  # initial noisy action
            t = get_schedule(self.inference_step, self.chunk_size)  # get denoising schedule
            for t_curr, t_prev in zip(t[:-1], t[1:]):  # denoising loop
                t_vec = torch.full((img1.shape[0],), t_curr, dtype=img1.dtype, device=img1.device)
                t_vec = self.time_embedd(t_vec)  # time embedding
                if self.obs_state:
                    t_vec = t_vec + obs
                if self.use_task_condition:
                    t_vec = t_vec + task_emb
                _, denoise_act = self.atten_forward(feat, sample, image_rotary_emb=image_rotary_emb, t=t_vec, tactile=tactile)
                # denoise_act : noise - action
                # t_prev - t_curr < 0 ---> -|△t|
                # action =  noise_act + |△t| * (action - noise)
                sample = sample + (t_prev - t_curr) * denoise_act

            return sample

    def img_encoding(self, img1, img2):
        assert img1.shape == img2.shape, "img1 and img2 must have the same shape"
        img = torch.cat([img1, img2], dim=0)  # concat img1 and img2 in batch dim
        feat = self.img_encoder(img)
        feat = self.img_head(feat)
        feat1, feat2 = feat.chunk(2, dim=0)

        # get rope freqs
        feat_h, feat_w = feat1.shape[-2:]
        image_rotary_emb = self.rope(feat_h, feat_w)

        # add img_id embedding
        feat1 = einops.rearrange(feat1, 'b c h w -> b (h w) c')
        feat2 = einops.rearrange(feat2, 'b c h w -> b (h w) c')
        img_id = torch.tensor([0]*feat1.shape[1] + [1]*feat2.shape[1]).to(img2.device)   # diff img_id for img1 and img2
        img_id = self.pos_idx_embedd(img_id).repeat(img1.shape[0], 1, 1)
        feat = torch.cat([feat1, feat2], dim=1)  # (b, 2*seq_len, dim)
        feat = feat + img_id

        return feat, image_rotary_emb


    def atten_forward(self, img, act, image_rotary_emb, t, tactile=None):
        act = self.action_encoder(act)   # (b, seq_len, act_dim) --> (b, seq_len, dim)
        act = act + self.action_embedd   # add learnable action positional embedding

        for mma in self.mmattn:
            img, act = mma(img, act, t, image_rotary_emb, tactile) 

        act = self.linear(act)
        return img, act
    
    def add_noise(self, act: torch.Tensor, t: torch.Tensor):
        noise = torch.randn_like(act).to(act.device)
        t = t.view(act.shape[0], 1, 1)
        act = (1 - t) * act + t * noise
        return act, noise
    
    def freeze(self):
        for param in self.img_encoder.parameters():
            param.requires_grad = False
    

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def init_tac_regions(self):
        _touch_regions = [
            ("fingerone_tip_touch", 3 * 3, (3, 3)),
            ("fingerone_top_touch", 12 * 8, (8, 12)),
            ("fingerone_palm_touch", 10 * 8, (8, 10)),
            ("fingertwo_tip_touch", 3 * 3, (3, 3)),
            ("fingertwo_top_touch", 12 * 8, (8, 12)),
            ("fingertwo_palm_touch", 10 * 8, (8, 10)),
            ("fingerthree_tip_touch", 3 * 3, (3, 3)),
            ("fingerthree_top_touch", 12 * 8, (8, 12)),
            ("fingerthree_palm_touch", 10 * 8, (8, 10)),
            ("fingerfour_tip_touch", 3 * 3, (3, 3)),
            ("fingerfour_top_touch", 12 * 8, (8, 12)),
            ("fingerfour_palm_touch", 10 * 8, (8, 10)),
            ("fingerfive_tip_touch", 3 * 3, (3, 3)),
            ("fingerfive_top_touch", 12 * 8, (8, 12)),
            ("fingerfive_middle_touch", 3 * 3, (3, 3)),
            ("fingerfive_palm_touch", 12 * 8, (8, 12)),
            ("palm_touch", 8 * 14, (8, 14)),
        ]
        _touch_start = 0
        self.tactile_data_index = {}

        for region_name, region_size, _ in _touch_regions:
            self.tactile_data_index[region_name] = [_touch_start, _touch_start + region_size]
            _touch_start += region_size
        assert _touch_start == 1062, "Total tactile data length should be 1062"


# low-rank adapter module
class PI_Adapter(nn.Module): 
    def __init__(self, dim, out_dim, rank=32):
        super().__init__()

        self.down = nn.Linear(dim, rank)
        self.up = nn.Linear(rank, out_dim)

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        x = self.down(x)
        x = self.up(x)
        return x


class MMAttention(nn.Module):
    def __init__(self, heads=8, dim=512, use_tactile=False, plugin=False, plugin_rank=32):
        super().__init__()
        head_dim = dim // heads
        self.head_dim = dim // heads
        self.head = heads
        self.use_tactile = use_tactile
        self.plugin = plugin

        ### img projection
        self.img_bais = adaLN(dim)  # adaLN for time conditioning

        # attention
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.img_qkv = nn.Linear(dim, dim * 3)
        self.img_qknorm = QKNorm(head_dim)
        self.img_proj = nn.Linear(dim, dim)
        # mlp
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(dim, dim*4, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim*4, dim, bias=True),
        )
        
        ### action projection
        self.act_bais = adaLN(dim)  # adaLN for time conditioning

        # attention
        self.act_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.act_qkv = nn.Linear(dim, dim * 3)
        self.act_qknorm = QKNorm(head_dim)
        self.act_proj = nn.Linear(dim, dim)
        # mlp
        self.act_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.act_mlp = nn.Sequential(
            nn.Linear(dim, dim*4, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim*4, dim, bias=True),
        )

        ### cross attention for tactile
        if self.use_tactile:
            self.tactile_key = nn.Linear(dim, dim)
            self.tactile_value = nn.Linear(dim, dim)
        
            if plugin:  # adapter modules for finetuning vision-based model
                self.img_qkv_pi = PI_Adapter(dim, dim*3, plugin_rank)
                self.img_proj_pi = PI_Adapter(dim, dim, plugin_rank)
                self.img_mlp_pi = PI_Adapter(dim, dim, plugin_rank)

                self.act_qkv_pi = PI_Adapter(dim, dim*3,plugin_rank)
                self.act_proj_pi = PI_Adapter(dim, dim, plugin_rank)
                self.act_mlp_pi = PI_Adapter(dim, dim, plugin_rank)


    def forward(self, img, act, t, image_rotary_emb, tactile=None):
        """
        Args:
            img: [B, 2*img_len, dim]
            act: [B, chunk, dim]
            t: [B, dim]
            image_rotary_emb: tuple[Tensor, Tensor]
            tactile: [B, 68, dim]
        """
        total_img_len = img.shape[1]
        scale1_feat, shift1_feat, gate1_feat, scale2_feat, shift2_feat, gate2_feat = self.img_bais(t)  # adaLN
        img_norm = self.img_norm1(img)  # layer norm
        img_norm = (1 + scale1_feat) * img_norm + shift1_feat
        img_qkv = self.img_qkv(img_norm)
        if self.use_tactile and self.plugin:
            img_qkv += self.img_qkv_pi(img_norm)

        img_q, img_k, img_v = einops.rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.head, D=self.head_dim)
        img_q, img_k = self.img_qknorm(img_q, img_k, img_v)  # qk norm
        # q,k,v: [B, H, l1, D]
        feat_len = int(total_img_len / 2)
        # apply rotary embedding twice for qk in diff img features
        img_q[:, :, :feat_len, :], img_k[:, :, :feat_len, :] = apply_rotary_emb(img_q[:, :, :feat_len, :], image_rotary_emb), apply_rotary_emb(img_k[:, :, :feat_len, :], image_rotary_emb)
        img_q[:, :, feat_len:, :], img_k[:, :, feat_len:, :] = apply_rotary_emb(img_q[:, :, feat_len:, :], image_rotary_emb), apply_rotary_emb(img_k[:, :, feat_len:, :], image_rotary_emb)
        

        scale1_act, shift1_act, gate1_act, scale2_act, shift2_act, gate2_act = self.act_bais(t) # adaLN
        act_norm = self.act_norm1(act)  # action is already added learnable pos emb
        act_norm = (1 + scale1_act) * act_norm + shift1_act
        act_qkv = self.act_qkv(act_norm)
        if self.use_tactile and self.plugin:
            act_qkv += self.act_qkv_pi(act_norm)

        act_q, act_k, act_v = einops.rearrange(act_qkv, "B L (K H D) -> K B H L D", K=3, H=self.head, D=self.head_dim)
        act_q, act_k = self.act_qknorm(act_q, act_k, act_v)  
        # q,k,v: [B, H, l2, D]

        q = torch.cat([img_q, act_q], dim=2)  # [B, H, l1+l2, D]
        k = torch.cat([img_k, act_k], dim=2)  # [B, H, l1+l2, D]
        v = torch.cat([img_v, act_v], dim=2)  # [B, H, l1+l2, D]
        attn = F.scaled_dot_product_attention(q, k, v)  # joint attention between img and act

        # cross atten for tactile
        if self.use_tactile and tactile is not None:
            tactile_k = self.tactile_key(tactile)  # [B, 68, dim]
            tactile_v = self.tactile_value(tactile)  # [B, 68, dim]
            tactile_k = einops.rearrange(tactile_k, "B L (H D) -> B H L D", H=self.head, D=self.head_dim)
            tactile_v = einops.rearrange(tactile_v, "B L (H D) -> B H L D", H=self.head, D=self.head_dim)

            cross_attn = F.scaled_dot_product_attention(q, tactile_k, tactile_v)  # joint attention with tactile
            attn = attn + cross_attn  # combine attention outputs
        
        attn = einops.rearrange(attn, "B H L D -> B L (H D)")
        img_attn, act_attn = attn[:, :total_img_len, :], attn[:, total_img_len:, :]  # split img and action

        img = img + gate1_feat * self.img_proj(img_attn) # residual after attn proj layer
        if self.use_tactile and self.plugin:
            img = img + gate1_feat * self.img_proj_pi(img_attn)

        img = img + gate2_feat * self.img_mlp((1 + scale2_feat) * self.img_norm2(img) + shift2_feat) # residual after mlp layer
        if self.use_tactile and self.plugin:
            img = img + gate2_feat * self.img_mlp_pi((1 + scale2_feat) * self.img_norm2(img) + shift2_feat)


        act = act + gate1_act * self.act_proj(act_attn)  # residual after attn proj layer
        if self.use_tactile and self.plugin:
            act = act + gate1_act * self.act_proj_pi(act_attn)

        act = act + gate2_act * self.act_mlp((1 + scale2_act) * self.act_norm2(act) + shift2_act)  # residual after mlp layer
        if self.use_tactile and self.plugin:
            act = act + gate2_act * self.act_mlp_pi((1 + scale2_act) * self.act_norm2(act) + shift2_act)

        return img, act


class adaLN(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim,  dim * 6)
        self.silu = nn.SiLU()

    def forward(self, vec: Tensor):
        out = self.linear(self.silu(vec))
        scale1, shift1, gate1, scale2, shift2, gate2 = out[:, None, :].chunk(6, dim=-1)
        return scale1, shift1, gate1, scale2, shift2, gate2

class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)

class timeEmb(nn.Module):
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
    

def modeling(action_dim, chunk_size, obs_state, use_tactile=False, plugin=False, plugin_rank=32, use_task_condition=False, num_tasks=10, inf_step=10, num_attn_blocks=6, heads=8, dim=512, rope_axes_dim=(256, 256), img_pretrain=None, freeze_backbone=True, pretrain_model_path=False, adapter_model_path=False):
    # def get_trainable_state_dict(model):
    #     return {
    #         name: param.detach().cpu()
    #         for name, param in model.named_parameters()
    #         if param.requires_grad
    #     }
    DeCO = DECO(
            act_dim=action_dim,
            chunk_size=chunk_size,
            obs_state=obs_state,
            use_tactile=use_tactile,
            plugin=plugin,
            plugin_rank=plugin_rank,
            use_task_condition=use_task_condition,
            num_tasks=num_tasks,
            inf_step=inf_step,
            num_attn_blocks=num_attn_blocks,
            heads=heads,
            dim=dim,
            rope_axes_dim=rope_axes_dim,
            img_pretrain=img_pretrain,
            freeze_backbone=freeze_backbone
        )
    
    if pretrain_model_path:  # load pretrained weights for finetuning or inference
        if use_tactile:  # vision-tactile model or vision model
            if plugin:  # adapter model or vision-tactile pretrained model
                if adapter_model_path:  # adapter inference or adapter finetuning
                    # load both base model and adapter model weights for inference, we save them together
                    print("loading adapter weights from {} for adapter inference".format(adapter_model_path))
                    model_dict = torch.load(adapter_model_path, map_location='cpu') 
                    DeCO.load_state_dict(model_dict, strict=True)
                else:
                    # finetune adapter, load base model weights and freeze them, only train adapter weights
                    print("loading vision pretrained weights from {} for adapter finetuning".format(pretrain_model_path))
                    model_dict = torch.load(pretrain_model_path, map_location='cpu')
                    model_state = DeCO.state_dict()
                    pretrain_dict = {k: v for k, v in model_dict.items() if k in model_state.keys() and v.shape == model_state[k].shape}
                    DeCO.load_state_dict(pretrain_dict, strict=False)
                    for name, param in DeCO.named_parameters():
                        if name in model_dict.keys():
                            param.requires_grad = False
                        else:
                            print('trainable params:', name)
            
            else:
                print("loading vision-tactile pretrained weights from {} for inference".format(pretrain_model_path))
                model_dict = torch.load(pretrain_model_path)
                DeCO.load_state_dict(model_dict, strict=True)

        else:
            # load vision model for inference
            print("loading vision pretrained weights from {} for inference".format(pretrain_model_path))
            model_dict = torch.load(pretrain_model_path)
            DeCO.load_state_dict(model_dict, strict=True)

        # save_dict = get_trainable_state_dict(DeCO)
        # torch.save(save_dict, './test.pth')

    return DeCO



if __name__ == '__main__':
    # model = torch.load('/root/yusun/ICML_codes/IL_training_codebase/models/mmrdt/1.pth')
    # total_params = 0
    # for k, v in model.items():
    #     print(k, v.shape)
    #     num = v.numel()
    #     total_params += num
    # print(f"\nTotal params: {total_params:,}")

    import yaml 
    with open('./deco.yaml', 'r') as f:
        config = yaml.safe_load(f)
    model = modeling(**config['model'])


    # from torchinfo import summary
    # # mmdit = DECO(act_dim=28, chunk_size=16, num_attn_blocks=3, inf_step=1, use_task_condition=True)
    # img1 = torch.randn(1, 3, 224, 224)
    # img2 = torch.randn(1, 3, 224, 224)
    # obs = torch.randn(1, 28)
    # act = torch.randn(1, 32, 28)
    # task_idx = torch.randint(0, 8, (1,))
    # tac1 = torch.randn(1, 1062)
    # tac2 = torch.randn(1, 1062)

    # act_train, _ = mmdit(img1, img2, obs=obs, act=act, task_idx=task_idx, training=True)
    # act_pred = mmdit(img1, img2, obs=obs, task_idx=task_idx, training=False)
    # print(act_train.shape, act_pred.shape)
    
    # summary(model, input_data=(img1, img2, obs, act, task_idx, tac1, tac2), device='cpu')