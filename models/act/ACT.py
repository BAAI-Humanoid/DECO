import math
import torch
import einops
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops.misc import FrozenBatchNorm2d
from torchvision.models._utils import IntermediateLayerGetter
from itertools import chain
from collections.abc import Callable


class ACT(nn.Module):

    def __init__(self, 
            use_vae=True,  # use VAE encoder for action and obs state to latent z
            obs_state: bool = True, # observation state of robot
            use_task_condition: bool = False, # task condition
            num_tasks: int = 10, 
            chunk_size: int = 50,  # prediction horizon
            dim_model: int = 512, 
            ffn_dim: int = 3200, 
            n_heads: int = 8, 
            droprate: float = 0.1, 
            pre_norm: bool = False, 
            num_vae_encoder_layers: int = 4,
            num_encoder_layers: int = 4,
            num_decoder_layers: int = 1,
            action_dim: int = 28, 
            ):
        # BERT style VAE encoder with input tokens [cls, robot_state, *action_sequence].
        # The cls token forms parameters of the latent's distribution (like this [*means, *log_variances]).
        super().__init__()
        
        self.obs_state = obs_state
        self.use_task_condition = use_task_condition
        self.chunk_size = chunk_size
        self.dim_model = dim_model
        self.use_vae = use_vae
        
        if self.use_vae:
            num_input_token_encoder = 1 + chunk_size
            self.vae_encoder = ACTEncoder(dim_model, ffn_dim, n_heads, droprate, pre_norm, num_layers=num_vae_encoder_layers)
            self.vae_encoder_cls_embed = nn.Embedding(1, dim_model)
            # Projection layer for action (joint-space target) to hidden dimension.
            self.act_encoder = nn.Linear(action_dim, dim_model)
            # Projection layer from the VAE encoder's output to the latent distribution's parameter space.
            self.vae_encoder_latent_output_proj = nn.Linear(dim_model, 64)
        
        
            # Projection layer for joint-space configuration to hidden dimension.
            if obs_state:
                self.obs_encoder = nn.Linear(action_dim, dim_model)
                num_input_token_encoder += 1
        
            self.register_buffer(
                "vae_encoder_pos_enc",
                create_sinusoidal_pos_embedding(num_input_token_encoder, dim_model).unsqueeze(0),
            )

        # Backbone for image feature extraction.
        backbone_model = getattr(torchvision.models, "resnet18")(
            replace_stride_with_dilation=[False, False, False],
            weights="ResNet18_Weights.IMAGENET1K_V1",
            norm_layer=FrozenBatchNorm2d,
        )
        # Note: The assumption here is that we are using a ResNet model (and hence layer4 is the final
        # feature map).
        # Note: The forward method of this returns a dict: {"feature_map": output}.
        self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

        # Transformer (acts as VAE decoder when training with the variational objective).
        self.encoder = ACTEncoder(dim_model, ffn_dim, n_heads, droprate, pre_norm, num_layers=num_encoder_layers)
        self.decoder = ACTDecoder(dim_model, ffn_dim, n_heads, droprate, pre_norm, num_layers=num_decoder_layers)

        # Transformer encoder input projections. The tokens will be structured like
        # [latent, (robot_state), (env_state), (image_feature_map_pixels)].
        n_1d_tokens = 1  # for the latent
        if self.obs_state:
            self.transformer_obs_encoder = nn.Linear(action_dim, dim_model)
            n_1d_tokens += 1

        if self.use_task_condition:
            self.task_encoder = nn.Embedding(num_tasks, dim_model)
            n_1d_tokens += 1
            
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, dim_model)
        self.encoder_latent_input_proj = nn.Linear(32, dim_model)
        self.encoder_img_feat_input_proj = nn.Conv2d(
            backbone_model.fc.in_features, dim_model, kernel_size=1
        )
            
        # Fixed sinusoidal positional embedding for the input to the transformer's encoder.
        self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(dim_model // 2)
        # Transformer decoder.
        # Learnable positional embedding for the transformer's decoder (in the style of DETR object queries).
        self.decoder_pos_embed = nn.Embedding(chunk_size, dim_model)

        # Final action regression head on the output of the transformer's decoder.
        self.action_head = nn.Linear(dim_model, action_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        """Xavier-uniform initialization of the transformer parameters as in the original code."""
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, img1, img2, obs=None, act=None, task_idx=None, tac1=None, tac2=None, action_mask=None, training=True): 
        batch_size = img1.shape[0]

        if training and self.use_vae:
            cls_embed = einops.repeat(
                self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size
            )  # (B, 1, D)
            action_embed = self.act_encoder(act)  # (B, S, D)
            
            if self.obs_state:
                robot_state_embed = self.obs_encoder(obs)
                robot_state_embed = robot_state_embed.unsqueeze(1)  # (B, 1, D)
                vae_encoder_input = [cls_embed, robot_state_embed, action_embed]  # (B, S+2, D)
            else:
                vae_encoder_input = [cls_embed, action_embed]
                
            vae_encoder_input = torch.cat(vae_encoder_input, axis=1)

            # Prepare fixed positional embedding.
            # Note: detach() shouldn't be necessary but leaving it the same as the original code just in case.
            pos_embed = self.vae_encoder_pos_enc.clone().detach()  # (1, S+2, D)

            # Prepare key padding mask for the transformer encoder. We have 1 or 2 extra tokens at the start of the
            # sequence depending whether we use the input states or not (cls and robot state)
            # False means not a padding token.
            cls_joint_is_pad = torch.full(
                (batch_size, 2 if self.obs_state else 1),
                False,
                device=obs.device,
            )
            key_padding_mask = torch.cat([cls_joint_is_pad, action_mask], axis=1)  
            
            # Forward pass through VAE encoder to get the latent PDF parameters.
            cls_token_out = self.vae_encoder(
                vae_encoder_input,
                pos_embed=pos_embed,
                key_padding_mask=key_padding_mask,
            )[:, 0, :]  # select the class token, with shape (B, D)
            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, :32]
            # This is 2log(sigma). Done this way to match the original implementation.
            log_sigma_x2 = latent_pdf_params[:, 32:]

            # Sample the latent with the reparameterization trick.
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            # When not using the VAE encoder, we set the latent to be all zeros.
            mu  = None
            log_sigma_x2 = None
            latent_sample = torch.zeros([batch_size, 32], dtype=torch.float32).to(obs.device) 

        if self.use_task_condition:
            condition = self.task_encoder(task_idx)
            condition = condition.unsqueeze(1) # (B, 1, D)
        # Prepare transformer encoder inputs.
        encoder_in_tokens = self.encoder_latent_input_proj(latent_sample).unsqueeze(1)  # (B, 1, D)
        encoder_in_pos_embed = self.encoder_1d_feature_pos_embed.weight.repeat(batch_size, 1, 1)  # (B, 1, D)
        # Robot state token.
        if self.obs_state:
            obs_state_in_tokens = self.transformer_obs_encoder(obs).unsqueeze(1)  # (b, 1, D)

        feat1 = self.backbone(img1)["feature_map"]  # (b, 512, h, w)
        feat2 = self.backbone(img2)["feature_map"]
        feat1_pos_embed = self.encoder_cam_feat_pos_embed(feat1).to(dtype=feat1.dtype)
        feat2_pos_embed = self.encoder_cam_feat_pos_embed(feat2).to(dtype=feat2.dtype)

        
        feat1 = self.encoder_img_feat_input_proj(feat1)
        feat2 = self.encoder_img_feat_input_proj(feat2)


        # Rearrange features to (sequence, batch, dim).
        feat1 = einops.rearrange(feat1, "b c h w -> b (h w) c")  #  (b, seq_len, D)
        feat2 = einops.rearrange(feat2, "b c h w -> b (h w) c")  #  (b, seq_len, D)
        feat_input = torch.cat([feat1, feat2], axis=1) # (b, 2*seq_len, D)

        feat1_pos_embed = einops.rearrange(feat1_pos_embed, "b c h w -> b (h w) c")
        feat1_pos_embed = feat1_pos_embed.repeat(batch_size, 1, 1)
        feat2_pos_embed = einops.rearrange(feat2_pos_embed, "b c h w -> b (h w) c")
        feat2_pos_embed = feat2_pos_embed.repeat(batch_size, 1, 1)

        pos_embed_input = torch.cat([feat1_pos_embed, feat2_pos_embed], axis=1)
        # Stack all tokens along the sequence dimension.
        encoder_in_tokens = torch.cat([encoder_in_tokens, feat_input], axis=1)

        if self.obs_state:
            encoder_in_tokens = torch.cat([obs_state_in_tokens, encoder_in_tokens], axis=1)

        if self.use_task_condition:
            encoder_in_tokens = torch.cat([condition, encoder_in_tokens], axis=1)
        
        encoder_in_pos_embed = torch.cat([encoder_in_pos_embed, pos_embed_input], axis=1)
        # Forward pass through the transformer modules.
        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)
        
        # TODO(rcadene, alexander-soare): remove call to `device` ; precompute and use buffer
        decoder_in = torch.zeros(
            (batch_size, self.chunk_size, self.dim_model),
            dtype=encoder_in_pos_embed.dtype,
            device=encoder_in_pos_embed.device,
        )
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(0),
        )
        # Move back to (B, S, C).
        actions = self.action_head(decoder_out)

        if training:
            return actions, (mu, log_sigma_x2)
        else:
            return actions


class ACTEncoder(nn.Module):
    """Convenience module for running multiple encoder layers, maybe followed by normalization."""

    def __init__(self, dim_model: int = 512, ffn_dim: int = 3200, n_heads: int = 8, droprate: float = 0.1, pre_norm: bool = False, num_layers: int = 4):
        super().__init__()
        num_layers = num_layers
        self.layers = nn.ModuleList([ACTEncoderLayer(dim_model, ffn_dim, n_heads, droprate, pre_norm) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(512)

    def forward(self, x: Tensor, pos_embed: Tensor=None, key_padding_mask: Tensor=None) -> Tensor:
        
        for layer in self.layers:
            x = layer(x, pos_embed=pos_embed, key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return x


class ACTEncoderLayer(nn.Module):
    def __init__(self, dim_model: int = 512, ffn_dim: int = 3200, n_heads: int = 8, droprate: float = 0.1, pre_norm: bool = False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim_model, 8, dropout=0.1, batch_first=True)

        # Feed forward layers.
        self.linear1 = nn.Linear(dim_model, ffn_dim)
        self.dropout = nn.Dropout(droprate)
        self.linear2 = nn.Linear(ffn_dim, dim_model)

        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.dropout1 = nn.Dropout(droprate)
        self.dropout2 = nn.Dropout(droprate)

        self.activation = get_activation_fn('relu')
        self.pre_norm = pre_norm

    def forward(self, x, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None) -> Tensor:
        skip = x
        
        if self.pre_norm:
            x = self.norm1(x)
        
        q = k = x if pos_embed is None else x + pos_embed
        x = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)
        x = x[0]  # note: [0] to select just the output, not the attention weights
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout2(x)
        if not self.pre_norm:
            x = self.norm2(x)
        return x


class ACTDecoder(nn.Module):
    def __init__(self, dim_model: int = 512, ffn_dim: int = 3200, n_heads: int = 8, droprate: float = 0.1, pre_norm: bool = False, num_layers: int = 1):
        """Convenience module for running multiple decoder layers followed by normalization."""
        super().__init__()
        self.layers = nn.ModuleList([ACTDecoderLayer(dim_model, ffn_dim, n_heads, droprate, pre_norm) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(dim_model)

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(
                x, encoder_out, decoder_pos_embed=decoder_pos_embed, encoder_pos_embed=encoder_pos_embed
            )
        if self.norm is not None:
            x = self.norm(x)
        return x


class ACTDecoderLayer(nn.Module):
    def __init__(self, dim_model: int = 512, ffn_dim: int = 3200, n_heads: int = 8, droprate: float = 0.1, pre_norm: bool = False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim_model, n_heads, dropout=droprate, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(dim_model, n_heads, dropout=droprate, batch_first=True)

        # Feed forward layers.
        self.linear1 = nn.Linear(dim_model, ffn_dim)
        self.dropout = nn.Dropout(droprate)
        self.linear2 = nn.Linear(ffn_dim,dim_model)

        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.norm3 = nn.LayerNorm(dim_model)
        self.dropout1 = nn.Dropout(droprate)
        self.dropout2 = nn.Dropout(droprate)
        self.dropout3 = nn.Dropout(droprate)

        self.activation = get_activation_fn("relu")
        self.pre_norm = pre_norm
        
    def maybe_add_pos_embed(self, tensor: Tensor, pos_embed: Tensor | None) -> Tensor:
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x: (Decoder Sequence, Batch, Channel) tensor of input tokens.
            encoder_out: (Encoder Sequence, B, C) output features from the last layer of the encoder we are
                cross-attending with.
            encoder_pos_embed: (ES, 1, C) positional embedding for keys (from the encoder).
            decoder_pos_embed: (DS, 1, C) positional embedding for the queries (from the decoder).
        Returns:
            (DS, B, C) tensor of decoder output features.
        """
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)
        x = self.self_attn(q, k, value=x)[0]  # select just the output, not the attention weights
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.multihead_attn(
            query=self.maybe_add_pos_embed(x, decoder_pos_embed),
            key=self.maybe_add_pos_embed(encoder_out, encoder_pos_embed),
            value=encoder_out,
        )[0]  # select just the output, not the attention weights
        x = skip + self.dropout2(x)
        if self.pre_norm:
            skip = x
            x = self.norm3(x)
        else:
            x = self.norm2(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout3(x)
        if not self.pre_norm:
            x = self.norm3(x)
        return x


def create_sinusoidal_pos_embedding(num_positions: int, dimension: int) -> Tensor:
    """1D sinusoidal positional embeddings as in Attention is All You Need.

    Args:
        num_positions: Number of token positions required.
    Returns: (num_positions, dimension) position embeddings (the first dimension is the batch dimension).

    """

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / dimension) for hid_j in range(dimension)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(num_positions)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.from_numpy(sinusoid_table).float()


class ACTSinusoidalPositionEmbedding2d(nn.Module):
    """2D sinusoidal positional embeddings similar to what's presented in Attention Is All You Need.

    The variation is that the position indices are normalized in [0, 2π] (not quite: the lower bound is 1/H
    for the vertical direction, and 1/W for the horizontal direction.
    """

    def __init__(self, dimension: int):
        """
        Args:
            dimension: The desired dimension of the embeddings.
        """
        super().__init__()
        self.dimension = dimension
        self._two_pi = 2 * math.pi
        self._eps = 1e-6
        # Inverse "common ratio" for the geometric progression in sinusoid frequencies.
        self._temperature = 10000

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: A (B, C, H, W) batch of 2D feature map to generate the embeddings for.
        Returns:
            A (1, C, H, W) batch of corresponding sinusoidal positional embeddings.
        """
        not_mask = torch.ones_like(x[0, :1])  # (1, H, W)
        # Note: These are like range(1, H+1) and range(1, W+1) respectively, but in most implementations
        # they would be range(0, H) and range(0, W). Keeping it at as is to match the original code.
        y_range = not_mask.cumsum(1, dtype=torch.float32)
        x_range = not_mask.cumsum(2, dtype=torch.float32)

        # "Normalize" the position index such that it ranges in [0, 2π].
        # Note: Adding epsilon on the denominator should not be needed as all values of y_embed and x_range
        # are non-zero by construction. This is an artifact of the original code.
        y_range = y_range / (y_range[:, -1:, :] + self._eps) * self._two_pi
        x_range = x_range / (x_range[:, :, -1:] + self._eps) * self._two_pi

        inverse_frequency = self._temperature ** (
            2 * (torch.arange(self.dimension, dtype=torch.float32, device=x.device) // 2) / self.dimension
        )

        x_range = x_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)
        y_range = y_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)

        # Note: this stack then flatten operation results in interleaved sine and cosine terms.
        # pos_embed_x and pos_embed_y are (1, H, W, C // 2).
        pos_embed_x = torch.stack((x_range[..., 0::2].sin(), x_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed_y = torch.stack((y_range[..., 0::2].sin(), y_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed = torch.cat((pos_embed_y, pos_embed_x), dim=3).permute(0, 3, 1, 2)  # (1, C, H, W)

        return pos_embed


def get_activation_fn(activation: str) -> Callable:
    """Return an activation function given a string."""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")

def modeling(action_dim, chunk_size, obs_state, use_task_condition=False, num_tasks=0, use_vae=True, num_vae_encoder_layers=4, num_encoder_layers=4, num_decoder_layers=1, pretrain_model_path=False):
    act = ACT(
            chunk_size=chunk_size, 
            obs_state=obs_state, 
            use_task_condition=use_task_condition, 
            num_tasks=num_tasks,
            use_vae=use_vae, 
            num_vae_encoder_layers=num_vae_encoder_layers, 
            num_encoder_layers=num_encoder_layers, 
            num_decoder_layers=num_decoder_layers, 
            action_dim=action_dim
        )
    if pretrain_model_path:
        print("loading pretrained weights from {}".format(pretrain_model_path))
        model_dict = torch.load(pretrain_model_path, map_location='cpu')
        act.load_state_dict(model_dict, strict=True)

    return act


if __name__ == '__main__':
    img1 = torch.randn(2, 3, 224, 224)
    img2 = torch.randn(2, 3, 224, 224)
    task_idx = torch.randint(0, 10, (2,))
    chunk_size = 10
    action_dim = 28

    obs = torch.randn(2, action_dim)
    action = torch.randn(2, chunk_size, action_dim)
    mask = torch.zeros(2, chunk_size)
    
    model = ACT(action_dim=action_dim, chunk_size=chunk_size, obs_state=True, use_task_condition=True, num_tasks=10, use_vae=True)
    # training
    out, (_, _) = model(img1, img2, obs=obs, act=action, task_idx=task_idx, action_mask=mask)
    print(out.shape)

    # inference
    with torch.no_grad():
        out = model(img1, img2, obs=obs, task_idx=task_idx, training=False)
    print(out.shape)
