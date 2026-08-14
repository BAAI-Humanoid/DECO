import torch
import torch.nn as nn
from typing import Tuple, Union


class RotaryPosEmbed(nn.Module):

    def __init__(self, dim: int, rope_axes_dim: Tuple[int, int], theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.rope_axes_dim = rope_axes_dim
        self.theta = theta

    def forward(self, height, width):
        dim_h, dim_w = self.dim // 2, self.dim // 2
        h_inv_freq = 1.0 / (self.theta ** (torch.arange(0, dim_h, 2, dtype=torch.float32)[: (dim_h // 2)].float() / dim_h))
        w_inv_freq = 1.0 / (self.theta ** (torch.arange(0, dim_w, 2, dtype=torch.float32)[: (dim_w // 2)].float() / dim_w))

        h_seq = torch.arange(self.rope_axes_dim[0])
        w_seq = torch.arange(self.rope_axes_dim[1])

        freqs_h = torch.outer(h_seq, h_inv_freq)
        freqs_w = torch.outer(w_seq, w_inv_freq)

        h_idx = torch.arange(height, device=freqs_h.device)
        w_idx = torch.arange(width, device=freqs_w.device)

        inner_h_idx = h_idx * self.rope_axes_dim[0] // height
        inner_w_idx = w_idx * self.rope_axes_dim[1] // width

        freqs_h = freqs_h[inner_h_idx]
        freqs_w = freqs_w[inner_w_idx]

        freqs_h = freqs_h.unsqueeze(1)
        freqs_w = freqs_w.unsqueeze(0)

        # Broadcast freqs_h and freqs_w to [height, width, dim//4]
        freqs_h = freqs_h.expand(height, width, -1)
        freqs_w = freqs_w.expand(height, width, -1)

        freqs = torch.cat([freqs_h, freqs_w], dim=-1)
        freqs = torch.cat([freqs, freqs], dim=-1)  # [height, width, dim]

        freqs = freqs.reshape(height * width, -1)

        return (freqs.cos(), freqs.sin())

def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor]:

    cos, sin = freqs_cis  # [S, D]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    cos, sin = cos.to(x.device), sin.to(x.device)
    x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)  # [B, H, S, D//2]
    x_rotated = torch.cat([-x_imag, x_real], dim=-1)

    out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

    return out


if __name__ == "__main__":
    # rope use example in cogview4
    rope = RotaryPosEmbed(dim=64, rope_axes_dim=(256, 256))  # dim equals to the attention_head_dim
    '''
    attention:
    # (b, dim, h, w) -----> (b, heads, h*w, head_dims)
    [1, 512, 16, 16] --> [1, 256, 512] --> [1, 256, 8, 64] --> [1, 8, 256, 64]
    '''

    image_rotary_emb  = rope(16, 16)  # （h,w)
    q = torch.randn(1, 8, 256, 64)  # [B, head, len, head_dim]
    k = torch.randn(1, 8, 256, 64)
    q_rot = apply_rotary_emb(q, image_rotary_emb)
    k_rot = apply_rotary_emb(k, image_rotary_emb)
    print(q_rot.shape, k_rot.shape)

    '''
    query[:, :, text_seq_length:, :] = apply_rotary_emb(
                query[:, :, text_seq_length:, :], image_rotary_emb, use_real_unbind_dim=-2
            )
    key[:, :, text_seq_length:, :] = apply_rotary_emb(
        key[:, :, text_seq_length:, :], image_rotary_emb, use_real_unbind_dim=-2
    )
    '''

    
