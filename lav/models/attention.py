import math
import torch
from torch import nn
from einops import rearrange, repeat

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()

        assert dim % num_heads == 0

        dim_head = dim // num_heads
        self.q = nn.Parameter(torch.randn(1, num_heads, 1, dim_head))

        self.linear_kv = nn.Linear(dim, dim * 2)

        self.num_heads = num_heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

    def forward(self, x):

        _, d, h, w = x.size()

        x = rearrange(x, 'b d h w -> b (h w) d')

        kv = self.linear_kv(x).chunk(2, dim=-1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.num_heads), kv)
        k = k + positionalencoding1d(d//self.num_heads, h*w).to(k.device)
        q = repeat(self.q, '() n l d -> b n l d', b=x.size(0))

        dots = torch.matmul(q, k.transpose(-1,-2)) * self.scale
        attn = torch.softmax(dots, dim=-1)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)').squeeze(1)

        return out

def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe
