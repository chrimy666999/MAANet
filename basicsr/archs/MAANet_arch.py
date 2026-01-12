import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import ops
from basicsr.utils.registry import ARCH_REGISTRY
from einops import rearrange
import numpy as np
from typing import Optional
import math


## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class GDFN_Adapter(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=None, drop=None):
        super().__init__()

        dim = in_features
        if hidden_features is None:
            hidden_features = in_features * 2.0
        ffn_expansion_factor = hidden_features / in_features
        self.ffn = FeedForward(dim=dim, ffn_expansion_factor=ffn_expansion_factor, bias=False)

    def forward(self, x, H, W):
        x_4d = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        output_4d = self.ffn(x_4d)
        output_3d = rearrange(output_4d, 'b c h w -> b (h w) c')
        return output_3d


def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)
def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class DynamicOffsetSampler(nn.Module):
    def __init__(self, in_channels, scale=2, offset_strategy='direct', groups=4, dyscope=False):
        super(DynamicOffsetSampler,self).__init__()
        self.scale = scale
        self.offset_strategy = offset_strategy
        self.groups = groups

        assert self.offset_strategy in ['direct', 'pre_shuffled']
        if self.offset_strategy == 'pre_shuffled':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if self.offset_strategy == 'pre_shuffled':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1)
            constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, self.scale * H, self.scale * W)

    def forward_direct(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pre_shuffled(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.offset_strategy == 'pre_shuffled':
            return self.forward_pre_shuffled(x)
        return self.forward_direct(x)

class Dual_Dynamic_Fusion_Upsampler_Block(nn.Module):
    def __init__(self, in_channels, scale=2, groups=4, dyscope=False):
        super(Dual_Dynamic_Fusion_Upsampler_Block, self).__init__()
        self.direct_sampler = DynamicOffsetSampler(in_channels, scale=scale, offset_strategy='direct', groups=groups, dyscope=dyscope)
        self.shuffled_sampler = DynamicOffsetSampler(in_channels, scale=scale, offset_strategy='pre_shuffled', groups=groups, dyscope=dyscope)
        self.fusion = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

    def forward(self, x):
        direct_output = self.direct_sampler(x)
        shuffled_output = self.shuffled_sampler(x)
        combined = torch.cat([direct_output, shuffled_output], dim=1)
        fused_output = self.fusion(combined)
        return fused_output


def window_partition(x, window_size):
    B, C, H, W = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size

    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect") 

    Hp, Wp = x.shape[-2:]
    x = x.view(B, C, Hp // window_size, window_size, Wp // window_size, window_size)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size, window_size)
    return x, Hp, Wp, (pad_h, pad_w)

def window_unpartition(windows, Hp, Wp, window_size, pad):
    pad_h, pad_w = pad
    B_ = windows.shape[0] // ((Hp // window_size) * (Wp // window_size))
    C = windows.shape[1]
    x = windows.view(B_, Hp // window_size, Wp // window_size, C, window_size, window_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B_, C, Hp, Wp)
    if pad_h or pad_w:
        x = x[:, :, :Hp - pad_h, :Wp - pad_w]
    return x

class ChannelFirstLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        assert data_format in ["channels_last", "channels_first"]
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        # channels_first
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class SpectralGatingUnit(nn.Module):
    def __init__(self, dim, groups=1, fft_norm='ortho', bottleneck_ratio=0.5):
        super().__init__()
        self.groups = groups
        self.fft_norm = fft_norm
        mid = max(1, int(dim * bottleneck_ratio))

        self.proj_in = nn.Conv2d(dim * 2, mid * 2, 1, bias=False, groups=self.groups)
        self.act = nn.GELU()
        self.proj_out = nn.Conv2d(mid * 2, dim * 2, 1, bias=False, groups=self.groups)

        self.mag_gate = nn.Sequential(
            nn.Conv2d(dim, max(1, dim // 2), 1, bias=True),
            nn.GELU(),
            nn.Conv2d(max(1, dim // 2), dim, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.size()

        Xf = torch.fft.rfft2(x, norm=self.fft_norm)

        mag = torch.abs(Xf)
        
        mag_gate = self.mag_gate(mag)
        Xf = Xf * mag_gate  

        real = Xf.real
        imag = Xf.imag
        cat = torch.cat([real, imag], dim=1)

        y = self.proj_in(cat)
        y = self.act(y)
        y = self.proj_out(y)

        real2, imag2 = torch.chunk(y, 2, dim=1)
        Xf2 = torch.complex(real2, imag2)

        out = torch.fft.irfft2(Xf2, s=(H, W), norm=self.fft_norm)
        return out

class Spectral_Spatial_Modulation_Unit(nn.Module):
    def __init__(self, dim, num_heads=4, window_size=8, bottleneck_ratio=0.5):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size

        self.pre_norm = ChannelFirstLayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.spectral_unit = SpectralGatingUnit(dim, bottleneck_ratio=bottleneck_ratio)

        self.v_proj1 = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.v_dw = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False)
        self.v_act = nn.GELU()
        self.v_proj2 = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

        self._temp = nn.Parameter(torch.ones(num_heads))  
        self.softplus = nn.Softplus()

        self.pos_encoding = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.proj_out = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

        self.channel_scale = nn.Parameter(1e-6 * torch.ones(dim))

    def forward(self, x):
        """
        x: [B, C, H, W]
        """
        B, C, H, W = x.shape
        shortcut = x
        pos_enc = self.pos_encoding(x)
        x = self.pre_norm(x)
        f = self.spectral_unit(x)

        s = self.v_proj1(x)
        s = self.v_dw(s)
        s = self.v_act(s)
        s = self.v_proj2(s)

        w = self.window_size
        f_win, Hp, Wp, pad = window_partition(f, w)
        s_win, _, _, _ = window_partition(s, w)

        f_tok = rearrange(f_win, 'bn (head ch) h w -> bn head ch (h w)', head=self.num_heads)
        s_tok = rearrange(s_win, 'bn (head ch) h w -> bn head ch (h w)', head=self.num_heads)

        temp = self.softplus(self._temp).view(1, self.num_heads, 1, 1)
        attn = (f_tok * s_tok) * temp  

        attn = F.softmax(attn, dim=-1)

        attn_win = rearrange(attn, 'bn head ch (h w) -> bn (head ch) h w', head=self.num_heads, h=w, w=w)

        attn_out = window_unpartition(attn_win, Hp, Wp, w, pad)

        attn_out = attn_out + pos_enc
        attn_out = self.proj_out(attn_out)

        attn_out = self.channel_scale.view(1, -1, 1, 1) * attn_out

        out = attn_out + shortcut
        return out
    

class MultiAxisKernelCore(nn.Module):
    def __init__(
        self,
        pdim: int,
        proj_dim_in: Optional[int] = None,
        k: int = 13,            
        num_bases: int = 8,      
        sk_size: int = 3        
    ):
        super().__init__()
        self.pdim = pdim
        self.proj_dim_in = proj_dim_in if proj_dim_in is not None else pdim
        self.k = k
        self.num_bases = num_bases
        self.sk_size = sk_size

        hidden = max(16, self.proj_dim_in // 2)

        self.lk_bases = nn.Parameter(torch.randn(num_bases, 1, k, k) * 0.01)

        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.proj_dim_in, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, pdim * num_bases, 1)
        )

        self.small_kernel_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.proj_dim_in, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, pdim * self.sk_size * self.sk_size, 1)
        )
        nn.init.zeros_(self.small_kernel_gen[-1].weight)
        nn.init.zeros_(self.small_kernel_gen[-1].bias)

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.proj_dim_in, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, pdim * 2, 1)
        )

        self.fuse = nn.Conv2d(pdim, pdim, 1)

    def forward(self, x: torch.Tensor, lk_filter: torch.Tensor = None) -> torch.Tensor:
        B, C, H, W = x.shape
        assert C >= self.pdim, "Input channels must be >= pdim"

        x1, x2 = x[:, :self.pdim], x[:, self.pdim:]

        rw = self.router(x[:, :self.proj_dim_in]).view(B, self.pdim, self.num_bases)
        rw = torch.softmax(rw, dim=-1)

        composed_k = (rw[..., None, None, None] * self.lk_bases[None, None, ...]).sum(dim=2)

        x1_reshaped = rearrange(x1, 'b c h w -> 1 (b c) h w')
        composed_k_groups = rearrange(composed_k, 'b c o k1 k2 -> (b c) o k1 k2')
        out_lk = F.conv2d(x1_reshaped, composed_k_groups, padding=self.k // 2, groups=B * self.pdim)
        out_lk = rearrange(out_lk, '1 (b c) h w -> b c h w', b=B, c=self.pdim)

        dyn_k = self.small_kernel_gen(x[:, :self.proj_dim_in]).view(B, self.pdim, 1, self.sk_size, self.sk_size)
        dyn_k = rearrange(dyn_k, 'b c o k1 k2 -> (b c) o k1 k2')
        out_dyn = F.conv2d(x1_reshaped, dyn_k, padding=self.sk_size // 2, groups=B * self.pdim)
        out_dyn = rearrange(out_dyn, '1 (b c) h w -> b c h w', b=B, c=self.pdim)

        if lk_filter is not None:
            out_ext = F.conv2d(x1, lk_filter, padding=lk_filter.shape[-1] // 2)
        else:
            out_ext = 0.0  

        g = self.gate(x[:, :self.proj_dim_in]).view(B, 2, self.pdim, 1, 1)
        g = torch.softmax(g, dim=1)
        g_lk, g_dyn = g[:, 0], g[:, 1]

        fused = g_lk * (out_lk + (out_ext if isinstance(out_ext, torch.Tensor) else 0.0)) + g_dyn * out_dyn

        y1 = self.fuse(fused) + x1

        y = torch.cat([y1, x2], dim=1)
        return y

    def extra_repr(self):
        return f'pdim={self.pdim}, proj_dim_in={self.proj_dim_in}, k={self.k}, num_bases={self.num_bases}, sk_size={self.sk_size}'


class Dynamic_Kernel_Synthesis_Block(nn.Module):
    def __init__(self, dim, mlp_dim):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim) 

        self.attention = MultiAxisKernelCore(
            pdim=dim,
            proj_dim_in=dim,
            k=13,             
            num_bases=6,      
            sk_size=3         
        )
        
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = GDFN_Adapter(in_features=dim, hidden_features=mlp_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        residual1 = x
        
        x_permuted = x.permute(0, 2, 3, 1)
    
        x_normed = self.norm1(x_permuted)
        
        x_norm_4d = x_normed.permute(0, 3, 1, 2)
        
        out_attention = self.attention(x_norm_4d)
        
        x = residual1 + out_attention
        
        residual2 = x
        x_ffn_input = rearrange(residual2, 'b c h w -> b (h w) c')
        x_norm_ffn = self.norm2(x_ffn_input)
        ffn_out = self.ffn(x_norm_ffn, H, W)
        ffn_out = rearrange(ffn_out, 'b (h w) c -> b c h w', h=H, w=W)
        
        x = residual2 + ffn_out
        
        return x


def adaptive_tiling(x, step, tile_size):

    b, c, h, w = x.shape
    
    y_starts = torch.arange(0, h - tile_size, step, device=x.device, dtype=torch.long) 
    x_starts = torch.arange(0, w - tile_size, step, device=x.device, dtype=torch.long) 

    edge_y = torch.tensor([h - tile_size], device=x.device, dtype=torch.long) 
    edge_x = torch.tensor([w - tile_size], device=x.device, dtype=torch.long) 
    all_y_starts = torch.unique(torch.cat([y_starts, edge_y]))
    all_x_starts = torch.unique(torch.cat([x_starts, edge_x]))

    delta_y = torch.arange(tile_size, device=x.device, dtype=torch.long)
    delta_x = torch.arange(tile_size, device=x.device, dtype=torch.long)

    indices_y = all_y_starts.view(-1, 1, 1, 1) + delta_y.view(1, 1, tile_size, 1)
    indices_x = all_x_starts.view(1, -1, 1, 1) + delta_x.view(1, 1, 1, tile_size)
    
    b_idx = torch.arange(b, device=x.device).view(b, 1, 1, 1, 1, 1)
    c_idx = torch.arange(c, device=x.device).view(1, c, 1, 1, 1, 1)
    
    tiles = x[b_idx, c_idx, indices_y, indices_x]
    
    tiles = rearrange(tiles, 'b c nh nw ph pw -> b (nh nw) c ph pw')
    
    return tiles, all_y_starts.numel(), all_x_starts.numel()

def aggregate_tiles(tiles, x, step, tile_size):

    b, c, h, w = x.shape
    
    y_starts = torch.arange(0, h - tile_size, step, device=x.device, dtype=torch.long) 
    x_starts = torch.arange(0, w - tile_size, step, device=x.device, dtype=torch.long) 
    edge_y = torch.tensor([h - tile_size], device=x.device, dtype=torch.long) 
    edge_x = torch.tensor([w - tile_size], device=x.device, dtype=torch.long) 
    all_y_starts = torch.unique(torch.cat([y_starts, edge_y]))
    all_x_starts = torch.unique(torch.cat([x_starts, edge_x]))
    nh, nw = all_y_starts.numel(), all_x_starts.numel()

    delta_y = torch.arange(tile_size, device=x.device, dtype=torch.long) 
    delta_x = torch.arange(tile_size, device=x.device, dtype=torch.long) 

    indices_y = all_y_starts.view(-1, 1, 1, 1) + delta_y.view(1, 1, tile_size, 1) 
    indices_x = all_x_starts.view(1, -1, 1, 1) + delta_x.view(1, 1, 1, tile_size) 
    
    linear_indices = indices_y * w + indices_x
    
    output = torch.zeros_like(x).view(b, c, -1) 
    counts = torch.zeros_like(x).view(b, c, -1)
    
    
    tiles_flat = rearrange(tiles, 'b (nh nw) c ph pw -> b c (nh nw ph pw)', nh=nh, nw=nw)
    
    target_indices = rearrange(linear_indices, 'nh nw ph pw -> (nh nw ph pw)')
    target_indices = target_indices.expand(b, c, -1)

    output.scatter_add_(2, target_indices, tiles_flat)
    
    counts.scatter_add_(2, target_indices, torch.ones_like(tiles_flat))

    output = output.view(b, c, h, w)
    counts = counts.view(b, c, h, w)
    
    return output / counts.clamp(min=1)



class NormBefore(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class LocalAttentionModule(nn.Module):
    def __init__(self, dim, heads, qk_dim):
        super().__init__()

        self.heads = heads
        self.dim = dim
        self.qk_dim = qk_dim
        self.scale = qk_dim ** -0.5
        self.to_q = nn.Linear(dim, qk_dim, bias=False)
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.proj(out)
    

class Adaptive_Tiling_Transformer_Block(nn.Module):
    def __init__(self, dim, mlp_dim, heads=8, tile_size=8, qk_dim=None): 
        super().__init__()
        self.tile_size = tile_size
        
        if qk_dim is None:
            qk_dim = dim

        self.local_attn_block = NormBefore(dim, LocalAttentionModule(dim=dim, heads=heads, qk_dim=qk_dim))
        self.ffn_block = NormBefore(dim, GDFN_Adapter(in_features=dim, hidden_features=mlp_dim))

    def forward(self, x):
        feature_complexity = torch.std(x).item()
        if feature_complexity > 0.5:
            step = self.tile_size - 4
        else:
            step = self.tile_size - 2
        step = max(1, step)
        
        tiles, _, _ = adaptive_tiling(x, step, self.tile_size)
        b, n, c, ph, pw = tiles.shape
        tiles = rearrange(tiles, 'b n c h w -> (b n) (h w) c')

        tiles_after_attn = self.local_attn_block(tiles) + tiles
        
        tiles_after_attn = rearrange(tiles_after_attn, '(b n) (h w) c -> b n c h w', n=n, h=ph, w=pw)
        x_after_attn = aggregate_tiles(tiles_after_attn, x, step, self.tile_size) 
        
        x_ffn_input = rearrange(x_after_attn, 'b c h w -> b (h w) c')
        H, W = x.shape[2], x.shape[3]
        x_out = self.ffn_block(x_ffn_input, H=H, W=W) + x_ffn_input
        x_out = rearrange(x_out, 'b (h w) c -> b c h w', h=x.shape[2], w=x.shape[3])
        
        return x_out

        
class Multi_Axis_Processing_Block(nn.Module):
    def __init__(self, dim, mlp_dim, heads, window_size, tile_size, qk_dim):
        super().__init__()
        self.spectral_branch = Spectral_Spatial_Modulation_Unit(dim=dim, num_heads=heads, window_size=window_size)
        self.global_branch = Dynamic_Kernel_Synthesis_Block(dim=dim, mlp_dim=mlp_dim)
        self.fusion_conv = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False)
        self.local_attn = Adaptive_Tiling_Transformer_Block(dim=dim, mlp_dim=mlp_dim, heads=heads, tile_size=tile_size, qk_dim=qk_dim)
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x):
        residual = x
        
        out_spectral = self.spectral_branch(x)
        out_global = self.global_branch(x)
        
        concatenated = torch.cat([out_spectral, out_global], dim=1)
        fused = self.fusion_conv(concatenated)
        
        out_local = self.local_attn(fused)
        
        out_conv = self.conv(out_local)
        
        return residual + out_conv


@ARCH_REGISTRY.register()
class MAANet(nn.Module):
    def __init__(self, dim, n_blocks=8, mlp_dim=72, heads=6, window_size=16, tile_size=16, qk_dim=36, upscaling_factor=4):
        super().__init__()
        self.upscaling_factor = upscaling_factor
        
        self.first_conv = nn.Conv2d(3, dim, 3, 1, 1)

        self.feats = nn.Sequential(*[
            Multi_Axis_Processing_Block( 
                dim=dim,
                mlp_dim=mlp_dim,
                heads=heads,
                window_size=window_size, 
                tile_size=tile_size, 
                qk_dim=qk_dim 
            ) for _ in range(n_blocks)
        ])

        self.upsampling = nn.Sequential()
        if upscaling_factor == 4:
            self.upsampling.add_module('dysample_x2_1', Dual_Dynamic_Fusion_Upsampler_Block(in_channels=dim, scale=2))
            self.upsampling.add_module('dysample_x2_2', Dual_Dynamic_Fusion_Upsampler_Block(in_channels=dim, scale=2))
        elif upscaling_factor == 2 or upscaling_factor == 3:
            self.upsampling.add_module(f'dysample_x{upscaling_factor}', Dual_Dynamic_Fusion_Upsampler_Block(in_channels=dim, scale=upscaling_factor))
    
        self.last_conv = nn.Conv2d(dim, 3, 3, 1, 1)
        if upscaling_factor != 1:
            self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)


    def forward(self, x):
        if self.upscaling_factor != 1: 
            base = F.interpolate(x, scale_factor=self.upscaling_factor, mode='bilinear', align_corners=False)
        else: 
            base = x
        
        feat = self.first_conv(x)
        
        body_feat = self.feats(feat)
        feat = body_feat + feat
    
        if self.upscaling_factor != 1:
            out = self.lrelu(self.upsampling(feat))
        else:
            out = feat
        
        out = self.last_conv(out) + base
       
        return out


if __name__== '__main__':
    # Test Model Complexity #
    from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis    
    # x = torch.randn(1, 3, 640, 360)
    x = torch.randn(1, 3, 320, 180)
    
    model = MAANet(
        dim=36,
        n_blocks=8,
        mlp_dim=72,
        heads=6, 
        window_size=16,
        tile_size=16,
        qk_dim=36, 
        upscaling_factor=4
    )
    
    print(model)
    print(f'params: {sum(map(lambda x: x.numel(), model.parameters()))}')
    print(flop_count_table(FlopCountAnalysis(model, x), activations=ActivationCountAnalysis(model, x)))
    output = model(x)
    print(output.shape)
