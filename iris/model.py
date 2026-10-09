"""Iris-3B: dual-level pixel-space diffusion transformer, ComfyUI-native port.

Reference implementation: speridlabs/iris-3b, ``src/iris3b/models/dit.py`` (Apache-2.0).
Parameter names match the released ``model.safetensors`` 1:1, so the checkpoint loads with
no key conversion. Everything runs through ``comfy.ops`` linears and ``optimized_attention``,
so weight casting, offloading, fp8 / int8-convrot quantization and attention overrides all
behave as they do for built-in models.

Level 1 (patch stage): text-conditioned transformer over 16x16 RGB patches -- 8 dual-stream
MM-DiT blocks then 16 single-stream blocks, GQA, sigmoid-gated attention, sandwich RMSNorm and
shared-core + per-block-bias adaLN.
Level 2 (pixel stage): 4 PiT blocks that decode every patch back to pixels with per-pixel
modulation, attending across patches through a compacted per-patch token.

Input/output are RGB in [-1, 1] / flow velocity in pixel space; there is no VAE.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.ldm.common_dit
import comfy.model_management
import comfy.patcher_extension
from comfy.ldm.flux.math import apply_rope, rope
from comfy.ldm.modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention
from comfy.ldm.modules.diffusionmodules.mmdit import TimestepEmbedder, get_1d_sincos_pos_embed_from_grid_torch

# the Qwen3-VL assistant-turn suffix "<|im_end|>\n<|im_start|>assistant\n" is 5 tokens; the
# reference truncates the caption, never the suffix, so an over-long prompt keeps its tail
TEXT_SUFFIX_TOKENS = 5


def modulate(x, shift, scale):
    return torch.addcmul(shift, x, 1 + scale)


def attention(q, k, v, heads, comfy_attention, mask=None, transformer_options={}):
    """q: [B, H, N, D], k/v: [B, H_kv, N, D] -> [B, N, H*D]. GQA when H_kv < H."""
    return optimized_attention(
        AttentionTensorContainer(q), AttentionTensorContainer(k), AttentionTensorContainer(v), heads,
        mask=mask, skip_reshape=True, enable_gqa=k.shape[1] != heads,
        preferred_attention=comfy_attention, transformer_options=transformer_options,
    )


def rope_2d(head_dim, height, width, theta, scale, device):
    """Isotropic axial 2D RoPE (reference ``rope_aspect="isotropic"``).

    One step shared by both axes so the long side spans [0, scale] and aspect ratio survives.
    Flux-format rotations [H*W, head_dim/2, 2, 2], (x, y) frequency pairs interleaved.
    """
    step = scale / max(max(height, width) - 1, 1)
    ys = torch.arange(height, device=device, dtype=torch.float32) * step
    xs = torch.arange(width, device=device, dtype=torch.float32) * step
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    rope_x = rope(grid_x.reshape(1, -1), head_dim // 2, theta).squeeze(0)
    rope_y = rope(grid_y.reshape(1, -1), head_dim // 2, theta).squeeze(0)
    return torch.stack([rope_x, rope_y], dim=2).reshape(height * width, head_dim // 2, 2, 2)


def rope_1d(head_dim, length, theta, device):
    return rope(torch.arange(length, device=device, dtype=torch.float32).reshape(1, -1), head_dim, theta).squeeze(0)


def pad_text(context, attention_mask, text_len):
    """Bring text states to the fixed ``text_len`` the model was trained with.

    The reference always runs ``text_len`` text tokens: pad rows are zero states that still
    enter the trunk's (unmasked) joint attention, and only the text adapter masks them. So pad
    here with zeros + a 0/1 mask instead of letting conds be repeated to a common length.
    Returns ([B, text_len, D], [B, text_len] float mask).
    """
    batch, length = context.shape[:2]
    if attention_mask is None:
        mask = torch.ones((batch, length), device=context.device, dtype=context.dtype)
    else:
        mask = (attention_mask[:, :length] > 0.5).to(context.dtype)
        context = context * mask.unsqueeze(-1)
    if length > text_len:
        logging.warning(f"Iris: prompt is {length} tokens, the model reads {text_len}; truncating the caption.")
        if attention_mask is None:
            keep = text_len - TEXT_SUFFIX_TOKENS
            context = torch.cat([context[:, :keep], context[:, -TEXT_SUFFIX_TOKENS:]], dim=1)
            mask = mask[:, :text_len]
        else:
            context, mask = context[:, :text_len], mask[:, :text_len]
    elif length < text_len:
        context = F.pad(context, (0, 0, 0, text_len - length))
        mask = F.pad(mask, (0, text_len - length))
    return context, mask


class SwiGLU(nn.Module):
    """Bias-free gated SiLU MLP, 2/3 width rule: hidden = int(2 * int(dim * ratio) / 3).

    ``hidden`` overrides the width: a checkpoint may carry the hidden axis zero-padded (exact,
    since silu(0) * 0 = 0) so every projection is int8/convrot friendly.
    """
    def __init__(self, dim, mlp_ratio=4.0, hidden=None, dtype=None, device=None, operations=None):
        super().__init__()
        hidden = hidden or int(2 * int(dim * mlp_ratio) / 3)
        self.w1 = operations.Linear(dim, hidden, bias=False, dtype=dtype, device=device)
        self.w3 = operations.Linear(dim, hidden, bias=False, dtype=dtype, device=device)
        self.w2 = operations.Linear(hidden, dim, bias=False, dtype=dtype, device=device)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class GeluMLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, dtype=None, device=None, operations=None):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = operations.Linear(dim, hidden, bias=True, dtype=dtype, device=device)
        self.fc2 = operations.Linear(hidden, dim, bias=True, dtype=dtype, device=device)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class SelfAttention(nn.Module):
    """Fused-QKV multi-head attention with optional per-head QK RMSNorm and RoPE."""
    def __init__(self, dim, num_heads, qk_norm=True, dtype=None, device=None, operations=None):
        super().__init__()
        self.comfy_attention = ComfyAttention()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = operations.Linear(dim, dim * 3, bias=False, dtype=dtype, device=device)
        if qk_norm:
            self.q_norm = operations.RMSNorm(self.head_dim, eps=1e-6, dtype=dtype, device=device)
            self.k_norm = operations.RMSNorm(self.head_dim, eps=1e-6, dtype=dtype, device=device)
        else:
            self.q_norm = self.k_norm = nn.Identity()
        self.proj = operations.Linear(dim, dim, bias=True, dtype=dtype, device=device)

    def forward(self, x, pe=None, mask=None, transformer_options={}):
        batch, n, _ = x.shape
        q, k, v = self.qkv(x).view(batch, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if pe is not None:
            q, k = apply_rope(q, k, pe[None, None])
        return self.proj(attention(q, k, v, self.num_heads, self.comfy_attention, mask, transformer_options))


# -- text adapter ("lap_blocks2"): per-token attention across 12 encoder layers, then a token refiner

class LayerwiseAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, dtype=None, device=None, operations=None):
        super().__init__()
        self.norm1 = operations.RMSNorm(dim, eps=1e-6, dtype=dtype, device=device)
        self.attn = SelfAttention(dim, num_heads, qk_norm=False, dtype=dtype, device=device, operations=operations)
        self.norm2 = operations.RMSNorm(dim, eps=1e-6, dtype=dtype, device=device)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            operations.Linear(dim, hidden, bias=True, dtype=dtype, device=device),
            nn.SiLU(),
            operations.Linear(hidden, dim, bias=True, dtype=dtype, device=device),
        )

    def forward(self, x, transformer_options={}):
        x = x + self.attn(self.norm1(x), transformer_options=transformer_options)
        return x + self.mlp(self.norm2(x))


class TextAdapterBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, dtype=None, device=None, operations=None):
        super().__init__()
        self.norm1 = operations.RMSNorm(dim, eps=1e-6, dtype=dtype, device=device)
        self.attn = SelfAttention(dim, num_heads, qk_norm=True, dtype=dtype, device=device, operations=operations)
        self.norm2 = operations.RMSNorm(dim, eps=1e-6, dtype=dtype, device=device)
        self.mlp = SwiGLU(dim, mlp_ratio, dtype=dtype, device=device, operations=operations)

    def forward(self, y, mask, transformer_options={}):
        y = y + self.attn(self.norm1(y), mask=mask, transformer_options=transformer_options)
        return y + self.mlp(self.norm2(y))


class TextRefiner(nn.Module):
    def __init__(self, text_dim, hidden_size, num_heads, mlp_ratio, dtype=None, device=None, operations=None):
        super().__init__()
        self.proj = operations.Linear(text_dim, hidden_size, bias=True, dtype=dtype, device=device)
        self.blocks = nn.ModuleList([
            TextAdapterBlock(hidden_size, num_heads, mlp_ratio, dtype=dtype, device=device, operations=operations)
            for _ in range(2)
        ])
        self.norm = operations.RMSNorm(hidden_size, eps=1e-6, dtype=dtype, device=device)

    def forward(self, y, mask, transformer_options={}):
        # pad keys are masked; every query keeps its own position so a pad row never goes all -inf
        length = y.shape[1]
        keep = (mask > 0.5)[:, None, None, :] | torch.eye(length, dtype=torch.bool, device=y.device)
        attn_mask = torch.zeros(keep.shape, dtype=y.dtype, device=y.device).masked_fill_(~keep, float("-inf"))
        y = self.proj(y)
        for block in self.blocks:
            y = block(y, attn_mask, transformer_options=transformer_options)
        return self.norm(y)


class LayerwiseTextEmbedder(nn.Module):
    def __init__(self, text_dim, hidden_size, num_layers, layer_num_heads, layer_mlp_ratio,
                 refiner_num_heads, refiner_mlp_ratio, dtype=None, device=None, operations=None):
        super().__init__()
        self.text_dim = text_dim
        self.num_layers = num_layers
        self.layer_blocks = nn.ModuleList([
            LayerwiseAttentionBlock(text_dim, layer_num_heads, layer_mlp_ratio, dtype=dtype, device=device, operations=operations)
            for _ in range(2)
        ])
        self.layer_pool = operations.Linear(num_layers, 1, bias=True, dtype=dtype, device=device)
        self.refiner = TextRefiner(text_dim, hidden_size, refiner_num_heads, refiner_mlp_ratio,
                                   dtype=dtype, device=device, operations=operations)

    def forward(self, y, mask, transformer_options={}):
        """y: [B, T, num_layers * text_dim] (layers flattened per token, as the TE emits them)."""
        batch, length, _ = y.shape
        y = y.reshape(batch * length, self.num_layers, self.text_dim)
        for block in self.layer_blocks:
            y = block(y, transformer_options=transformer_options)
        y = self.layer_pool(y.transpose(1, 2)).squeeze(-1)
        return self.refiner(y.reshape(batch, length, self.text_dim), mask, transformer_options=transformer_options)


# -- patch stage

class SharedBiasModulation(nn.Module):
    """adaLN-single: the model owns one modulation core per stream; each block only adds a bias."""
    def __init__(self, dim, dtype=None, device=None):
        super().__init__()
        self.bias = nn.Parameter(torch.empty(6 * dim, dtype=dtype, device=device))

    def forward(self, core):
        return core + comfy.model_management.cast_to(self.bias, dtype=core.dtype, device=core.device)


class JointAttention(nn.Module):
    """MM-DiT joint attention: per-stream GQA projections + QK norms, one softmax, text first."""
    def __init__(self, dim, num_heads, num_kv_heads, dtype=None, device=None, operations=None):
        super().__init__()
        self.comfy_attention = ComfyAttention()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        for s in ("x", "y"):
            setattr(self, f"q_proj_{s}", operations.Linear(dim, dim, bias=False, dtype=dtype, device=device))
            setattr(self, f"k_proj_{s}", operations.Linear(dim, kv_dim, bias=False, dtype=dtype, device=device))
            setattr(self, f"v_proj_{s}", operations.Linear(dim, kv_dim, bias=False, dtype=dtype, device=device))
            setattr(self, f"q_norm_{s}", operations.RMSNorm(self.head_dim, eps=1e-6, dtype=dtype, device=device))
            setattr(self, f"k_norm_{s}", operations.RMSNorm(self.head_dim, eps=1e-6, dtype=dtype, device=device))
            setattr(self, f"proj_{s}", operations.Linear(dim, dim, bias=True, dtype=dtype, device=device))

    def _qkv(self, h, s, pe):
        batch, n, _ = h.shape
        q = getattr(self, f"q_proj_{s}")(h).view(batch, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = getattr(self, f"k_proj_{s}")(h).view(batch, n, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = getattr(self, f"v_proj_{s}")(h).view(batch, n, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(getattr(self, f"q_norm_{s}")(q), getattr(self, f"k_norm_{s}")(k), pe[None, None])
        return q, k, v

    def forward(self, hx, hy, pe_img, pe_txt, gate_x, gate_y, transformer_options={}):
        n_txt = hy.shape[1]
        qx, kx, vx = self._qkv(hx, "x", pe_img)
        qy, ky, vy = self._qkv(hy, "y", pe_txt)
        out = attention(torch.cat([qy, qx], dim=2), torch.cat([ky, kx], dim=2), torch.cat([vy, vx], dim=2),
                        self.num_heads, self.comfy_attention, transformer_options=transformer_options)
        return self.proj_x(out[:, n_txt:] * gate_x), self.proj_y(out[:, :n_txt] * gate_y)


class DualStreamBlock(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_ratio, mlp_hidden=None, dtype=None, device=None, operations=None):
        super().__init__()
        kw = dict(dtype=dtype, device=device)
        for s in ("x", "y"):
            setattr(self, f"norm_{s}1", operations.RMSNorm(dim, eps=1e-6, **kw))
            setattr(self, f"norm_{s}2", operations.RMSNorm(dim, eps=1e-6, **kw))
            setattr(self, f"attn_gate_{s}", operations.Linear(dim, dim, bias=False, **kw))
            setattr(self, f"attn_post_norm_{s}", operations.RMSNorm(dim, eps=1e-6, **kw))
            setattr(self, f"mlp_{s}", SwiGLU(dim, mlp_ratio, mlp_hidden, operations=operations, **kw))
            setattr(self, f"mlp_post_norm_{s}", operations.RMSNorm(dim, eps=1e-6, **kw))
        self.attn = JointAttention(dim, num_heads, num_kv_heads, operations=operations, **kw)
        self.adaln_img = SharedBiasModulation(dim, **kw)
        self.adaln_txt = SharedBiasModulation(dim, **kw)

    def forward(self, x, y, core_img, core_txt, pe_img, pe_txt, last=False, transformer_options={}):
        xs1, xc1, xg1, xs2, xc2, xg2 = self.adaln_img(core_img).chunk(6, dim=-1)
        ys1, yc1, yg1, ys2, yc2, yg2 = self.adaln_txt(core_txt).chunk(6, dim=-1)
        hx = modulate(self.norm_x1(x), xs1, xc1)
        hy = modulate(self.norm_y1(y), ys1, yc1)
        attn_x, attn_y = self.attn(hx, hy, pe_img, pe_txt, torch.sigmoid(self.attn_gate_x(hx)),
                                   torch.sigmoid(self.attn_gate_y(hy)), transformer_options=transformer_options)
        x = torch.addcmul(x, xg1, self.attn_post_norm_x(attn_x))
        x = torch.addcmul(x, xg2, self.mlp_post_norm_x(self.mlp_x(modulate(self.norm_x2(x), xs2, xc2))))
        if last:
            return x, y
        y = torch.addcmul(y, yg1, self.attn_post_norm_y(attn_y))
        y = torch.addcmul(y, yg2, self.mlp_post_norm_y(self.mlp_y(modulate(self.norm_y2(y), ys2, yc2))))
        return x, y


class SingleStreamBlock(nn.Module):
    """Text and image share every weight; tokens are concatenated text first."""
    def __init__(self, dim, num_heads, num_kv_heads, mlp_ratio, mlp_hidden=None, dtype=None, device=None, operations=None):
        super().__init__()
        kw = dict(dtype=dtype, device=device)
        self.comfy_attention = ComfyAttention()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.norm1 = operations.RMSNorm(dim, eps=1e-6, **kw)
        self.norm2 = operations.RMSNorm(dim, eps=1e-6, **kw)
        self.q_proj = operations.Linear(dim, dim, bias=False, **kw)
        self.k_proj = operations.Linear(dim, num_kv_heads * self.head_dim, bias=False, **kw)
        self.v_proj = operations.Linear(dim, num_kv_heads * self.head_dim, bias=False, **kw)
        self.q_norm = operations.RMSNorm(self.head_dim, eps=1e-6, **kw)
        self.k_norm = operations.RMSNorm(self.head_dim, eps=1e-6, **kw)
        self.attn_gate = operations.Linear(dim, dim, bias=False, **kw)
        self.attn_proj = operations.Linear(dim, dim, bias=True, **kw)
        self.attn_post_norm = operations.RMSNorm(dim, eps=1e-6, **kw)
        self.mlp = SwiGLU(dim, mlp_ratio, mlp_hidden, operations=operations, **kw)
        self.mlp_post_norm = operations.RMSNorm(dim, eps=1e-6, **kw)
        self.adaln = SharedBiasModulation(dim, **kw)

    def forward(self, x, y, core_img, core_txt, pe_img, pe_txt, last=False, transformer_options={}):
        batch, n_img, _ = x.shape
        n_txt = y.shape[1]
        s1, c1, g1, s2, c2, g2 = self.adaln(core_img).chunk(6, dim=-1)
        tokens = torch.cat([y, x], dim=1)
        h = modulate(self.norm1(tokens), s1, c1)
        n = n_txt + n_img
        q = self.q_norm(self.q_proj(h).view(batch, n, self.num_heads, self.head_dim).transpose(1, 2))
        k = self.k_norm(self.k_proj(h).view(batch, n, self.num_kv_heads, self.head_dim).transpose(1, 2))
        v = self.v_proj(h).view(batch, n, self.num_kv_heads, self.head_dim).transpose(1, 2)
        qy, ky = apply_rope(q[:, :, :n_txt], k[:, :, :n_txt], pe_txt[None, None])
        qx, kx = apply_rope(q[:, :, n_txt:], k[:, :, n_txt:], pe_img[None, None])
        attn = attention(torch.cat([qy, qx], dim=2), torch.cat([ky, kx], dim=2), v,
                         self.num_heads, self.comfy_attention, transformer_options=transformer_options)
        if last:
            # the final block's text output is discarded: text only supplied keys/values above
            attn, h, tokens = attn[:, n_txt:], h[:, n_txt:], x
        attn = self.attn_post_norm(self.attn_proj(attn * torch.sigmoid(self.attn_gate(h))))
        tokens = torch.addcmul(tokens, g1, attn)
        tokens = torch.addcmul(tokens, g2, self.mlp_post_norm(self.mlp(modulate(self.norm2(tokens), s2, c2))))
        if last:
            return tokens, y
        return tokens[:, n_txt:], tokens[:, :n_txt]


# -- pixel stage

class PixelEmbedder(nn.Module):
    """Per-pixel linear + full-resolution 2D sincos, grouped into [B*L, p*p, d] per-patch sequences."""
    def __init__(self, in_channels, hidden_size, patch_size, dtype=None, device=None, operations=None):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.proj = operations.Linear(in_channels, hidden_size, bias=True, dtype=dtype, device=device)

    def forward(self, x):
        batch, _, height, width = x.shape
        p = self.patch_size
        tokens = self.proj(x.permute(0, 2, 3, 1))
        grid_y, grid_x = torch.meshgrid(torch.arange(height, device=x.device, dtype=torch.float64),
                                        torch.arange(width, device=x.device, dtype=torch.float64), indexing="ij")
        d = self.hidden_size // 2
        pos = torch.cat([get_1d_sincos_pos_embed_from_grid_torch(d, grid_x, device=x.device, dtype=torch.float64),
                         get_1d_sincos_pos_embed_from_grid_torch(d, grid_y, device=x.device, dtype=torch.float64)], dim=1)
        tokens = tokens + pos.view(height, width, -1).to(tokens.dtype)
        tokens = tokens.view(batch, height // p, p, width // p, p, -1).permute(0, 1, 3, 2, 4, 5)
        return tokens.reshape(batch * (height // p) * (width // p), p * p, -1)


class PiTBlock(nn.Module):
    """Pixel transformer block, post-modulation layout: (scale, shift) on each branch output, no gates."""
    def __init__(self, hidden_size, cond_dim, pixels_per_patch, attn_hidden_size, num_heads, mlp_ratio,
                 dtype=None, device=None, operations=None):
        super().__init__()
        kw = dict(dtype=dtype, device=device)
        self.pixels_per_patch = pixels_per_patch
        self.norm1 = operations.RMSNorm(hidden_size, eps=1e-6, **kw)
        self.norm2 = operations.RMSNorm(hidden_size, eps=1e-6, **kw)
        self.adaln = operations.Linear(cond_dim, 4 * hidden_size * pixels_per_patch, bias=True, **kw)
        self.compress = operations.Linear(pixels_per_patch * hidden_size, attn_hidden_size, bias=True, **kw)
        self.expand = operations.Linear(attn_hidden_size, pixels_per_patch * hidden_size, bias=True, **kw)
        self.attn = SelfAttention(attn_hidden_size, num_heads, qk_norm=True, operations=operations, **kw)
        self.mlp = GeluMLP(hidden_size, mlp_ratio, operations=operations, **kw)

    def _global_attn(self, pixels, pe, batch, transformer_options):
        """Compact each patch to one token, attend over the patch grid, expand back."""
        compact = self.compress(pixels.reshape(pixels.shape[0], -1))
        attn = self.attn(compact.view(batch, -1, compact.shape[-1]), pe=pe, transformer_options=transformer_options)
        return self.expand(attn.reshape(pixels.shape[0], -1)).view_as(pixels)

    def forward(self, x, cond, pe, batch, transformer_options={}):
        scale1, shift1, scale2, shift2 = self.adaln(cond).view(x.shape[0], self.pixels_per_patch, -1).chunk(4, dim=-1)
        x = x + modulate(self._global_attn(self.norm1(x), pe, batch, transformer_options), shift1, scale1)
        return x + modulate(self.mlp(self.norm2(x)), shift2, scale2)


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, dtype=None, device=None, operations=None):
        super().__init__()
        self.norm = operations.RMSNorm(hidden_size, eps=1e-6, dtype=dtype, device=device)
        self.linear = operations.Linear(hidden_size, out_channels, bias=True, dtype=dtype, device=device)

    def forward(self, x):
        return self.linear(self.norm(x))


class Embedder(nn.Module):
    def __init__(self, in_dim, out_dim, dtype=None, device=None, operations=None):
        super().__init__()
        self.proj = operations.Linear(in_dim, out_dim, bias=True, dtype=dtype, device=device)

    def forward(self, x):
        return self.proj(x)


class IrisDiT(nn.Module):
    """Forward:
      x:              [B, 3, H, W] noisy RGB in [-1, 1]
      timesteps:      [B] model time in [0, 1000] (1000 * shifted sigma)
      context:        [B, T, 12 * 2560] Qwen3-VL-4B hidden states, 12 layers flattened per token
      attention_mask: [B, T] (1 = real token)
    Returns the flow velocity (noise - x0), [B, 3, H, W].
    """
    def __init__(self, in_channels=3, patch_size=16, hidden_size=2560, depth=24, dual_depth=8,
                 num_heads=20, num_kv_heads=5, mlp_ratio=4.0, mlp_hidden=None, timestep_max_period=10.0,
                 rope_theta=10000.0, rope_scale=16.0, text_rope_theta=10000.0,
                 text_dim=2560, text_len=300, text_layers=12, text_layer_heads=32, text_layer_mlp_ratio=1.3,
                 pixel_hidden_size=16, pixel_attn_hidden_size=1280, pixel_num_heads=10, pixel_depth=4,
                 pixel_mlp_ratio=4.0, image_model=None, dtype=None, device=None, operations=None):
        super().__init__()
        kw = dict(dtype=dtype, device=device, operations=operations)
        self.dtype = dtype
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.text_len = text_len
        self.rope_theta = rope_theta
        self.rope_scale = rope_scale
        self.text_rope_theta = text_rope_theta
        self.pixel_attn_head_dim = pixel_attn_hidden_size // pixel_num_heads

        self.s_embedder = Embedder(patch_size * patch_size * in_channels, hidden_size, **kw)
        self.t_embedder = TimestepEmbedder(hidden_size, max_period=timestep_max_period, **kw)
        self.y_embedder = LayerwiseTextEmbedder(text_dim, hidden_size, text_layers, text_layer_heads, text_layer_mlp_ratio,
                                                num_heads, mlp_ratio, **kw)
        self.y_pos_embedding = nn.Parameter(torch.empty(1, text_len, hidden_size, dtype=dtype, device=device))
        self.modulation_cores = nn.ModuleDict({
            "adaln_img": operations.Linear(hidden_size, 6 * hidden_size, bias=True, dtype=dtype, device=device),
            "adaln_txt": operations.Linear(hidden_size, 6 * hidden_size, bias=True, dtype=dtype, device=device),
        })
        self.blocks = nn.ModuleList([
            (DualStreamBlock if i < dual_depth else SingleStreamBlock)(hidden_size, num_heads, num_kv_heads, mlp_ratio, mlp_hidden, **kw)
            for i in range(depth)
        ])
        self.pixel_embedder = PixelEmbedder(in_channels, pixel_hidden_size, patch_size, **kw)
        self.pixel_blocks = nn.ModuleList([
            PiTBlock(pixel_hidden_size, hidden_size, patch_size * patch_size, pixel_attn_hidden_size, pixel_num_heads,
                     pixel_mlp_ratio, **kw)
            for _ in range(pixel_depth)
        ])
        self.final_layer = FinalLayer(pixel_hidden_size, in_channels, **kw)

    def forward(self, x, timesteps, context=None, attention_mask=None, transformer_options={}, **kwargs):
        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._forward, self,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options),
        ).execute(x, timesteps, context, attention_mask, transformer_options, **kwargs)

    def _forward(self, x, timesteps, context=None, attention_mask=None, transformer_options={}, **kwargs):
        p = self.patch_size
        h_orig, w_orig = x.shape[-2:]
        x = comfy.ldm.common_dit.pad_to_patch_size(x, (p, p))
        batch, _, height, width = x.shape
        grid_h, grid_w = height // p, width // p
        head_dim = self.hidden_size // self.num_heads

        t_emb = self.t_embedder(timesteps.reshape(-1), x.dtype).view(batch, 1, -1)
        cond = F.silu(t_emb)

        context, text_mask = pad_text(context, attention_mask, self.text_len)
        y = self.y_embedder(context, text_mask, transformer_options=transformer_options)
        y = y + comfy.model_management.cast_to(self.y_pos_embedding, dtype=y.dtype, device=y.device)

        pe_img = rope_2d(head_dim, grid_h, grid_w, self.rope_theta, self.rope_scale, x.device)
        pe_txt = rope_1d(head_dim, y.shape[1], self.text_rope_theta, x.device)
        core_img = self.modulation_cores["adaln_img"](cond)
        core_txt = self.modulation_cores["adaln_txt"](cond)

        s = self.s_embedder(F.unfold(x, kernel_size=p, stride=p).transpose(1, 2))
        for i, block in enumerate(self.blocks):
            s, y = block(s, y, core_img, core_txt, pe_img, pe_txt, last=i == len(self.blocks) - 1,
                         transformer_options=transformer_options)
        s = F.silu(t_emb + s)  # timestep re-fused into every patch token

        s_cond = s.reshape(-1, s.shape[-1])
        pe_pix = rope_2d(self.pixel_attn_head_dim, grid_h, grid_w, self.rope_theta, self.rope_scale, x.device)
        pixels = self.pixel_embedder(x)
        for block in self.pixel_blocks:
            pixels = block(pixels, s_cond, pe_pix, batch, transformer_options=transformer_options)
        out = self.final_layer(pixels)  # [B*L, p*p, C]

        out = out.view(batch, grid_h * grid_w, p * p, -1).permute(0, 3, 2, 1).reshape(batch, -1, grid_h * grid_w)
        out = F.fold(out, output_size=(height, width), kernel_size=p, stride=p)
        return out[:, :, :h_orig, :w_orig]
