"""Registers Iris-3B with ComfyUI's native model loading.

After ``register()`` the stock nodes handle everything:
  Load Diffusion Model (UNETLoader)  -> detects the checkpoint below
  Load CLIP, type "krea2"            -> Qwen3-VL-4B with the same 12-layer tap Iris trained on
  Load VAE, "pixel_space"            -> no-op pixel VAE
  Empty Latent Image                 -> auto-expanded to a 3-channel full-resolution latent
Nothing in ComfyUI is replaced: detection is chained in front of the built-in function and the
model config is appended to ``comfy.supported_models.models``.
"""

import logging
import math
import re

import torch

import comfy.conds
import comfy.latent_formats
import comfy.model_base
import comfy.model_detection
import comfy.supported_models
import comfy.supported_models_base
import comfy.text_encoders.krea2

from .model import IrisDiT, pad_text


# trunk SwiGLU width is padded to this multiple at load (exact: the extra units are always
# zero) so every checkpoint, LoRA and int8/convrot quantization shares one layout
MLP_MULTIPLE = 256
TRUNK_MLP = re.compile(r"^blocks\.\d+\.mlp(_[xy])?\.(w1|w2|w3)\.weight$")


def pad_mlp(state_dict, prefix=""):
    for k in list(state_dict.keys()):
        m = TRUNK_MLP.match(k[len(prefix):]) if k.startswith(prefix) else None
        v = state_dict[k]
        if m is None or not v.is_floating_point():
            continue
        axis = 1 if m.group(2) == "w2" else 0
        pad = -v.shape[axis] % MLP_MULTIPLE
        if pad:
            state_dict[k] = torch.nn.functional.pad(v, (0, pad) if axis == 1 else (0, 0, 0, pad))
    return state_dict


class IrisPixel(comfy.latent_formats.ChromaRadiance):
    """Pixel space: the "latent" is RGB in [-1, 1]."""
    pass


class IrisModel(comfy.model_base.BaseModel):
    def __init__(self, model_config, model_type=comfy.model_base.ModelType.FLOW, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=IrisDiT)
        self.memory_usage_factor_conds = ("ref_latents",)

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            # fixed text_len for every cond, so cond/uncond batch together exactly like the
            # reference instead of being repeated to a common length by CONDCrossAttn
            context, mask = pad_text(cross_attn, kwargs.get("attention_mask", None), self.diffusion_model.text_len)
            out["c_crossattn"] = comfy.conds.CONDRegular(context)
            out["attention_mask"] = comfy.conds.CONDRegular(mask)
        ref_latents = kwargs.get("reference_latents", None)
        if ref_latents is not None:
            out["ref_latents"] = comfy.conds.CONDList([self.process_latent_in(lat) for lat in ref_latents])
        return out

    def extra_conds_shapes(self, **kwargs):
        ref_latents = kwargs.get("reference_latents", None)
        if ref_latents is None:
            return {}
        return {"ref_latents": [1, 3, sum(math.prod(lat.size()) for lat in ref_latents) // 3]}


class Iris(comfy.supported_models_base.BASE):
    unet_config = {
        "image_model": "iris",
    }

    unet_extra_config = {}

    sampling_settings = {
        "shift": 4.0,  # flow.shift of the released 1024px checkpoint
    }

    latent_format = IrisPixel
    memory_usage_factor = 0.05
    supported_inference_dtypes = [torch.bfloat16, torch.float32]

    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    def get_model(self, state_dict, prefix="", device=None):
        return IrisModel(self, device=device)

    def process_unet_state_dict(self, state_dict):
        return pad_mlp(state_dict)

    def process_vae_state_dict(self, state_dict):
        return {"pixel_space_vae": torch.tensor(1.0)}

    def clip_target(self, state_dict={}):
        return comfy.supported_models_base.ClipTarget(comfy.text_encoders.krea2.Krea2Tokenizer, comfy.text_encoders.krea2.te())


class IrisDepthModel(IrisModel):
    """Iris-3B depth fine-tune (folded by scripts/fold_depth.py, tagged iris_task=depth).

    Depth is one forward pass at model time 1000 on the clean RGB image, and the model output
    itself is the (relative log) depth. So the image is passed in unscaled at any sigma and
    the "denoised" result is the raw output: one sampler step from sigma 1 to 0 with no added
    noise yields the depth map, identical on all three channels."""

    def __init__(self, model_config, model_type=comfy.model_base.ModelType.FLOW, device=None):
        super().__init__(model_config, model_type, device=device)
        base = type(self.model_sampling)

        class DepthSampling(base):
            def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
                return latent_image

            def calculate_denoised(self, sigma, model_output, model_input):
                return model_output

            def inverse_noise_scaling(self, sigma, latent):
                return latent

        self.model_sampling = DepthSampling(model_config)


class IrisDepth(Iris):
    unet_config = {
        "image_model": "iris_depth",
    }

    sampling_settings = {
        "shift": 1.0,
    }

    def get_model(self, state_dict, prefix="", device=None):
        return IrisDepthModel(self, device=device)


def detect_iris_config(state_dict, key_prefix=""):
    """unet_config for an Iris-3B state dict, derived from tensor shapes; None if it is not one."""
    keys = state_dict.keys()
    if "{}y_embedder.layer_pool.weight".format(key_prefix) not in keys or "{}pixel_blocks.0.compress.weight".format(key_prefix) not in keys:
        return None

    def shape(name):
        return state_dict["{}{}".format(key_prefix, name)].shape

    hidden_size, patch_dim = shape("s_embedder.proj.weight")
    pixel_hidden_size, in_channels = shape("pixel_embedder.proj.weight")
    head_dim = shape("blocks.0.attn.q_norm_x.weight")[0]
    pixel_attn_hidden_size = shape("pixel_blocks.0.compress.weight")[0]
    return {
        "image_model": "iris",
        "in_channels": in_channels,
        "patch_size": round((patch_dim // in_channels) ** 0.5),
        "hidden_size": hidden_size,
        "depth": comfy.model_detection.count_blocks(keys, "{}blocks.".format(key_prefix) + "{}."),
        "dual_depth": comfy.model_detection.count_blocks(keys, "{}blocks.".format(key_prefix) + "{}.attn.q_proj_x."),
        "num_heads": hidden_size // head_dim,
        "num_kv_heads": shape("blocks.0.attn.k_proj_x.weight")[0] // head_dim,
        "mlp_hidden": -(-shape("blocks.0.mlp_x.w1.weight")[0] // MLP_MULTIPLE) * MLP_MULTIPLE,
        "text_dim": shape("y_embedder.refiner.proj.weight")[1],
        "text_len": shape("y_pos_embedding")[1],
        "text_layers": shape("y_embedder.layer_pool.weight")[1],
        "pixel_hidden_size": pixel_hidden_size,
        "pixel_attn_hidden_size": pixel_attn_hidden_size,
        "pixel_num_heads": pixel_attn_hidden_size // shape("pixel_blocks.0.attn.q_norm.weight")[0],
        "pixel_depth": comfy.model_detection.count_blocks(keys, "{}pixel_blocks.".format(key_prefix) + "{}."),
    }


def register():
    detect = comfy.model_detection.detect_unet_config
    if not getattr(detect, "_iris", False):
        def detect_unet_config(state_dict, key_prefix, metadata=None):
            config = detect_iris_config(state_dict, key_prefix)
            if config is not None and (metadata or {}).get("iris_task") == "depth":
                config["image_model"] = "iris_depth"
            if config is not None:
                return config
            return detect(state_dict, key_prefix, metadata=metadata)
        detect_unet_config._iris = True
        comfy.model_detection.detect_unet_config = detect_unet_config

    for config in (Iris, IrisDepth):
        name = config.unet_config["image_model"]
        if not any(getattr(m, "unet_config", {}).get("image_model") == name for m in comfy.supported_models.models):
            comfy.supported_models.models.append(config)
    logging.info("Iris-3B: registered with Load Diffusion Model")
