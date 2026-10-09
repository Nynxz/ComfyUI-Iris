"""Headless end-to-end run through ComfyUI's stock node classes (the same graph a user builds).

    <comfy>/.venv/bin/python tests/e2e.py --comfy <ComfyUI> --out <dir> --unet iris-3b/iris-3b_bf16.safetensors \
        [--unet iris-3b/iris-3b_int8_convrot.safetensors ...] --clip qwen3vl_4b_bf16.safetensors

Model names are relative to the configured diffusion_models / text_encoders folders
(extra_model_paths.yaml is honoured). Writes one PNG per UNet, same seed and prompt.
"""

import argparse
import os
import sys
import time

DEFAULT_PROMPT = "a red fox sleeping in fresh snow, golden hour, soft light, highly detailed photograph"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--unet", action="append", required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--negative", default="")
    ap.add_argument("--size", type=int, nargs=2, default=(1024, 1024), metavar=("W", "H"))
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--sampler", default="euler")
    ap.add_argument("--scheduler", default="simple")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pack_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, args.comfy)
    sys.argv = sys.argv[:1]
    import comfy.options

    comfy.options.enable_args_parsing()
    import folder_paths
    from utils.extra_config import load_extra_path_config

    extra = os.path.join(args.comfy, "extra_model_paths.yaml")
    if os.path.isfile(extra):
        load_extra_path_config(extra)

    import numpy as np
    import torch
    from PIL import Image

    sys.path.insert(0, os.path.dirname(pack_root))
    import importlib

    importlib.import_module(os.path.basename(pack_root))  # registers Iris, as ComfyUI's loader would
    import nodes

    (clip,) = nodes.CLIPLoader().load_clip(args.clip, type="krea2")
    (vae,) = nodes.VAELoader().load_vae("pixel_space")
    (positive,) = nodes.CLIPTextEncode().encode(clip, args.prompt)
    (negative,) = nodes.CLIPTextEncode().encode(clip, args.negative)
    (latent,) = nodes.EmptyLatentImage().generate(args.size[0], args.size[1], 1)
    os.makedirs(args.out, exist_ok=True)

    for unet in args.unet:
        (model,) = nodes.UNETLoader().load_unet(unet, "default")
        t0 = time.time()
        (samples,) = nodes.KSampler().sample(model, args.seed, args.steps, args.cfg, args.sampler, args.scheduler,
                                             positive, negative, latent, denoise=1.0)
        (image,) = nodes.VAEDecode().decode(vae, samples)
        print(f"{unet}: {time.time() - t0:.1f}s  latent {tuple(samples['samples'].shape)}")
        name = os.path.splitext(os.path.basename(unet))[0]
        path = os.path.join(args.out, f"{name}_{args.sampler}_{args.steps}.png")
        Image.fromarray((image[0].cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)).save(path)
        print("wrote", path)
        del model
        import comfy.model_management

        comfy.model_management.unload_all_models()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
