"""Upscaler and depth through ComfyUI's stock nodes, as one comparison sheet.

    <comfy>/.venv/bin/python tests/e2e_tasks.py --comfy <ComfyUI> --image photo.png --out sheet.jpg

Upscale (LoRA on base Iris, and the full upscaler checkpoint): Load Image -> 256px crop ->
ImageScaleBy x4 (bicubic) -> VAE Encode (pixel_space) -> LatentMultiply x2 -> KSamplerAdvanced
(no added noise, 2 steps from step 1 = one step 0.5 -> 0, shift 1, CFG 1, empty prompt) ->
VAE Decode. The x2 undoes the sampler scaling the input by (1 - sigma) = 0.5; the step itself
is x - 0.5 * v, the released restorer.
Depth (folded depth checkpoint): VAE Encode -> KSamplerAdvanced (no added noise, 1 step from
sigma 1) -> VAE Decode; the pack's depth sampling returns the model output as the image.
"""

import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="iris-3b/iris-3b_bf16.safetensors")
    ap.add_argument("--lora", default="iris/iris-3b-upscaler_r128.safetensors")
    ap.add_argument("--upscaler", default="iris-3b/iris-3b-upscaler_bf16.safetensors")
    ap.add_argument("--depth", default="iris-3b/iris-3b-depth_bf16.safetensors")
    ap.add_argument("--clip", default="qwen3vl_4b_bf16.safetensors")
    args = ap.parse_args()

    pack_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, args.comfy)
    sys.argv = sys.argv[:1]
    import comfy.options

    comfy.options.enable_args_parsing()
    from utils.extra_config import load_extra_path_config

    load_extra_path_config(os.path.join(args.comfy, "extra_model_paths.yaml"))
    import importlib

    import numpy as np
    import torch
    from PIL import Image, ImageDraw, ImageFont

    sys.path.insert(0, os.path.dirname(pack_root))
    importlib.import_module(os.path.basename(pack_root))
    import comfy.model_management
    import nodes
    from comfy_extras.nodes_model_advanced import ModelSamplingSD3

    def to_image(pil):
        return torch.from_numpy(np.asarray(pil.convert("RGB")).astype(np.float32) / 255.0)[None]

    def to_pil(image):
        return Image.fromarray((image[0].cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8))

    src = Image.open(args.image).convert("RGB")
    side = min(src.size)
    crop = src.crop(((src.width - side) // 2, (src.height - side) // 2, (src.width + side) // 2, (src.height + side) // 2))
    lq = crop.resize((256, 256), Image.LANCZOS)

    (clip,) = nodes.CLIPLoader().load_clip(args.clip, type="krea2")
    (vae,) = nodes.VAELoader().load_vae("pixel_space")
    (empty,) = nodes.CLIPTextEncode().encode(clip, "")
    del clip
    comfy.model_management.unload_all_models()

    (bicubic,) = nodes.ImageScaleBy().upscale(to_image(lq), "bicubic", 4.0)

    def upscale(model):
        (model,) = ModelSamplingSD3().patch(model, 1.0)
        (latent,) = nodes.VAEEncode().encode(vae, bicubic)
        latent = {**latent, "samples": latent["samples"] * 2.0}  # what the LatentMultiply node does
        (out,) = nodes.KSamplerAdvanced().sample(model, "disable", 0, 2, 1.0, "euler", "simple", empty, empty, latent,
                                                 1, 10000, "disable")
        return to_pil(nodes.VAEDecode().decode(vae, out)[0])

    (base,) = nodes.UNETLoader().load_unet(args.base, "default")
    (with_lora,) = nodes.LoraLoaderModelOnly().load_lora_model_only(base, args.lora, 1.0)
    lora_out = upscale(with_lora)
    print("upscale via LoRA done", flush=True)
    del base, with_lora
    comfy.model_management.unload_all_models()

    (full,) = nodes.UNETLoader().load_unet(args.upscaler, "default")
    full_out = upscale(full)
    print("upscale via full checkpoint done", flush=True)
    del full
    comfy.model_management.unload_all_models()

    (depth_model,) = nodes.UNETLoader().load_unet(args.depth, "default")
    rgb = crop.resize((768, 768), Image.LANCZOS)
    (latent,) = nodes.VAEEncode().encode(vae, to_image(rgb))
    (out,) = nodes.KSamplerAdvanced().sample(depth_model, "disable", 0, 1, 1.0, "euler", "simple", empty, empty, latent,
                                             0, 10000, "disable")
    depth_out = to_pil(nodes.VAEDecode().decode(vae, out)[0])
    print("depth done", flush=True)

    a = np.asarray(lora_out).astype(np.float64)
    b = np.asarray(full_out).astype(np.float64)
    psnr = 10 * np.log10(255 ** 2 / max(((a - b) ** 2).mean(), 1e-9))
    print(f"upscale: LoRA vs full checkpoint PSNR {psnr:.1f} dB", flush=True)

    cell, label = 512, 34
    tiles = [("input 256px, bicubic x4", Image.fromarray(np.asarray(to_pil(bicubic)))), ("upscaler LoRA (base + r128)", lora_out),
             ("upscaler full checkpoint", full_out), ("depth (folded checkpoint)", depth_out)]
    font = ImageFont.load_default(size=20)
    sheet = Image.new("RGB", (cell * len(tiles), cell + label), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (title, im) in enumerate(tiles):
        draw.text((i * cell + 8, 6), title, fill="black", font=font)
        sheet.paste(im.resize((cell, cell), Image.LANCZOS), (i * cell, label))
    sheet.save(args.out, quality=92)
    lora_out.save(os.path.splitext(args.out)[0] + "_lora_full.png")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
