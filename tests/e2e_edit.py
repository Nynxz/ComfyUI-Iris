"""Edit LoRAs through ComfyUI's stock nodes, as a grid: one row per edit, one column per LoRA.

    <comfy>/.venv/bin/python tests/e2e_edit.py --comfy <ComfyUI> --manifest preview.jsonl \
        --lora iris/step250.safetensors --lora iris/step500.safetensors --out grid.jpg

Graph per cell: UNETLoader -> LoraLoaderModelOnly -> ModelSamplingSD3; CLIPLoader (krea2) for the
instruction and an empty negative; VAE Encode (pixel_space) of the source -> ReferenceLatent on BOTH
the positive and the negative; KSampler; VAE Decode. Manifest rows: {"image": path, "instruction": str}.
"""

import argparse
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--lora", action="append", required=True, help="name in the loras folder; repeatable")
    ap.add_argument("--unet", default="iris-3b/iris-3b_int8_convrot.safetensors")
    ap.add_argument("--clip", default="qwen3vl_4b_bf16.safetensors")
    ap.add_argument("--side", type=int, default=768)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg", default="3.0", help="CFG scale, or comma-separated scales (one column each)")
    ap.add_argument("--shift", type=float, default=3.5)
    ap.add_argument("--sampler", default="euler")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pack_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, args.comfy)
    sys.argv = sys.argv[:1]
    import comfy.options

    comfy.options.enable_args_parsing()
    from utils.extra_config import load_extra_path_config

    extra = os.path.join(args.comfy, "extra_model_paths.yaml")
    if os.path.isfile(extra):
        load_extra_path_config(extra)

    import importlib

    import numpy as np
    import torch
    from PIL import Image, ImageDraw, ImageFont

    sys.path.insert(0, os.path.dirname(pack_root))
    importlib.import_module(os.path.basename(pack_root))
    import nodes
    from comfy_extras.nodes_edit_model import ReferenceLatent
    from comfy_extras.nodes_model_advanced import ModelSamplingSD3

    rows = [json.loads(line) for line in open(args.manifest) if line.strip()]
    (base,) = nodes.UNETLoader().load_unet(args.unet, "default")
    (clip,) = nodes.CLIPLoader().load_clip(args.clip, type="krea2")
    (vae,) = nodes.VAELoader().load_vae("pixel_space")
    (empty,) = nodes.CLIPTextEncode().encode(clip, "")

    sources, prompts = [], []
    for row in rows:
        img = Image.open(row["image"])  # relative paths resolve against the working directory
        img = img.convert("RGB")
        scale = (args.side * args.side / (img.width * img.height)) ** 0.5
        img = img.resize((max(16, round(img.width * scale / 16) * 16), max(16, round(img.height * scale / 16) * 16)),
                         Image.LANCZOS)
        sources.append(img)
        (pos,) = nodes.CLIPTextEncode().encode(clip, row["instruction"])
        prompts.append(pos)

    def run_lora(model, cfg):
        outs = []
        for img, pos in zip(sources, prompts):
            pixels = torch.from_numpy(np.asarray(img).astype(np.float32) / 255.0)[None]
            (ref,) = nodes.VAEEncode().encode(vae, pixels)
            positive = ReferenceLatent.execute(pos, ref).result[0]
            negative = ReferenceLatent.execute(empty, ref).result[0]
            (samples,) = nodes.KSampler().sample(model, args.seed, args.steps, cfg, args.sampler, "simple",
                                                 positive, negative, ref, denoise=1.0)
            (image,) = nodes.VAEDecode().decode(vae, samples)
            outs.append(Image.fromarray((image[0].cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)))
        return outs

    cfgs = [float(c) for c in args.cfg.split(",")]
    columns, results = [], []  # one column per (lora, cfg); results[col][row] -> PIL
    for lora in args.lora:
        (model,) = nodes.LoraLoaderModelOnly().load_lora_model_only(base, lora, 1.0)
        (model,) = ModelSamplingSD3().patch(model, args.shift)
        for cfg in cfgs:
            columns.append(os.path.splitext(os.path.basename(lora))[0] + (f" cfg {cfg:g}" if len(cfgs) > 1 else ""))
            results.append(run_lora(model, cfg))
        print(f"done {lora}", flush=True)

    cell, label = 320, 34
    cell, label = 320, 34
    font = ImageFont.load_default(size=20)
    sheet = Image.new("RGB", ((1 + len(columns)) * cell, label + len(rows) * (label + cell)), "white")
    draw = ImageDraw.Draw(sheet)
    for c, title in enumerate(["source"] + columns):
        draw.text((c * cell + 8, 6), title, fill="black", font=font)
    y = label
    for r, row in enumerate(rows):
        draw.rectangle((0, y, sheet.width, y + label), fill=(235, 235, 235))
        draw.text((8, y + 6), f'"{row["instruction"]}"', fill="black", font=font)
        y += label
        for c, im in enumerate([sources[r]] + [col[r] for col in results]):
            s = cell / max(im.size)
            im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
            sheet.paste(im, (c * cell + (cell - im.width) // 2, y + (cell - im.height) // 2))
        y += cell
    sheet.save(args.out, quality=90)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
