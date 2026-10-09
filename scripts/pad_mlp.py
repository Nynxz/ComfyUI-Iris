"""Zero-pad the trunk SwiGLU hidden width so int8 / convrot can quantize all of it.

    python scripts/pad_mlp.py iris-3b_bf16.safetensors iris-3b_bf16_mlp6912.safetensors [--multiple 256]
    python quant_int8_convrot.py iris-3b_bf16_mlp6912.safetensors iris-3b_int8_convrot.safetensors

Iris's SwiGLU hidden is int(2 * 10240 / 3) = 6826, which breaks int8 twice: w1/w3 have N=6826
(cuBLAS int8 GEMM needs N % 4 == 0) and w2 has K=6826 (no Hadamard group size divides it).
Padding to 6912 = 27 * 256 adds zero rows to w1/w3 and zero columns to w2. The extra hidden
units are silu(0) * 0 = 0, so the model's output is unchanged; ComfyUI-Iris reads the width
from the checkpoint. Only the patch-trunk MLPs are padded (the text adapter stays BF16).
"""

import argparse
import re

import torch
from safetensors import safe_open
from safetensors.torch import save_file

TRUNK_MLP = re.compile(r"^blocks\.\d+\.mlp(_[xy])?\.(w1|w2|w3)\.weight$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--multiple", type=int, default=256)
    args = ap.parse_args()
    out, padded = {}, 0
    with safe_open(args.src, "pt") as f:
        for key in f.keys():
            t = f.get_tensor(key)
            m = TRUNK_MLP.match(key)
            if m:
                axis = 1 if m.group(2) == "w2" else 0
                pad = -t.shape[axis] % args.multiple
                if pad:
                    t = torch.nn.functional.pad(t, (0, pad) if axis == 1 else (0, 0, 0, pad))
                    padded += 1
            out[key] = t.contiguous()
        metadata = f.metadata() or {}
    save_file(out, args.dst, metadata=metadata)
    print(f"wrote {args.dst}: padded {padded} trunk MLP weights to a multiple of {args.multiple}")


if __name__ == "__main__":
    main()
