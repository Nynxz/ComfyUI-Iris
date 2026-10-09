"""Cast the released FP32 Iris-3B checkpoint to BF16 (12 GB -> 6 GB), keys unchanged.

    python scripts/convert_bf16.py iris-3b.safetensors iris-3b_bf16.safetensors

ComfyUI already casts the FP32 file to BF16 at load; this only saves disk/RAM, and gives
quant_int8_convrot.py a BF16 source so its passthrough tensors come out BF16.
"""

import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main(src, dst):
    with safe_open(src, "pt") as f:
        out = {k: f.get_tensor(k).to(torch.bfloat16) for k in f.keys()}
        metadata = f.metadata() or {}
    save_file(out, dst, metadata=metadata)
    print(f"wrote {dst}: {len(out)} tensors")


if __name__ == "__main__":
    main(*sys.argv[1:3])
