"""Fold Iris-3B's depth fine-tune into the base architecture (exact).

    python scripts/fold_depth.py depth/model.safetensors iris-3b-depth.safetensors

The depth model feeds RGB plus an all-zero 4th channel and reduces the 3-channel output
with a 1x1 conv. The zero channel contributes nothing, so its input weights are dropped,
and the 1x1 conv is linear, so it is merged into final_layer.linear and repeated on all
three output channels (depth comes out as a grey image). The result has the base
checkpoint's keys and shapes; the file is tagged `iris_task=depth`.
"""

import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main(src, dst):
    out = {}
    with safe_open(src, "pt") as f:
        for key in f.keys():
            if key.startswith("pixel."):
                out[key[len("pixel."):]] = f.get_tensor(key)
        reducer_w = f.get_tensor("depth_reducer.weight").float().reshape(1, 3)
        reducer_b = f.get_tensor("depth_reducer.bias").float()

    w = out["s_embedder.proj.weight"]  # [D, 4 * p * p], channel-major (unfold layout)
    out["s_embedder.proj.weight"] = w[:, : w.shape[1] // 4 * 3].contiguous()
    out["pixel_embedder.proj.weight"] = out["pixel_embedder.proj.weight"][:, :3].contiguous()

    fw, fb = out["final_layer.linear.weight"].float(), out["final_layer.linear.bias"].float()
    depth_w = reducer_w @ fw  # [1, hidden]
    depth_b = reducer_w @ fb + reducer_b  # [1]
    dtype = out["final_layer.linear.weight"].dtype
    out["final_layer.linear.weight"] = depth_w.repeat(3, 1).to(dtype).contiguous()
    out["final_layer.linear.bias"] = depth_b.repeat(3).to(dtype).contiguous()
    save_file(out, dst, metadata={"format": "pt", "iris_task": "depth"})
    print(f"wrote {dst}: {len(out)} tensors")


if __name__ == "__main__":
    main(*sys.argv[1:3])
