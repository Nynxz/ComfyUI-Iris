"""Extract a ComfyUI LoRA from a fine-tuned Iris-3B checkpoint (fine-tune minus base).

    python scripts/extract_lora.py base.safetensors finetune.safetensors out.safetensors --rank 128

Big linear layers (both sides >= --min-dim) get a rank-r SVD of their weight difference
(`lora_down` / `lora_up` / `alpha`). Every other tensor that changed (norms, biases, small
linears, embeddings, per-block modulation biases) is stored as an exact full difference
(`.diff` / `.diff_b`), which ComfyUI's LoRA loader applies as-is. Trunk MLP adapters are
zero-padded to the 6912 width ComfyUI-Iris loads models with. --report prints how much of
each layer's difference the chosen rank keeps.
"""

import argparse
import re

import torch
from safetensors import safe_open
from safetensors.torch import save_file

PREFIX = "diffusion_model."
MLP_MULTIPLE = 256
TRUNK_MLP = re.compile(r"^blocks\.\d+\.mlp(_[xy])?\.(w1|w2|w3)$")


def pad_mlp(module, down, up):
    m = TRUNK_MLP.match(module)
    if not m:
        return down, up
    if m.group(2) == "w2":
        return torch.nn.functional.pad(down, (0, -down.shape[1] % MLP_MULTIPLE)), up
    return down, torch.nn.functional.pad(up, (0, 0, 0, -up.shape[0] % MLP_MULTIPLE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("finetune")
    ap.add_argument("out")
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--min-dim", type=int, default=256, help="smaller linears are stored as exact diffs")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    out, kept_energy, total_energy, n_lora, n_diff = {}, 0.0, 0.0, 0, 0
    with safe_open(args.base, "pt") as fb, safe_open(args.finetune, "pt") as ff:
        base_keys, ft_keys = set(fb.keys()), set(ff.keys())
        if base_keys != ft_keys:
            raise SystemExit(f"key mismatch: {sorted(base_keys ^ ft_keys)[:5]} (fold the fine-tune first?)")
        for key in sorted(ft_keys):
            delta = ff.get_tensor(key).float() - fb.get_tensor(key).float()
            if not delta.abs().max() > 0:
                continue
            if key.endswith(".weight"):
                module = key[: -len(".weight")]
                if delta.ndim == 2 and min(delta.shape) >= args.min_dim:
                    d = delta.to(args.device)
                    u, s, vh = torch.linalg.svd(d, full_matrices=False)
                    r = min(args.rank, s.numel())
                    energy = float((s[:r] ** 2).sum() / (s ** 2).sum())
                    kept_energy += float((s[:r] ** 2).sum())
                    total_energy += float((s ** 2).sum())
                    up = (u[:, :r] * s[:r]).cpu()
                    down = vh[:r].cpu()
                    down, up = pad_mlp(module, down, up)
                    out[f"{PREFIX}{module}.lora_down.weight"] = down.to(dtype).contiguous()
                    out[f"{PREFIX}{module}.lora_up.weight"] = up.to(dtype).contiguous()
                    out[f"{PREFIX}{module}.alpha"] = torch.tensor(float(r))
                    n_lora += 1
                    if args.report:
                        rel = float(d.norm() / ff.get_tensor(key).float().norm())
                        print(f"  {module:44s} |dW|/|W| {rel:6.3f}  rank-{r} keeps {energy:6.1%}")
                    continue
                out[f"{PREFIX}{module}.diff"] = delta.to(dtype).contiguous()
            elif key.endswith(".bias") and f"{key[: -len('.bias')]}.weight" in ft_keys:
                out[f"{PREFIX}{key[: -len('.bias')]}.diff_b"] = delta.to(dtype).contiguous()
            else:
                out[f"{PREFIX}{key}.diff"] = delta.to(dtype).contiguous()
            n_diff += 1
    save_file(out, args.out, metadata={"iris_lora_rank": str(args.rank), "iris_lora_source": args.finetune.split("/")[-1]})
    size = sum(t.numel() * t.element_size() for t in out.values()) / 2**30
    print(f"wrote {args.out}: {n_lora} low-rank linears (rank {args.rank}, "
          f"{kept_energy / max(total_energy, 1e-30):.1%} of their difference), {n_diff} exact diffs, {size:.2f} GB")


if __name__ == "__main__":
    main()
