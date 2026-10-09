"""The stock CLIPLoader "krea2" text encoder vs. Iris-3B's reference text encoding.

    CUDA_VISIBLE_DEVICES= <comfy>/.venv/bin/python tests/te_parity.py --comfy <ComfyUI> --te qwen3vl_4b_bf16.safetensors

The reference side reproduces iris3b.text.qwen3_vl.Qwen3VLTextEncoder._run with the transformers
Qwen3-VL language model loaded from the same file: system/user prefix fed through the decoder,
states sliced from the caption start, hidden_states at layers [2, 5, ..., 35].
"""

import argparse
import sys

import torch

PREFIX = ("<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial "
          "relationships of the objects and background:<|im_end|>\n<|im_start|>user\n")
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
LAYERS = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35]
PROMPTS = ["a red fox sleeping in fresh snow, golden hour", "", "A neon sign above a night-market stall that reads “OPEN LATE”"]


def reference(te_path, tokenizer_dir):
    from safetensors.torch import load_file
    from transformers import Qwen2Tokenizer, Qwen3VLTextConfig, Qwen3VLTextModel

    cfg = Qwen3VLTextConfig(vocab_size=151936, hidden_size=2560, intermediate_size=9728, num_hidden_layers=36,
                            num_attention_heads=32, num_key_value_heads=8, head_dim=128, rms_norm_eps=1e-6,
                            max_position_embeddings=262144,
                            rope_parameters={"rope_type": "default", "rope_theta": 5000000.0,
                                             "mrope_section": [24, 20, 20], "mrope_interleaved": True})
    model = Qwen3VLTextModel(cfg).eval()
    prefix = "model.language_model."
    sd = {k[len(prefix):]: v.float() for k, v in load_file(te_path).items() if k.startswith(prefix)}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected and not [m for m in missing if "rotary" not in m], (missing, unexpected)
    tok = Qwen2Tokenizer.from_pretrained(tokenizer_dir)
    pre = tok.encode(PREFIX, add_special_tokens=False)
    suf = tok.encode(SUFFIX, add_special_tokens=False)
    outs = []
    for prompt in PROMPTS:
        ids = pre + tok(prompt, add_special_tokens=False)["input_ids"] + suf
        with torch.no_grad():
            hs = model(input_ids=torch.tensor([ids]), output_hidden_states=True).hidden_states
        outs.append(torch.stack([hs[i][0, len(pre):] for i in LAYERS], dim=1))  # [T, 12, 2560]
    return outs


def comfy_side(te_path):
    import comfy.sd

    clip = comfy.sd.load_clip([te_path], clip_type=comfy.sd.CLIPType.KREA2)
    outs = []
    for prompt in PROMPTS:
        cond = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))[0]
        assert "attention_mask" not in cond[1] or cond[1]["attention_mask"].all()
        outs.append(cond[0][0].float().reshape(cond[0].shape[1], 12, 2560))
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", required=True)
    ap.add_argument("--te", required=True)
    args = ap.parse_args()
    sys.path.insert(0, args.comfy)
    sys.argv = sys.argv[:1] + ["--cpu"]
    import comfy.options

    comfy.options.enable_args_parsing()
    ref = reference(args.te, f"{args.comfy}/comfy/text_encoders/qwen25_tokenizer")
    ours = comfy_side(args.te)
    for prompt, r, o in zip(PROMPTS, ref, ours):
        assert r.shape == o.shape, (prompt, r.shape, o.shape)
        err = ((o - r).norm(dim=-1) / r.norm(dim=-1)).max().item()
        print(f"{r.shape[0]:3d} tokens  max per-token relerr {err:.2e}  {prompt!r}")
        assert err < 2e-2, err
    print("OK")


if __name__ == "__main__":
    main()
