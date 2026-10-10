"""Numerical parity of the ComfyUI port against the reference Iris-3B implementation.

    CUDA_VISIBLE_DEVICES= <comfy>/.venv/bin/python tests/parity.py --comfy <ComfyUI> --ref <iris-3b repo> \
        [--checkpoint iris-3b.safetensors]

Without --checkpoint it compares a small random-weight config. With it, it also runs the
released weights at 256x256 (fp32, CPU, ~25 GB RAM) and checks detection + key coverage.
"""

import argparse
import sys
import types

import torch


def setup_paths(comfy, ref):
    sys.path.insert(0, comfy)
    sys.path.insert(0, f"{ref}/src")
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
    sys.argv = sys.argv[:1] + ["--cpu"]
    import comfy.options

    comfy.options.enable_args_parsing()  # so comfy.cli_args picks up --cpu
    if "omegaconf" not in sys.modules:  # the reference config module only uses it inside functions
        sys.modules["omegaconf"] = types.SimpleNamespace(OmegaConf=None)


def reference_model(**overrides):
    from iris3b.config import ModelConfig, PixelStageConfig
    import iris3b.models.blocks  # noqa: F401  (registers block types)
    from iris3b.models.dit import IrisDiT

    pixel = overrides.pop("pixel", {})
    return IrisDiT(ModelConfig(pixel=PixelStageConfig(**pixel), **overrides)).eval()


def port_model(unet_config):
    import comfy.ops
    from iris.model import IrisDiT

    return IrisDiT(**unet_config, dtype=torch.float32, operations=comfy.ops.disable_weight_init).eval()


def compare(ref, port, height, width, text_len, n_layers, text_dim, real_tokens, seed=0):
    g = torch.Generator().manual_seed(seed)
    batch = 2
    x = torch.randn(batch, 3, height, width, generator=g)
    t = torch.tensor([937.5, 412.0])
    mask = torch.zeros(batch, text_len, dtype=torch.long)
    for row, n in enumerate(real_tokens):
        mask[row, :n] = 1
    y = torch.randn(batch, text_len, n_layers, text_dim, generator=g) * mask[:, :, None, None]
    with torch.no_grad():
        out_ref = ref(x, t, y, y_mask=mask).x
        # the port receives the TE layout: unpadded real tokens, layers flattened per token
        outs = []
        for row, n in enumerate(real_tokens):
            ctx = y[row:row + 1, :n].reshape(1, n, n_layers * text_dim)
            outs.append(port(x[row:row + 1], t[row:row + 1], context=ctx))
        out_port = torch.cat(outs)
    err = ((out_port - out_ref).norm() / out_ref.norm()).item()
    cos = torch.nn.functional.cosine_similarity(out_port.flatten(), out_ref.flatten(), dim=0).item()
    return err, cos


def random_config_test():
    torch.manual_seed(0)
    ref = reference_model(hidden_size=256, depth=4, dual_depth=2, num_heads=4, num_kv_heads=2, text_dim=64,
                          text_len=20, text_lap_num_layers=3, text_lap_num_heads=4,
                          pixel={"hidden_size": 8, "attn_hidden_size": 64, "num_heads": 2, "depth": 2})
    with torch.no_grad():
        for p in ref.parameters():  # adaLN-zero init would make most blocks identity
            p.normal_(0, 0.05)
    from iris.integration import detect_iris_config, pad_mlp

    sd = ref.state_dict()
    config = detect_iris_config(sd)
    sd = pad_mlp(dict(sd))
    config["text_layer_heads"] = 4  # the layer-attention head count is not visible in shapes
    port = port_model(config)
    missing, unexpected = port.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    err, cos = compare(ref, port, 64, 96, 20, 3, 64, real_tokens=(7, 20))
    print(f"random config 64x96: relerr {err:.2e}  cos {cos:.8f}")
    assert err < 1e-4, err


def edit_test(irisedit_repo):
    """The port with a reference image vs IrisEdit's training forward (frame-tagged refs)."""
    sys.path.insert(0, irisedit_repo)
    from irisedit import model as edit_model
    from iris.integration import detect_iris_config, pad_mlp

    torch.manual_seed(1)
    ref_model = reference_model(hidden_size=256, depth=4, dual_depth=2, num_heads=4, num_kv_heads=2, text_dim=64,
                                text_len=20, text_lap_num_layers=3, text_lap_num_heads=4, rope_frame_pairs=2,
                                pixel={"hidden_size": 8, "attn_hidden_size": 64, "num_heads": 2, "depth": 2})
    with torch.no_grad():
        for p in ref_model.parameters():
            p.normal_(0, 0.05)
    config = detect_iris_config(ref_model.state_dict())
    config["text_layer_heads"] = 4
    port = port_model(config)
    port.load_state_dict(pad_mlp(dict(ref_model.state_dict())))
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, 3, 64, 96, generator=g)
    ref = torch.rand(2, 3, 64, 96, generator=g) * 2 - 1
    t = torch.tensor([700.0, 250.0])
    mask = torch.zeros(2, 20, dtype=torch.long)
    mask[:, :9] = 1
    y = torch.randn(2, 20, 3, 64, generator=g) * mask[:, :, None, None]
    with torch.no_grad():
        out_ref = edit_model.forward(ref_model, x, [ref], t, y, mask)
        out_port = port(x, t, context=y[:, :9].reshape(2, 9, 3 * 64), ref_latents=[ref])
        no_ref = port(x, t, context=y[:, :9].reshape(2, 9, 3 * 64))
    err = ((out_port - out_ref).norm() / out_ref.norm()).item()
    moved = ((no_ref - out_ref).norm() / out_ref.norm()).item()
    print(f"edit (reference image) vs IrisEdit trainer: relerr {err:.2e}  (without the reference: {moved:.2e})")
    assert err < 1e-4, err


def checkpoint_test(path):
    from safetensors.torch import load_file
    from iris.integration import Iris, detect_iris_config
    import comfy.model_detection

    sd = load_file(path)
    config = comfy.model_detection.detect_unet_config(sd, "")
    print("detected:", config)
    assert config == detect_iris_config(sd)
    assert type(comfy.model_detection.model_config_from_unet_config(config, sd)) is Iris

    port = port_model(config)
    missing, unexpected = port.load_state_dict(Iris(config).process_unet_state_dict(dict(sd)), strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    ref = reference_model()
    ref.load_state_dict(sd, strict=True)
    del sd
    err, cos = compare(ref, port, 256, 256, 300, 12, 2560, real_tokens=(23, 5))
    print(f"released weights 256x256: relerr {err:.2e}  cos {cos:.8f}")
    assert err < 1e-3, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--checkpoint")
    ap.add_argument("--irisedit", help="IrisEdit repo: also check the edit path against its trainer")
    args = ap.parse_args()
    setup_paths(args.comfy, args.ref)
    from iris.integration import register

    register()
    random_config_test()
    if args.irisedit:
        edit_test(args.irisedit)
    if args.checkpoint:
        checkpoint_test(args.checkpoint)
    print("OK")


if __name__ == "__main__":
    main()
