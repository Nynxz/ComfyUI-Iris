# ComfyUI-Iris

Native ComfyUI support for **[Iris-3B](https://github.com/speridlabs/iris-3b)**, a 3B-parameter
text-to-image diffusion transformer that generates directly in **pixel space**: there is no VAE and
no latent space, so every pixel comes from the model.

<p align="center"><img src="example_workflows/iris-3b_text_to_image.jpg" width="420"></p>

- **No requirements.** The pack is pure ComfyUI and installs nothing.
- **No custom nodes.** It teaches ComfyUI's model loader to recognise Iris, so the standard
  Load Diffusion Model → KSampler graph runs it.
- **Works with ComfyUI features.** It uses the same building blocks as built-in models, so
  offloading, fp8, int8, and attention backends and options apply as usual.

## Install

Clone into `ComfyUI/custom_nodes` and restart ComfyUI:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Nynxz/ComfyUI-Iris
```

## Models

| File | Folder | Download |
|---|---|---|
| `iris-3b_bf16.safetensors` (6.0 GB) **or** `iris-3b_int8_convrot.safetensors` (3.4 GB) | `models/diffusion_models` | [nynxz/iris-3b](https://huggingface.co/nynxz/iris-3b) |
| `qwen3vl_4b_bf16.safetensors` (8.9 GB) **or** `qwen3vl_4b_fp8_scaled.safetensors` (5.2 GB) | `models/text_encoders` | [Comfy-Org/Qwen3-VL](https://huggingface.co/Comfy-Org/Qwen3-VL/tree/main/text_encoders) |

You can also load the original FP32 `model.safetensors` from
[speridlabs/iris-3b](https://huggingface.co/speridlabs/iris-3b) directly.

### bf16 or int8?

| | bf16 | int8 convrot |
|---|---|---|
| Size | 6.0 GB | 3.4 GB |
| Speed (RTX 5090, 1024², 50 steps, CFG 3) | 15.8 s | **9.5 s** |
| Quality | the original model | very close: same composition, ~30 dB PSNR vs bf16 |
| GPU | any | RTX 20-series or newer (int8 tensor cores) |

**bf16** keeps every weight in bfloat16, which is the precision ComfyUI runs the original model at anyway.
**int8 convrot** stores the transformer's large matrices as 8-bit integers. Each matrix is
first rotated (a Hadamard transform, "convrot") so outliers spread out and quantize cleanly. At
run time activations are also quantized to int8 and the multiplies run on int8 tensor cores. The
sensitive parts (text adapter, timestep and modulation layers, and the final pixel head)
stay in bf16. See [Making the weights](#making-the-weights).

## Workflow

Drag **[example_workflows/iris-3b_text_to_image.json](example_workflows/iris-3b_text_to_image.json)**
into ComfyUI, or open it from **Templates → Custom Nodes → ComfyUI-Iris**. It's the standard
text-to-image graph:

| Node | Setting |
|---|---|
| Load Diffusion Model | `iris-3b_bf16.safetensors` or `iris-3b_int8_convrot.safetensors` |
| Load CLIP | `qwen3vl_4b_bf16.safetensors` (or `_fp8_scaled`), type **`krea2`** |
| Load VAE | **`pixel_space`** (pixel-space model, so there is no real VAE) |
| ModelSamplingSD3 | shift **4** (the default; optional) |
| Empty Latent Image | about 1 megapixel, sides a multiple of 16 (e.g. 1024×1024, 1216×832) |
| CLIP Text Encode (negative) | **leave empty** |
| KSampler | `euler` / `simple`, 30–50 steps, CFG **3** |

**Tips**

- **Negative prompt:** keep it an empty string. Iris was trained with the encoded empty prompt as its
  guidance null. *Conditioning Zero Out* is not the same thing and gives worse results.
- **Text encoder:** CLIP type `krea2` is not a workaround. Krea 2 and Iris use the exact same
  Qwen3-VL-4B setup (system prompt and 12 hidden layers), so ComfyUI's built-in encoder is a perfect match.
- **Prompts:** Iris reads up to ~295 prompt tokens; longer prompts are truncated.
- **Exact reference sampler:** the official samples use 100 steps of DPM-Solver++(2M) at CFG 3.
  To reproduce that, use **SamplerCustom** + **SamplerDPMPP_2M_SDE** (`eta 0`, `midpoint`) +
  **BasicScheduler** (`simple`, 100 steps). Plain `dpmpp_2m` is a different formula for flow models.

## Making the weights

The files on [nynxz/iris-3b](https://huggingface.co/nynxz/iris-3b) were made from the official
FP32 checkpoint with the scripts in this repo:

```bash
# bf16: a straight cast, keys unchanged
python scripts/convert_bf16.py iris-3b.safetensors iris-3b_bf16.safetensors

# int8 convrot (quantizer: comfy-model-tools/quant_int8_convrot.py)
python scripts/pad_mlp.py iris-3b_bf16.safetensors iris-3b_bf16_mlp6912.safetensors
python quant_int8_convrot.py iris-3b_bf16_mlp6912.safetensors iris-3b_int8_convrot.safetensors
```

`pad_mlp.py` is required. Iris's MLP hidden size is 6826, which int8 GEMMs can't use (it must be a multiple of 4)
and which the convrot rotation can't divide. Without padding the quantizer still writes a file, but sampling
then fails with `INT8 GEMM not supported on this GPU for these dimensions`. The script zero-pads the
size to 6912, which leaves the output unchanged (the extra units always output 0). The quantizer then
converts 276 layers with a mean weight error of 0.85%.

## Verification

The port was checked against the reference implementation:

| Check | Result |
|---|---|
| Diffusion model vs `speridlabs/iris-3b` (released weights, fp32, 256×256) | 1.3e-5 relative error |
| `krea2` text encoder vs Iris's own Qwen3-VL encoding | ~2e-6 per token |
| Full graph through ComfyUI's own nodes, bf16 and int8 | images above |

```bash
CUDA_VISIBLE_DEVICES= python tests/parity.py --comfy <ComfyUI> --ref <iris-3b repo> [--checkpoint iris-3b.safetensors]
CUDA_VISIBLE_DEVICES= python tests/te_parity.py --comfy <ComfyUI> --te qwen3vl_4b_bf16.safetensors
python tests/e2e.py --comfy <ComfyUI> --out out --unet iris-3b_bf16.safetensors --clip qwen3vl_4b_bf16.safetensors
```

## Not yet supported

Iris-3B also ships fine-tunes for monocular depth and image restoration/upscaling. They are
not wired up here yet.

## Credits & license

Iris-3B by [Speridlabs](https://speridlabs.com/research/iris)
([paper](https://arxiv.org/abs/2610.09450), [code](https://github.com/speridlabs/iris-3b)). This
pack reimplements its model code on ComfyUI's building blocks. Apache-2.0, like the original; see
[`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
