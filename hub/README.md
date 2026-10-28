---
license: apache-2.0
base_model: Cloudflare/clef-flash
base_model_relation: quantized
tags:
- clef
- clef-flash
- cloudflare
- decision-model
- systemone
- jax
- tpu
- tpu-v5e
- google-colab
- int8
- w8a8
- quantized
---

# clef-flash TPU int8

An int8 JAX port of [Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash) for a single **Google TPU v5e** chip, including the free **v5e-1 TPU** in Colab. It implements the Jev / SystemOne `/v1/systemone` request and response format.

- **ARC-Challenge:** 98.2% accuracy; Cloudflare reports 98.3% for the BF16 model.
- **Latency:** 138 ms per request with int8 weights and 102 ms with int8 weights and activations (W8A8).
- **TPU memory:** approximately 7 GB.
- **Startup:** precompiled TPU programs avoid first-request compilation on supported runtimes.

## Run on Colab

[`notebook.ipynb`](https://huggingface.co/heisvibhor-1/clef-flash-tpu-int8/blob/main/notebook.ipynb) starts an HTTP server for `/v1/systemone` and exposes it through a public `cloudflared` URL.

1. Open **Use this model > Google Colab**, or go to [huggingface.co/heisvibhor-1/clef-flash-tpu-int8/colab](https://huggingface.co/heisvibhor-1/clef-flash-tpu-int8/colab).
2. Set Runtime > Change runtime type to **v5e-1 TPU**.
3. Run all cells. The server cell prints the endpoint URL.

## Use from Python

TPU environments already include JAX. Install the remaining dependencies:

```sh
pip install "transformers>=5.10" safetensors
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

```python
import sys
from huggingface_hub import snapshot_download

repo = "heisvibhor-1/clef-flash-tpu-int8"
sys.path.insert(0, snapshot_download(repo, ignore_patterns=["compiled/*"]))
import clef_tpu

snapshot_download(repo, allow_patterns=[f"compiled/{clef_tpu.runtime_id()}/*"])
model = clef_tpu.load_model()

response = clef_tpu.systemone(model, {
    "model": "clef-flash",
    "state": "Our checkout started returning errors and orders are blocked.",
    "questions": {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle the message?",
            "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages"},
        },
        "outage": {"type": "noul", "instructions": "Is a service down?"},
    },
}, int8_activations=True)
print(response["answers"])
```

The request and response formats are documented under [Input format](https://huggingface.co/Cloudflare/clef-flash#input-format) and [Jev / SystemOne API](https://huggingface.co/Cloudflare/clef-flash#jev--systemone-api). Omit `int8_activations=True` to use BF16 activations with the int8 weights.

<!-- arc-challenge-results:start -->
## Results on ARC-Challenge

The benchmark contains 1,172 questions from [Decision Index](https://github.com/apolinario/decision-index), benchmark 27. Results use the same scoring method as Cloudflare's published evaluation.

| Model | Correct | Accuracy | Median latency |
|---|---:|---:|---:|
| Cloudflare clef-flash, BF16 (published) | ≈1,152 | 98.3% | |
| clef-flash TPU, int8 weights | 1,151 | 98.2% | 138 ms |
| clef-flash TPU, W8A8 | 1,149 | 98.0% | 102 ms |

The one-question difference from the published BF16 result is within the ±0.8-point sampling margin. W8A8 changes 9 of the 1,172 answers. Latency was measured one request at a time on one TPU v5e chip. Cloudflare's H200 latency is not directly comparable.

<details>
<summary>Versions and sources</summary>

jax 0.11.2, jaxlib 0.11.2, transformers 5.20.2, safetensors 0.8.0, torch 2.14.1+cpu, huggingface_hub 1.33.0. Rows built with [decision-index](https://github.com/apolinario/decision-index) @ `25c6ec7b6b5f` from `allenai/ai2_arc` @ `210d026faf99`; weights @ `408c77846fa6`.

</details>
<!-- arc-challenge-results:end -->

## Implementation

- Attention, linear-attention, and MLP weights use int8 values with one fp32 scale per output channel. The median SQNR across 200 matrices is 40 dB, or approximately 1% error.
- Norms, embeddings, the LM head, and the joint schema head are bit-identical to the source model.
- The vision encoder is not included.
- `clef_tpu.py` implements the Transformers Qwen3.5 model and Cloudflare `JointSchemaHead` in JAX. Layers of the same type are stacked for execution with `lax.scan`.

## Compiled TPU programs

Prompts are padded to multiples of 1,024 tokens. Each length requires a separate program, which takes approximately 22 seconds to compile. `compiled/<runtime id>/` contains programs for a specific TPU type and JAX/libtpu build. `load_model` selects the directory matching the current runtime.

Programs missing for the current runtime compile on first use. To compile all lengths up to 8,192 tokens:

```python
clef_tpu.compile_backbones(model, 8192, save_dir=...)
```

The compiled files are pickled XLA executables. Only load them from a trusted source.

## Files

| File | Contents |
|---|---|
| `*_attention.safetensors`, `mlp.safetensors`, `layer_norms.safetensors` | Backbone weights |
| `embeddings.safetensors`, `lm_head.safetensors` | Embeddings and LM head (BF16, kept in host memory) |
| `joint_head.safetensors`, `joint_schema_model.py` | Cloudflare's decision head and prompt builder, unchanged |
| `clef_tpu.py` | JAX model |
| `notebook.ipynb` | Colab server |
| `compiled/` | Compiled TPU programs, one folder per runtime |
| `conversion.json` | Source commit and tensor list |

## Limits

- Tested on TPU v5e-1 with prompts up to 6,820 tokens.
- Prompts longer than 16,384 tokens are truncated, matching Cloudflare's implementation.
- `transformers.AutoModel` cannot load this model; use `clef_tpu.py`.

## License

Apache-2.0. The same license applies to [Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash) and its base model, [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B). `LICENSE` is copied from the source repository. “Clef” and “Cloudflare” identify the source of the weights.
