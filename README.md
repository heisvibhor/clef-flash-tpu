# clef-flash TPU int8

Source for [heisvibhor-1/clef-flash-tpu-int8](https://huggingface.co/heisvibhor-1/clef-flash-tpu-int8), an int8 JAX port of [Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash) that runs on one Google TPU v5e chip, including Colab's free v5e-1. It serves the Jev / SystemOne `/v1/systemone` API.

ARC-Challenge: 98.2% with int8 weights at 138 ms per request, and 98.0% with W8A8 at 102 ms. Cloudflare reports 98.3% for the BF16 model.

## Layout

| Path | Contents |
|---|---|
| `hub/clef_tpu.py` | JAX model: Qwen3.5 backbone, Cloudflare's joint schema head, compiled-program cache, `systemone()` entry point |
| `hub/notebook.ipynb` | Colab notebook that serves `/v1/systemone` behind a `cloudflared` tunnel |
| `hub/README.md` | Hugging Face model card |
| `notebooks/convert.ipynb` | Converts the BF16 checkpoint to int8 safetensors, checks the result, and uploads it |
| `notebooks/eval_arc_challenge.ipynb` | Runs ARC-Challenge through the Decision Index runner and writes the results into the model card |

The weights, tokenizer, `joint_schema_model.py`, and `compiled/` programs live only in the Hugging Face repo.

## Workflow

1. **Convert.** Run `notebooks/convert.ipynb` on Colab with about 35 GB of free disk. It downloads `Cloudflare/clef-flash`, writes int8 safetensors to `/content/clef-flash-tpu`, checks every int8 matrix against the original, and uploads the folder to Hugging Face.
2. **Publish the code.** From `law-ai`:

   ```sh
   hf upload heisvibhor-1/clef-flash-tpu-int8 clef-flash-tpu/hub .
   ```

3. **Compile.** On the target TPU, build programs for each 1,024-token prompt length in both activation modes, then upload the `compiled/<runtime id>/` folder:

   ```python
   for int8_activations in (False, True):
       clef_tpu.compile_backbones(model, 8192, int8_activations, save_dir=clef_tpu.compiled_backbones_dir())
   ```

4. **Evaluate.** Run `notebooks/eval_arc_challenge.ipynb` on a v5e-1 TPU. It needs an `HF_TOKEN` Colab secret with write access, since its last cell updates the `arc-challenge-results` section of the model card.
5. **Serve.** Open `hub/notebook.ipynb` in Colab with a v5e-1 TPU and run all cells. It prints the public `/v1/systemone` URL.
