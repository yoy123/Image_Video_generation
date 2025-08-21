# Local Image & Video Generation (Diffusers + Gradio)

This project is a minimal local scaffold for image-conditioned image and video generation using Hugging Face Diffusers, PyTorch and a small Gradio UI.

WARNING: This scaffold downloads and uses large Stable Diffusion-style models from Hugging Face. You must have the model weights locally or be logged in with `huggingface-cli login` and accept the model license.

Requirements
- Linux / macOS / Windows
- Python 3.10+
- A GPU with recent CUDA recommended. CPU mode works but is *very* slow.
- ffmpeg installed system-wide (used for video encoding).

Quick start

1. Create and activate a Conda environment

Option A — use the provided environment file (recommended):

```bash
conda env create -f environment.yml
conda activate ivg
```

Option B — create a minimal Conda env and install with pip:

```bash
conda create -n ivg python=3.10 -y
conda activate ivg
pip install --upgrade pip
pip install -r requirements.txt
```

2. Get model weights

- Pick a compatible img2img Stable Diffusion model on Hugging Face (for example `runwayml/stable-diffusion-v1-5` or other img2img-capable weights). Set the model id when running or modify the placeholder in the Gradio app.
- Optionally run `huggingface-cli login` so diffusers can download private/ gated weights.

3. Run the Gradio app

```bash
python app/gradio_app.py
```

Override the model without code changes by setting an environment variable before launching:

```bash
export MODEL_ID_OR_PATH="runwayml/stable-diffusion-v1-5"  # or path to a local .gguf/.safetensors
python app/gradio_app.py
```

To download a local model file (e.g., a .safetensors or .gguf) directly into the bundled WebUI models folder, use the helper:

```bash
python scripts/download_model.py "https://example.com/path/to/model.safetensors" --dest my-models
```

If you use a local .gguf model, generation is routed through the local AUTOMATIC1111 WebUI API at http://127.0.0.1:7860. You can point to a different URL with WEBUI_API_URL.

GPU and multi‑GPU

- Use your CUDA GPU(s) by starting with the GPU flag and optional device map:

	- CLI flags (from `app/gradio_app.py`): `--gpu` enables CUDA, `--cuda-devices 0,1` selects GPUs, `--device-map auto|balanced` enables accelerate sharding.
	- Or set environment variables:
		- `IVG_DEVICE_MAP=balanced` or `auto` to shard with accelerate.
		- `IVG_MAX_MEMORY="0=1GB,1=48GB"` to bias sharding across GPUs (keys are GPU indices, values are per‑GPU limits in GiB/GB).
		- `IVG_PRIMARY_CUDA=0` to pin the primary device index when not using device_map.
		- `IVG_FLUX_STAGED=1` to enable a staged Flux path optimized for dual GPUs.
		- `REDUX_REPO_ID=black-forest-labs/FLUX.1-Redux-dev` to load the Redux adapter.

- Tokenizer, feature_extractor, and image_encoder may run on CPU by design; no model CPU offload is used. No resizing is performed unless you request it.

Local checkpoint routing (no torch)

- If `MODEL_ID_OR_PATH` points to a local checkpoint file (`.safetensors`, `.ckpt`, `.gguf`), the app routes generation to the bundled AUTOMATIC1111 WebUI instead of Diffusers/PyTorch. This avoids importing torch for local‑file runs.
- Configure the WebUI endpoint with `WEBUI_API_URL` (defaults to `http://127.0.0.1:7860`).

Troubleshooting

- CUDA OOM: free up VRAM, reduce `num_inference_steps`, or set `IVG_DEVICE_MAP=balanced` and use `IVG_MAX_MEMORY` to bias work to the less‑loaded GPU.
- First run may download models from Hugging Face; ensure you’ve accepted licenses or have weights locally.

Files of interest
- `src/generate.py` — simple image->image wrapper using Diffusers' Img2Img pipeline.
- `src/video.py` — frame-by-frame image->video helper (generates frames then encodes with ffmpeg/imageio).
- `app/gradio_app.py` — small Gradio UI to upload a conditioning image, enter a prompt, and create images or short videos.
- `requirements.txt` — Python dependencies to install.

Notes & next steps
- This is a starter scaffold. Replace the placeholder `MODEL_ID` with your chosen model.
- For production or research workflows you should add caching, model config options, better safety filtering, error handling, rate limiting, and a manifest of tested model IDs.

License
- This scaffold contains no model weights. Follow the license of any model you download and respect usage restrictions.
