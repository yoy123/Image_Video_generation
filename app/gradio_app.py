import os
import sys
# Ensure project root is on sys.path so `from src import ...` works when running this script directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import tempfile
import gradio as gr
from pathlib import Path
import socket

from src.generate import generate_image, MODEL_PATH
from src.video import image_to_video

# Replace this with your preferred model id (must support img2img)
# Tip: set env var MODEL_ID_OR_PATH to override without code changes.
MODEL_ID = MODEL_PATH

# Default device: prefer GPU only if CUDA_VISIBLE_DEVICES is set and non-empty.
DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"


def _list_local_models(models_root: Path | None = None) -> list[str]:
    """Return a list of candidate local model paths under ./models.

    Recognized as models:
    - Files: .safetensors, .ckpt, .gguf, .bin, .pth, .pt
    - Directories that contain any of the above files (at any depth), or model_index.json/config.json
    """
    mr = models_root or Path(__file__).resolve().parent.parent / "models"
    choices: list[str] = []
    exts = {".safetensors", ".ckpt", ".gguf", ".bin", ".pth", ".pt"}

    def _is_ip_adapter_path(p: Path) -> bool:
        name = str(p).lower()
        ip_keys = ["ip_adapter", "ip-adapter", "ipadapter"]
        if any(k in name for k in ip_keys):
            return True
        # If it's a directory, peek inside for typical IP-Adapter filenames
        if p.is_dir():
            try:
                for q in p.iterdir():
                    qn = q.name.lower()
                    if any(k in qn for k in ip_keys):
                        return True
                # Light recursive check for nested common filenames
                for q in p.rglob("*"):
                    if q.is_file():
                        qn = q.name.lower()
                        if any(k in qn for k in ip_keys):
                            return True
            except Exception:
                pass
        return False
    try:
        if mr.is_dir():
            for child in sorted(mr.iterdir()):
                if child.is_file():
                    if _is_ip_adapter_path(child):
                        continue
                    if child.suffix.lower() in exts:
                        choices.append(str(child))
                elif child.is_dir():
                    if _is_ip_adapter_path(child):
                        continue
                    indicative_files = (
                        "pytorch_model.safetensors",
                        "pytorch_model.bin",
                        "diffusion_pytorch_model.safetensors",
                        "diffusion_pytorch_model.bin",
                        "model.safetensors",
                        "model.ckpt",
                        "model.bin",
                        "model.pth",
                        "model.pt",
                        "model_index.json",
                        "config.json",
                    )
                    if any((child / fn).exists() for fn in indicative_files):
                        choices.append(str(child))
                        continue
                    # Fallback: search for any recognized weight file inside directory
                    found = False
                    try:
                        for p in child.rglob("*"):
                            if p.is_file() and p.suffix.lower() in exts:
                                if _is_ip_adapter_path(p):
                                    continue
                                choices.append(str(child))
                                found = True
                                break
                    except Exception:
                        pass
                    if found:
                        continue
    except Exception:
        # Silent fallback: empty list if anything goes wrong
        pass
    return choices


def _list_ip_adapters(models_root: Path | None = None) -> list[str]:
    """Return likely IP-Adapter paths under ./models (folders or files).

    Prefers models/IP_Adapter if present; matches names containing ip-adapter/ip_adapter/ipadapter
    and files with .safetensors/.bin/.pt/.pth extensions.
    """
    mr = models_root or Path(__file__).resolve().parent.parent / "models"
    keys = ["ip_adapter", "ip-adapter", "ipadapter"]
    exts = {".safetensors", ".bin", ".pt", ".pth"}
    out: list[str] = []
    try:
        if mr.is_dir():
            preferreds = [mr / "IP_Adapter", mr / "ip_adapter", mr / "IP-Adapter"]
            roots = [p for p in preferreds if p.exists()] or [mr]
            for base in roots:
                if not base.is_dir():
                    continue
                for child in sorted(base.iterdir()):
                    name = child.name.lower()
                    if any(k in name for k in keys):
                        out.append(str(child))
                        continue
                    if child.is_file() and child.suffix.lower() in exts and any(k in name for k in keys):
                        out.append(str(child))
                    elif child.is_dir():
                        try:
                            found = False
                            for p in child.rglob("*"):
                                if p.is_file():
                                    pn = p.name.lower()
                                    if (p.suffix.lower() in exts and any(k in pn for k in keys)) or any(k in pn for k in keys):
                                        out.append(str(child))
                                        found = True
                                        break
                            if found:
                                continue
                        except Exception:
                            pass
    except Exception:
        pass
    seen = set()
    dedup: list[str] = []
    for v in out:
        if v not in seen:
            seen.add(v)
            dedup.append(v)
    return dedup


def make_image(
    input_image,
    prompt,
    guidance_scale=7.5,
    steps: int = 30,
    strength: float = 0.75,
    seed: int | None = None,
    negative_prompt: str | None = None,
    scheduler_name: str | None = None,
    device_map_choice: str | None = None,
    max_memory_str: str | None = None,
    ip_adapter_repo: str | None = None,
    ip_adapter_scale: float | None = None,
    ip_subject_aware: bool = False,
    ip_bg_scale_factor: float = 0.5,
    ip_mask_softness: int = 32,
    model_id_override: str | None = None,
    redux_repo_override: str | None = None,
):
    temp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    input_path = temp.name
    input_image.save(input_path)
    out_path = str(Path(temp.name).with_suffix(".out.png"))
    model_to_use = (model_id_override or "").strip() or MODEL_ID
    # Configure optional Redux adapter via env var for this call.
    redux_repo = (redux_repo_override or "").strip()
    if redux_repo:
        os.environ["REDUX_REPO_ID"] = redux_repo
    else:
        os.environ.pop("REDUX_REPO_ID", None)
    # Advanced: configure device_map and max_memory per request
    if device_map_choice and device_map_choice.lower() in {"auto", "balanced"}:
        os.environ["IVG_DEVICE_MAP"] = device_map_choice.lower()
    else:
        os.environ.pop("IVG_DEVICE_MAP", None)
    if max_memory_str and str(max_memory_str).strip():
        os.environ["IVG_MAX_MEMORY"] = str(max_memory_str).strip()
    else:
        os.environ.pop("IVG_MAX_MEMORY", None)
    if ip_adapter_repo and str(ip_adapter_repo).strip():
        os.environ["IP_ADAPTER_REPO_ID"] = str(ip_adapter_repo).strip()
    else:
        os.environ.pop("IP_ADAPTER_REPO_ID", None)

    generate_image(
        input_image_path=input_path,
        prompt=prompt,
        output_path=out_path,
        model_id=model_to_use,
        device=DEVICE,
        guidance_scale=float(guidance_scale),
        num_inference_steps=int(steps),
        strength=float(strength),
        seed=(int(seed) if seed is not None and str(seed).strip() != "" else None),
        negative_prompt=(negative_prompt or None),
    scheduler_name=(scheduler_name or None),
    ip_adapter_scale=(float(ip_adapter_scale) if ip_adapter_scale is not None else None),
    ip_subject_aware=bool(ip_subject_aware),
    ip_bg_scale_factor=float(ip_bg_scale_factor),
    ip_mask_softness=int(ip_mask_softness),
    )
    return out_path


def make_video(
    input_image,
    prompt,
    num_frames=8,
    fps=8,
    seed: int | None = None,
    device_map_choice: str | None = None,
    max_memory_str: str | None = None,
    ip_adapter_repo: str | None = None,
    ip_adapter_scale: float | None = None,
    ip_subject_aware: bool = False,
    ip_bg_scale_factor: float = 0.5,
    ip_mask_softness: int = 32,
    model_id_override: str | None = None,
    redux_repo_override: str | None = None,
):
    temp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    input_path = temp.name
    input_image.save(input_path)
    out_video = str(Path(temp.name).with_suffix(".mp4"))
    model_to_use = (model_id_override or "").strip() or MODEL_ID
    redux_repo = (redux_repo_override or "").strip()
    if redux_repo:
        os.environ["REDUX_REPO_ID"] = redux_repo
    else:
        os.environ.pop("REDUX_REPO_ID", None)
    # Advanced: configure device_map and max_memory per request
    if device_map_choice and device_map_choice.lower() in {"auto", "balanced"}:
        os.environ["IVG_DEVICE_MAP"] = device_map_choice.lower()
    else:
        os.environ.pop("IVG_DEVICE_MAP", None)
    if max_memory_str and str(max_memory_str).strip():
        os.environ["IVG_MAX_MEMORY"] = str(max_memory_str).strip()
    else:
        os.environ.pop("IVG_MAX_MEMORY", None)
    if ip_adapter_repo and str(ip_adapter_repo).strip():
        os.environ["IP_ADAPTER_REPO_ID"] = str(ip_adapter_repo).strip()
    else:
        os.environ.pop("IP_ADAPTER_REPO_ID", None)

    image_to_video(
        input_image_path=input_path,
        prompt=prompt,
        model_id=model_to_use,
        out_path=out_video,
        num_frames=int(num_frames),
        fps=int(fps),
        device=DEVICE,
    seed=(int(seed) if seed is not None and str(seed).strip() != "" else None),
    )
    return out_video


def launch():
    def _find_free_port(start=7861, max_tries=200):
        for p in range(start, start + max_tries):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    s.bind(("127.0.0.1", p))
                    return p
                except OSError:
                    continue
        return 0

    # Ensure dropdown panels are scrollable if there are many items
    _css = """
    /* Global scroll for dropdown option lists (portals or in-container) */
    body [role='listbox'], body ul[role='listbox'] { max-height: 320px !important; overflow-y: auto !important; }
    .gradio-container [role='listbox'], .gradio-container ul[role='listbox'] { max-height: 320px !important; overflow-y: auto !important; }
    /* Specific hooks for our dropdowns */
    #local_models_dd [role='listbox'], #ip_adapter_dd [role='listbox'] { max-height: 320px !important; overflow-y: auto !important; }
    """
    with gr.Blocks(css=_css) as demo:
        gr.Markdown("# Local Image → Image & Image → Video (starter)")
        with gr.Row():
            inp = gr.Image(type="pil", label="Conditioning image")
            with gr.Column():
                prompt = gr.Textbox(lines=3, placeholder="A cinematic portrait of ...", label="Prompt")
                negative = gr.Textbox(lines=2, placeholder="Low quality, blurry, ...", label="Negative Prompt")
                with gr.Row():
                    guidance = gr.Slider(minimum=1, maximum=15, step=0.1, value=7.5, label="CFG (Guidance)")
                    steps = gr.Slider(minimum=1, maximum=75, step=1, value=30, label="Steps")
                    strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.75, label="Denoise Strength")
                with gr.Row():
                    scheduler = gr.Dropdown(
                        choices=["Euler a", "Euler", "DDIM", "DPM++ 2M", "Heun", "LMS", "DEIS", "UniPC"],
                        value="Euler a",
                        label="Scheduler",
                    )
                    seed = gr.Number(value=42, label="Seed", precision=0)
                default_model = os.environ.get("MODEL_ID_OR_PATH", "SicariusSicariiStuff/flux.1dev-abliteratedv2") or str(MODEL_ID)
                default_redux = os.environ.get("REDUX_REPO_ID", "black-forest-labs/FLUX.1-Redux-dev")
                # Local models dropdown sourced from ./models
                local_model_choices = _list_local_models()
                # Preselect if default model points to a local entry
                local_preselect = default_model if isinstance(default_model, str) and default_model in local_model_choices else None
                with gr.Row():
                    local_models_dd = gr.Dropdown(
                        choices=local_model_choices or ["— No local models found —"],
                        value=local_preselect,
                        label="Local Models (./models)",
                        info="Pick a local model to populate the field below",
                        scale=4,
                        elem_id="local_models_dd",
                    )
                    refresh_local_btn = gr.Button("Refresh", scale=1)
                local_models_info = gr.Markdown(f"Found {len(local_model_choices)} local model(s) under ./models")
                model_box = gr.Textbox(value=default_model, label="Model ID or Path", info="Hugging Face model id or local .safetensors/.ckpt/.gguf path")
                redux_box = gr.Textbox(value=default_redux, label="Redux Adapter Repo (optional)", info="e.g., black-forest-labs/FLUX.1-Redux-dev")
                generate_btn = gr.Button("Generate Image")
                generate_vid_btn = gr.Button("Generate Video")
                frames = gr.Number(value=8, label="Frames (video)")
                fps = gr.Number(value=8, label="FPS")
        
        # Simple helper to write dropdown selection into the model textbox
        def _use_local_model(selected: str | None):
            return selected or ""

        if 'local_models_dd' in locals():
            local_models_dd.change(fn=_use_local_model, inputs=local_models_dd, outputs=model_box)
            
            def _refresh_local_models():
                choices = _list_local_models()
                info_md = f"Found {len(choices)} local model(s) under ./models"
                # Return updates for dropdown and info text; clear selection
                return gr.update(choices=choices or ["— No local models found —"], value=None), info_md

            refresh_local_btn.click(fn=_refresh_local_models, inputs=None, outputs=[local_models_dd, local_models_info])
        with gr.Accordion("Advanced (GPU & Reference)", open=False):
            with gr.Row():
                device_map_choice = gr.Dropdown(choices=["none", "auto", "balanced"], value="balanced", label="Device Map (accelerate)")
                max_memory = gr.Textbox(value="0=1GB,1=48GB", label="Max Memory (per GPU)", info="e.g., 0=1GB,1=48GB")
            with gr.Row():
                ip_adapter_repo = gr.Textbox(value="", label="IP-Adapter Repo (optional)", info="HF repo id, local folder, or file path")
                ip_adapter_scale = gr.Slider(minimum=0.0, maximum=1.5, step=0.05, value=0.7, label="IP-Adapter Scale")
            # IP-Adapter helper UI
            ip_choices = _list_ip_adapters()
            with gr.Row():
                ip_adapter_dd = gr.Dropdown(
                    choices=ip_choices or ["— No IP-Adapter paths found —"],
                    value=None,
                    label="IP-Adapter Paths (./models)",
                    info="Pick to fill the field above",
                    scale=4,
                    elem_id="ip_adapter_dd",
                )
                ip_refresh_btn = gr.Button("Refresh", scale=1)
            ip_info = gr.Markdown(f"Found {len(ip_choices)} IP-Adapter path(s) under ./models")
            with gr.Row():
                ip_subject_aware = gr.Checkbox(value=False, label="Subject-aware blending (faces/center)")
                ip_bg_scale_factor = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.5, label="Background Scale Factor")
                ip_mask_softness = gr.Slider(minimum=0, maximum=64, step=1, value=32, label="Mask Softness")
        out_image = gr.Image(label="Output Image")
        out_video = gr.Video(label="Output Video")
        # Wire events inside Blocks context
        generate_btn.click(
            fn=make_image,
            inputs=[inp, prompt, guidance, steps, strength, seed, negative, scheduler, device_map_choice, max_memory, ip_adapter_repo, ip_adapter_scale, ip_subject_aware, ip_bg_scale_factor, ip_mask_softness, model_box, redux_box],
            outputs=out_image,
        )
        generate_vid_btn.click(
            fn=make_video,
            inputs=[inp, prompt, frames, fps, seed, device_map_choice, max_memory, ip_adapter_repo, ip_adapter_scale, ip_subject_aware, ip_bg_scale_factor, ip_mask_softness, model_box, redux_box],
            outputs=out_video,
        )

        # Wire IP-Adapter helpers
        def _use_ip_path(selected: str | None):
            return selected or ""

        if 'ip_adapter_dd' in locals():
            ip_adapter_dd.change(fn=_use_ip_path, inputs=ip_adapter_dd, outputs=ip_adapter_repo)

            def _refresh_ip_paths():
                choices = _list_ip_adapters()
                info_md = f"Found {len(choices)} IP-Adapter path(s) under ./models"
                return gr.update(choices=choices or ["— No IP-Adapter paths found —"], value=None), info_md

            ip_refresh_btn.click(fn=_refresh_ip_paths, inputs=None, outputs=[ip_adapter_dd, ip_info])

    # Use a port that doesn't collide with AUTOMATIC1111 default (7860); auto-pick if needed
    requested = os.environ.get("GRADIO_PORT")
    port = int(requested) if requested and requested.isdigit() else _find_free_port()
    if port == 0:
        # as a last resort let Gradio try its default range
        demo.launch(share=False)
    else:
        demo.launch(share=False, server_port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true", help="Force using GPU if available (requires CUDA-enabled PyTorch).")
    parser.add_argument("--cuda-devices", type=str, default=None, help="Set CUDA_VISIBLE_DEVICES (e.g. '0' or '0,1').")
    parser.add_argument("--device-map", type=str, default=None, choices=["auto", "balanced", "none"], help="Enable accelerate device_map sharding across GPUs ('auto' or 'balanced'). Use 'none' to disable.")
    args = parser.parse_args()

    # Optionally override visible CUDA devices
    if args.cuda_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    # Determine DEVICE: prefer GPU if requested and available
    if args.gpu:
        try:
            import torch as _torch

            if _torch.cuda.is_available():
                DEVICE = "cuda"
                # Configure device_map if requested
                if args.device_map and args.device_map in {"auto", "balanced"}:
                    os.environ["IVG_DEVICE_MAP"] = args.device_map
                elif args.device_map == "none":
                    os.environ.pop("IVG_DEVICE_MAP", None)
            else:
                print("--gpu requested but CUDA not available to PyTorch; falling back to CPU")
                DEVICE = "cpu"
        except Exception:
            print("Failed to probe torch for CUDA; falling back to CPU")
            DEVICE = "cpu"

    launch()
