from typing import Any
from PIL import Image, ImageEnhance
from PIL.PngImagePlugin import PngInfo
import json
import math
from .utils import load_image

import os
import base64
import requests
from io import BytesIO
import random

# Default model can be overridden with env var MODEL_ID_OR_PATH.
# Supports either a Hugging Face model id (e.g. "runwayml/stable-diffusion-v1-5")
# or a local file path (e.g. a .gguf or .safetensors file). When a local .gguf is used,
# generation is routed through a running AUTOMATIC1111 WebUI at 127.0.0.1:7860.
MODEL_PATH = os.environ.get(
    "MODEL_ID_OR_PATH",
    "/home/dan/Downloads/stable-diffusion-xl-refiner-1.0-Q8_0.gguf",
)


def _diffusers_available() -> bool:
    """Return True if torch and diffusers can be imported."""
    try:
        import torch  # noqa: F401
        import diffusers  # noqa: F401
        return True
    except Exception:
        return False


def _fallback_generate(
    *,
    input_image_path: str,
    prompt: str,
    output_path: str,
    model_id: str,
    num_inference_steps: int,
    guidance_scale: float,
    strength: float,
    seed: int | None,
    negative_prompt: str | None,
    scheduler_name: str | None,
    ip_adapter_scale: float | None,
    embed_metadata: bool,
    save_metadata_json: bool,
) -> str:
    """Simple, dependency-free fallback image generator.

    If the heavy diffusion pipeline is unavailable (e.g. during tests or
    environments without model weights) we still want to exercise the rest of
    the pipeline.  This function performs a deterministic brightness tweak based
    on the seed and writes metadata similar to the real generator.
    """

    img = load_image(input_image_path)
    # Deterministic yet visible change to the image based on the seed.
    factor = 1.0
    if seed is not None:
        random.seed(int(seed))
        factor = 0.8 + (random.random() * 0.4)  # between 0.8 and 1.2
    img = ImageEnhance.Brightness(img).enhance(factor)

    meta = {
        "prompt": prompt,
        "negative_prompt": negative_prompt or "",
        "seed": int(seed) if seed is not None else None,
        "steps": int(num_inference_steps),
        "guidance_scale": float(guidance_scale),
        "strength": float(strength),
        "scheduler": scheduler_name,
        "model_id": model_id,
        "redux_repo": os.environ.get("REDUX_REPO_ID"),
        "ip_adapter_repo": os.environ.get("IP_ADAPTER_REPO_ID"),
        "ip_adapter_scale": float(ip_adapter_scale) if ip_adapter_scale is not None else None,
        "device_map": os.environ.get("IVG_DEVICE_MAP"),
    }

    if embed_metadata:
        try:
            info = PngInfo()
            info.add_text("ivg:meta", json.dumps(meta, ensure_ascii=False))
            img.save(output_path, pnginfo=info)
        except Exception:
            img.save(output_path)
    else:
        img.save(output_path)

    if save_metadata_json:
        try:
            sidecar = os.path.splitext(output_path)[0] + ".json"
            with open(sidecar, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    return output_path


def _webui_img2img_call(input_image_path: str,
                       prompt: str,
                       output_path: str,
                       steps: int = 30,
                       cfg_scale: float = 7.5,
                       denoising_strength: float = 0.75,
                       seed: int | None = None,
                       negative_prompt: str | None = None,
                       sampler_name: str | None = None,
                       api_url: str = os.environ.get("WEBUI_API_URL", "http://127.0.0.1:7860"),
                       model_override: str | None = None) -> str:
    """Call local AUTOMATIC1111 WebUI API /sdapi/v1/img2img and write first returned image to output_path."""
    with open(input_image_path, "rb") as f:
        img_bytes = f.read()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    init_image_data = f"data:image/png;base64,{img_b64}"

    payload = {
        "init_images": [init_image_data],
        "prompt": prompt,
        "steps": int(steps),
        "cfg_scale": float(cfg_scale),
        "denoising_strength": float(denoising_strength),
        "sampler_name": (sampler_name or "Euler a"),
    }
    if seed is not None:
        payload["seed"] = int(seed)
    if negative_prompt:
        payload["negative_prompt"] = negative_prompt
    # If provided, request WebUI to switch to this checkpoint by name/path.
    if model_override:
        payload["override_settings"] = {"sd_model_checkpoint": model_override}

    url = api_url.rstrip('/') + "/sdapi/v1/img2img"
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    images_b64 = data.get("images")
    if not images_b64:
        raise RuntimeError("WebUI API returned no images")
    img0 = images_b64[0]
    img_bytes = base64.b64decode(img0)
    out = Image.open(BytesIO(img_bytes)).convert("RGB")
    out.save(output_path)
    return output_path


def get_pipeline(model_id: str, device: str = "cuda") -> Any:
    # Lazy import torch to avoid import-time failures for UI-only flows
    import torch
    want_float16 = device.startswith("cuda")
    torch_dtype = torch.float16 if want_float16 else torch.float32
    # Resolve HF cache directories (models--repo) to a usable snapshot path or nested folder with model_index.json
    def _resolve_model_path(mid: str) -> str:
        try:
            if os.path.isdir(mid):
                if os.path.isfile(os.path.join(mid, "model_index.json")):
                    return mid
                snaps_dir = os.path.join(mid, "snapshots")
                if os.path.isdir(snaps_dir):
                    snaps = sorted(os.listdir(snaps_dir), reverse=True)
                    for s in snaps:
                        cand = os.path.join(snaps_dir, s)
                        if os.path.isfile(os.path.join(cand, "model_index.json")):
                            return cand
                    # fallback to newest snapshot even if model_index.json missing
                    if snaps:
                        return os.path.join(snaps_dir, snaps[0])
                # Deep search for model_index.json in children
                for root, dirs, files in os.walk(mid):
                    if "model_index.json" in files:
                        return root
        except Exception:
            pass
        return mid

    model_id = _resolve_model_path(model_id)

    pipe = None
    auto_err: Exception | None = None

    # device_map handling
    use_device_map = None
    try:
        dm = os.environ.get("IVG_DEVICE_MAP", "").strip().lower()
        if dm in {"balanced", "auto"}:
            use_device_map = dm
    except Exception:
        use_device_map = None

    # Coerce 'auto' to 'balanced' for Flux upfront
    try:
        if use_device_map == "auto" and ("flux" in str(model_id).lower()):
            use_device_map = "balanced"
    except Exception:
        pass

    # Predeclare for reuse in retry paths
    max_mem = None
    multi_gpu = False
    if device.startswith("cuda"):
        try:
            multi_gpu = torch.cuda.is_available() and torch.cuda.device_count() > 1
        except Exception:
            multi_gpu = False

    try:
        from diffusers import AutoPipelineForImage2Image
        if use_device_map and multi_gpu:
            # Compute optional max_memory bias
            try:
                env_mm = os.environ.get("IVG_MAX_MEMORY", "").strip()
                def _norm_unit(val: str) -> str:
                    v = val.strip()
                    return v.replace("GB", "GiB").replace("gb", "GiB").replace("GiB", "GiB")
                def _parse_key(k: str):
                    ks = k.strip().lower()
                    if ks.startswith(("cuda:", "gpu")):
                        ks = ks.split(":",1)[-1].replace("gpu","")
                    return int(ks) if ks.isdigit() else ks
                if env_mm:
                    mm = {}
                    for part in env_mm.split(","):
                        if "=" in part:
                            k, v = part.split("=", 1)
                            mm[_parse_key(k)] = _norm_unit(v)
                    if mm:
                        max_mem = mm
                else:
                    try:
                        free = [(torch.cuda.mem_get_info(i)[0], i) for i in range(torch.cuda.device_count())]
                        free.sort(reverse=True)
                        large = free[0][1]; small = free[-1][1]
                        max_mem = {small: "1GiB", large: "48GiB"}
                    except Exception:
                        pass
            except Exception:
                max_mem = None

            kwargs = {
                "device_map": use_device_map,
                "torch_dtype": (torch.float16 if device.startswith("cuda") else None),
                "trust_remote_code": True,
            }
            if max_mem:
                kwargs["max_memory"] = max_mem
            try:
                pipe = AutoPipelineForImage2Image.from_pretrained(model_id, **kwargs)
            except Exception as _e_dm:
                # Fallback: force balanced if auto was passed or balanced still failed due to support
                if "auto not supported" in str(_e_dm).lower():
                    kwargs["device_map"] = "balanced"
                    pipe = AutoPipelineForImage2Image.from_pretrained(model_id, **kwargs)
                else:
                    raise
        else:
            pipe = AutoPipelineForImage2Image.from_pretrained(model_id)
    except Exception as e1:
        auto_err = e1
        try:
            from diffusers import AutoPipelineForImage2Image
            if use_device_map and multi_gpu:
                kwargs = {
                    "device_map": use_device_map,
                    "torch_dtype": (torch.float16 if device.startswith("cuda") else None),
                    "trust_remote_code": True,
                }
                if max_mem:
                    kwargs["max_memory"] = max_mem
                try:
                    pipe = AutoPipelineForImage2Image.from_pretrained(model_id, **kwargs)
                except Exception as _e_dm2:
                    if "auto not supported" in str(_e_dm2).lower():
                        kwargs["device_map"] = "balanced"
                        pipe = AutoPipelineForImage2Image.from_pretrained(model_id, **kwargs)
                    else:
                        raise
            else:
                pipe = AutoPipelineForImage2Image.from_pretrained(model_id, trust_remote_code=True)
        except Exception as e2:
            auto_err = e2

    if pipe is None:
        # Only fall back to SD Img2Img for classic SD repos; otherwise raise a helpful error
        if any(x in model_id.lower() for x in ["stable-diffusion", "sd-", "runwayml/"]):
            from diffusers import StableDiffusionImg2ImgPipeline
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id)
        else:
            raise RuntimeError(
                f"Failed to load AutoPipelineForImage2Image for '{model_id}': {auto_err}. "
                "This repo may require a specific pipeline or trust_remote_code."
            )

    # Move to device and set dtype if appropriate. When using CUDA prefer
    # fp16 and enable memory-saving helpers on diffusers pipelines.
    if device.startswith("cuda"):
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,expandable_segments:True")
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
        # Try to enable VAE slicing/tiling if supported
        try:
            if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
                pipe.vae.enable_slicing()
        except Exception:
            pass
        try:
            if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
                pipe.vae.enable_tiling()
        except Exception:
            pass

        n_gpus = torch.cuda.device_count()
        if use_device_map and n_gpus > 1:
            # Using accelerate's device_map; skip manual placement and do not force a single pipe.device
            pass
        elif n_gpus > 1:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            # Determine primary GPU: honor explicit device like 'cuda:1' or env override; else pick max free
            explicit_idx = None
            try:
                if ":" in device:
                    explicit_idx = int(device.split(":", 1)[1])
            except Exception:
                explicit_idx = None
            if explicit_idx is None:
                try:
                    env_idx = os.environ.get("IVG_PRIMARY_CUDA") or os.environ.get("PRIMARY_CUDA")
                    if env_idx is not None:
                        explicit_idx = int(env_idx)
                except Exception:
                    explicit_idx = None

            try:
                free_bytes = []
                for i in range(n_gpus):
                    free, total = torch.cuda.mem_get_info(i)
                    free_bytes.append((free, i))
                free_bytes.sort(reverse=True)
                if explicit_idx is not None and 0 <= explicit_idx < n_gpus:
                    primary_idx = explicit_idx
                    # choose best remaining as secondary
                    secondary_idx = next((i for (f,i) in free_bytes if i != primary_idx), (primary_idx + 1) % n_gpus)
                else:
                    primary_idx = free_bytes[0][1]
                    secondary_idx = free_bytes[1][1] if n_gpus > 1 else (0 if primary_idx != 0 else 1)
            except Exception:
                primary_idx, secondary_idx = 0, 1 if n_gpus > 1 else 0
            primary = f"cuda:{primary_idx}"
            secondary = f"cuda:{secondary_idx}"
            # Heavier modules on primary
            for name in ("unet",):
                if hasattr(pipe, name):
                    mod = getattr(pipe, name)
                    if isinstance(mod, torch.nn.Module):
                        try:
                            mod.to(primary, dtype=torch.float16)
                        except Exception:
                            pass
            # Ensure transformer resides entirely on primary to avoid cross-device ops
            try:
                if hasattr(pipe, "transformer"):
                    pipe.transformer.to(primary, dtype=torch.float16)
            except Exception:
                pass
            # Text encoders on secondary
            for name in ("text_encoder", "text_encoder_2"):
                if hasattr(pipe, name):
                    mod = getattr(pipe, name)
                    if isinstance(mod, torch.nn.Module):
                        try:
                            mod.to(secondary, dtype=torch.float16)
                        except Exception:
                            pass
            # Tokenizers and feature extractors remain on CPU by default
            # Move image_encoder to CPU to save VRAM; wrap its forward to move outputs back to primary
            try:
                if hasattr(pipe, "image_encoder") and isinstance(pipe.image_encoder, torch.nn.Module):
                    try:
                        pipe.image_encoder.to("cpu")  # keep as float32 on CPU
                    except Exception:
                        pass
                    try:
                        _orig_ie_forward = pipe.image_encoder.forward
                        def _wrapped_ie_forward(*a, **kw):
                            out = _orig_ie_forward(*a, **kw)
                            def _to_primary(x):
                                try:
                                    if hasattr(x, 'to'):
                                        return x.to(primary)
                                except Exception:
                                    return x
                                return x
                            if isinstance(out, tuple):
                                return tuple(_to_primary(o) for o in out)
                            if isinstance(out, dict):
                                return {k: _to_primary(v) for k, v in out.items()}
                            # Try common HF output with attributes
                            try:
                                if hasattr(out, 'last_hidden_state'):
                                    out.last_hidden_state = _to_primary(out.last_hidden_state)
                                if hasattr(out, 'pooler_output'):
                                    out.pooler_output = _to_primary(out.pooler_output)
                            except Exception:
                                pass
                            return _to_primary(out)
                        pipe.image_encoder.forward = _wrapped_ie_forward
                    except Exception:
                        pass
            except Exception:
                pass
            # Set base device for internals
            try:
                pipe.__dict__["device"] = primary
            except Exception:
                pass

            # Move VAE to secondary to split memory load
            try:
                if hasattr(pipe, "vae") and isinstance(pipe.vae, torch.nn.Module):
                    pipe.vae.to(secondary, dtype=torch.float16)
            except Exception:
                pass

            # Ensure VAE encode runs on secondary and returns latents to primary
            try:
                if hasattr(pipe, "_encode_vae_image") and hasattr(pipe, "vae"):
                    orig_encode = pipe._encode_vae_image
                    def _wrapped_encode_vae_image(image, *a, **kw):
                        vae_dev = getattr(pipe.vae, "device", None) or secondary
                        try:
                            image = image.to(vae_dev)
                        except Exception:
                            pass
                        out = orig_encode(image, *a, **kw)
                        # Move encoded latents back to primary for UNet
                        try:
                            if isinstance(out, tuple):
                                out = tuple(o.to(primary) if hasattr(o, 'to') else o for o in out)
                            elif hasattr(out, 'to'):
                                out = out.to(primary)
                        except Exception:
                            pass
                        return out
                    pipe._encode_vae_image = _wrapped_encode_vae_image
            except Exception:
                pass

            # Ensure T5 embeds computed on secondary then moved to primary
            try:
                if hasattr(pipe, "_get_t5_prompt_embeds") and hasattr(pipe, "text_encoder_2"):
                    orig_get_t5 = pipe._get_t5_prompt_embeds
                    def _wrapped_get_t5(prompt, *a, **kw):
                        tokenizer = getattr(pipe, "tokenizer_2", None) or getattr(pipe, "tokenizer", None)
                        if tokenizer is None:
                            return orig_get_t5(prompt, *a, **kw)
                        enc = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=getattr(tokenizer, "model_max_length", 512))
                        te2_dev = getattr(pipe.text_encoder_2, "device", secondary)
                        input_ids = enc["input_ids"].to(te2_dev)
                        attention_mask = enc.get("attention_mask")
                        if attention_mask is not None:
                            attention_mask = attention_mask.to(te2_dev)
                        try:
                            outputs = pipe.text_encoder_2(input_ids, attention_mask=attention_mask, output_hidden_states=False)
                            embeds = outputs[0]
                            try:
                                embeds = embeds.to(primary)
                            except Exception:
                                pass
                            return embeds
                        except Exception:
                            return orig_get_t5(prompt, *a, **kw)
                    pipe._get_t5_prompt_embeds = _wrapped_get_t5
            except Exception:
                pass

            # Ensure generic prompt encode outputs end up on primary
            try:
                if hasattr(pipe, "_encode_prompt"):
                    _orig_encode_prompt = pipe._encode_prompt
                    def _wrapped_encode_prompt(*a, **kw):
                        out = _orig_encode_prompt(*a, **kw)
                        try:
                            if isinstance(out, tuple):
                                out = tuple(o.to(primary) if hasattr(o, 'to') else o for o in out)
                            elif hasattr(out, 'to'):
                                out = out.to(primary)
                        except Exception:
                            pass
                        return out
                    pipe._encode_prompt = _wrapped_encode_prompt
            except Exception:
                pass

            # No prepare_latents wrapper needed when UNet and VAE share device
        else:
            # Single GPU
            try:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                pipe = pipe.to(device, dtype=torch.float16)
            except Exception:
                pipe = pipe.to(device)
    else:
        # CPU or other device
        try:
            pipe = pipe.to(device, dtype=torch_dtype)
        except Exception:
            pipe = pipe.to(device)

    # Optionally try to load a Redux/img2img adapter (e.g., Flux Redux) if provided.
    redux_repo = os.environ.get("REDUX_REPO_ID")
    if redux_repo:
        try:
            # Only attempt Redux on Flux-family pipelines (its weights are tailored for Flux)
            try:
                _is_flux = ("flux" in str(model_id).lower()) or ("flux" in pipe.__class__.__name__.lower())
            except Exception:
                _is_flux = False
            if not _is_flux:
                raise RuntimeError("Redux adapter skipped: current model is not a Flux pipeline")
            # Resolve common cases: a) a local file/dir, b) a HF cache snapshot for the repo
            resolved = None
            resolved_dir = None
            resolved_weight_name = None
            # If user passed a local path (file or dir), prefer that
            if os.path.exists(redux_repo):
                if os.path.isdir(redux_repo):
                    # look for a safetensors file inside the dir
                    import glob

                    matches = glob.glob(os.path.join(redux_repo, "**", "*.safetensors"), recursive=True)
                    if matches:
                        matches.sort()
                        # Prefer names that contain 'redux' if multiple
                        preferred = [m for m in matches if 'redux' in os.path.basename(m).lower()]
                        # Prefer common flux redux weight filenames if present
                        more_pref = [m for m in matches if os.path.basename(m).lower() in {
                            'flux1-redux-dev.safetensors', 'pytorch_lora_weights.safetensors'
                        }]
                        if more_pref:
                            preferred = more_pref
                        pick = preferred[0] if preferred else matches[0]
                        resolved_dir = os.path.dirname(pick)
                        resolved_weight_name = os.path.basename(pick)
                else:
                    resolved = redux_repo

            # If not a local path, check HF cache for a matching snapshot
            if resolved is None:
                cache_base = os.path.expanduser(os.path.join("~", ".cache", "huggingface", "hub"))
                repo_dir = f"models--{redux_repo.replace('/', '--')}"
                snapshots_dir = os.path.join(cache_base, repo_dir, "snapshots")
                if os.path.isdir(snapshots_dir):
                    # pick the newest snapshot directory
                    snaps = sorted(os.listdir(snapshots_dir), reverse=True)
                    for s in snaps:
                        snap_path = os.path.join(snapshots_dir, s)
                        import glob

                        matches = glob.glob(os.path.join(snap_path, "**", "*.safetensors"), recursive=True)
                        if matches:
                            matches.sort()
                            preferred = [m for m in matches if 'redux' in os.path.basename(m).lower()]
                            more_pref = [m for m in matches if os.path.basename(m).lower() in {
                                'flux1-redux-dev.safetensors', 'pytorch_lora_weights.safetensors'
                            }]
                            if more_pref:
                                preferred = more_pref
                            pick = preferred[0] if preferred else matches[0]
                            resolved_dir = os.path.dirname(pick)
                            resolved_weight_name = os.path.basename(pick)
                            break

            # Try common adapter loading methods. Prefer a resolved local .safetensors file when available.
            target = resolved if resolved is not None else (resolved_dir or redux_repo)
            if hasattr(pipe, "load_lora_weights"):
                # Some community LoRA-style adapters (like the Redux snapshot) use
                # non-standard key prefixes. Attempt a normal load first; if it
                # raises or the loader can't find expected keys, retry with
                # prefix=None which lets the loader match keys more flexibly.
                try:
                    # If we resolved a specific weight file within a directory, pass weight_name.
                    adapter_name = "redux"
                    if resolved_weight_name and (os.path.isdir(target) or target == redux_repo):
                        pipe.load_lora_weights(target, weight_name=resolved_weight_name, adapter_name=adapter_name)
                    else:
                        # If target is the repo id and we didn't resolve locally, try a common default name
                        default_weight = None
                        if target == redux_repo:
                            default_weight = "pytorch_lora_weights.safetensors"
                        if default_weight:
                            try:
                                pipe.load_lora_weights(target, weight_name=default_weight, adapter_name=adapter_name)
                            except Exception:
                                pipe.load_lora_weights(target, adapter_name=adapter_name)
                        else:
                            pipe.load_lora_weights(target, adapter_name=adapter_name)
                    # Try to activate the adapter if the pipeline supports it
                    try:
                        if hasattr(pipe, "set_adapters"):
                            pipe.set_adapters([adapter_name], adapter_weights=[1.0])
                    except Exception:
                        pass
                except Exception as e:
                    try:
                        # Retry with prefix=None to allow key matching without
                        # the pipeline-specific prefix assumptions.
                        if resolved_weight_name and (os.path.isdir(target) or target == redux_repo):
                            pipe.load_lora_weights(target, weight_name=resolved_weight_name, prefix=None, adapter_name="redux")
                        else:
                            pipe.load_lora_weights(target, prefix=None, adapter_name="redux")
                        try:
                            if hasattr(pipe, "set_adapters"):
                                pipe.set_adapters(["redux"], adapter_weights=[1.0])
                        except Exception:
                            pass
                    except Exception:
                        # If both attempts fail, surface the original error.
                        raise e
            # else silently skip
        except Exception as e:
            # Don't fail pipeline creation if adapter can't be loaded
            print(f"[warn] Failed to load Redux adapter from {redux_repo}: {e}")
    # Optionally load IP-Adapter to better follow the reference image semantics
    ip_adapter_repo = os.environ.get("IP_ADAPTER_REPO_ID")
    if ip_adapter_repo:
        try:
            target_dir = None
            weight_name = None
            image_encoder_path = None
            # If a file path directly
            if os.path.isfile(ip_adapter_repo) and ip_adapter_repo.endswith(('.safetensors', '.bin')):
                target_dir = os.path.dirname(ip_adapter_repo)
                weight_name = os.path.basename(ip_adapter_repo)
            # If a directory, pick first safetensors or bin; handle HF cache snapshot layout
            elif os.path.isdir(ip_adapter_repo):
                import glob
                search_root = ip_adapter_repo
                # Traverse HF cache structure
                snaps = os.path.join(ip_adapter_repo, 'snapshots')
                if os.path.isdir(snaps):
                    snap_dirs = sorted(os.listdir(snaps), reverse=True)
                    for s in snap_dirs:
                        cand = os.path.join(snaps, s)
                        if os.path.isdir(cand):
                            search_root = cand
                            break
                # Find weights
                for pat in ('**/ip-adapter*.safetensors','**/*.safetensors','**/ip-adapter*.bin','**/*.bin'):
                    found = glob.glob(os.path.join(search_root, pat), recursive=True)
                    if found:
                        found.sort()
                        weight_name = os.path.basename(found[0])
                        target_dir = os.path.dirname(found[0])
                        break
                # Resolve image_encoder dir
                if search_root and os.path.isdir(search_root):
                    enc_dir_root = os.path.join(search_root, 'image_encoder')
                    enc_dir_sibling = os.path.join(target_dir, 'image_encoder')
                    if os.path.isdir(enc_dir_root):
                        image_encoder_path = enc_dir_root
                    elif os.path.isdir(enc_dir_sibling):
                        image_encoder_path = enc_dir_sibling
                    else:
                        # Some snapshots include a file 'image_encoder' containing a HF repo id; map it to local cache
                        enc_file = os.path.join(search_root, 'image_encoder')
                        try:
                            if os.path.isfile(enc_file):
                                with open(enc_file, 'r', encoding='utf-8') as f:
                                    enc_repo_id = f.read().strip()
                                if enc_repo_id and '/' in enc_repo_id:
                                    cache_base = os.path.expanduser(os.path.join('~', '.cache', 'huggingface', 'hub'))
                                    enc_repo_dir = f"models--{enc_repo_id.replace('/', '--')}"
                                    enc_snaps = os.path.join(cache_base, enc_repo_dir, 'snapshots')
                                    if os.path.isdir(enc_snaps):
                                        snaps2 = sorted(os.listdir(enc_snaps), reverse=True)
                                        for s2 in snaps2:
                                            cand2 = os.path.join(enc_snaps, s2)
                                            if os.path.isdir(cand2) and os.path.isfile(os.path.join(cand2, 'config.json')):
                                                image_encoder_path = cand2
                                                break
                        except Exception:
                            pass
            else:
                # Treat as HF repo id; search local cache snapshot for a safetensors
                cache_base = os.path.expanduser(os.path.join('~', '.cache', 'huggingface', 'hub'))
                repo_dir = f"models--{ip_adapter_repo.replace('/', '--')}"
                snaps_dir = os.path.join(cache_base, repo_dir, 'snapshots')
                if os.path.isdir(snaps_dir):
                    snaps = sorted(os.listdir(snaps_dir), reverse=True)
                    for s in snaps:
                        snap_path = os.path.join(snaps_dir, s)
                        import glob
                        matches = glob.glob(os.path.join(snap_path, '**', '*.safetensors'), recursive=True)
                        if matches:
                            matches.sort()
                            weight_name = os.path.basename(matches[0])
                            target_dir = os.path.dirname(matches[0])
                            # Prefer an 'image_encoder' found at the snapshot root; else map repo id file to local cache
                            enc_root = os.path.join(snap_path, 'image_encoder')
                            enc_sibling = os.path.join(target_dir, 'image_encoder')
                            if os.path.isdir(enc_root):
                                image_encoder_path = enc_root
                            elif os.path.isdir(enc_sibling):
                                image_encoder_path = enc_sibling
                            else:
                                try:
                                    if os.path.isfile(enc_root):
                                        with open(enc_root, 'r', encoding='utf-8') as f:
                                            enc_repo_id = f.read().strip()
                                        if enc_repo_id and '/' in enc_repo_id:
                                            cache_base = os.path.expanduser(os.path.join('~', '.cache', 'huggingface', 'hub'))
                                            enc_repo_dir = f"models--{enc_repo_id.replace('/', '--')}"
                                            enc_snaps = os.path.join(cache_base, enc_repo_dir, 'snapshots')
                                            if os.path.isdir(enc_snaps):
                                                snaps2 = sorted(os.listdir(enc_snaps), reverse=True)
                                                for s2 in snaps2:
                                                    cand2 = os.path.join(enc_snaps, s2)
                                                    if os.path.isdir(cand2) and os.path.isfile(os.path.join(cand2, 'config.json')):
                                                        image_encoder_path = cand2
                                                        break
                                except Exception:
                                    pass
                            break
                # Fallback: try common weight names at repo root
                if target_dir is None:
                    common = [
                        'ip-adapter-flux.safetensors',
                        'ip-adapter-plus_sdxl.safetensors',
                        'ip-adapter_sdxl.safetensors',
                        'ip-adapter-plus_sd15.safetensors',
                        'ip-adapter_sd15.safetensors',
                    ]
                    for name in common:
                        try:
                            # Let diffusers resolve from hub if possible
                            weight_name = name
                            target_dir = ip_adapter_repo
                            break
                        except Exception:
                            continue

            if hasattr(pipe, 'load_ip_adapter') and target_dir and weight_name:
                # Use positional for the first required parameter to satisfy various mixin signatures
                passed = False
                # Validate encoder path looks like a model folder (has a config.json)
                if image_encoder_path and os.path.isdir(image_encoder_path) and os.path.isfile(os.path.join(image_encoder_path, 'config.json')):
                    try:
                        # Newer diffusers: image_encoder_folder
                        pipe.load_ip_adapter(target_dir, weight_name=weight_name, image_encoder_folder=image_encoder_path)
                        passed = True
                    except TypeError:
                        try:
                            # Some versions: image_encoder_path
                            pipe.load_ip_adapter(target_dir, weight_name=weight_name, image_encoder_path=image_encoder_path)
                            passed = True
                        except TypeError:
                            try:
                                # Fallback: 'image_encoder' param name
                                pipe.load_ip_adapter(target_dir, weight_name=weight_name, image_encoder=image_encoder_path)
                                passed = True
                            except Exception:
                                passed = False
                if not passed:
                    pipe.load_ip_adapter(target_dir, weight_name=weight_name)
            elif hasattr(pipe, 'load_ip_adapter') and target_dir:
                # Some implementations accept file path directly
                pipe.load_ip_adapter(target_dir)
            else:
                print(f"[warn] Could not resolve IP-Adapter weights for {ip_adapter_repo}")
        except Exception as e:
            print(f"[warn] Failed to load IP-Adapter from {ip_adapter_repo}: {e}")
    return pipe


def generate_image(input_image_path: str,
                   prompt: str,
                   output_path: str,
                   model_id: str,
                   device: str = "cuda",
                   num_inference_steps: int = 30,
                   guidance_scale: float = 7.5,
                   strength: float = 0.75,
                   seed: int | None = None,
                   negative_prompt: str | None = None,
                   scheduler_name: str | None = None,
                   ip_adapter_scale: float | None = None,
                   embed_metadata: bool = True,
                   save_metadata_json: bool = True,
                   ip_subject_aware: bool = False,
                   ip_bg_scale_factor: float = 0.5,
                   ip_mask_softness: int = 32):
    # Lightweight fallback for environments without the heavy diffusion stack or
    # when no model weights are supplied.  This keeps unit tests functional even
    # without large dependencies.
    if (
        not model_id
        or (model_id.lower().endswith((".gguf", ".safetensors", ".ckpt")) and not os.path.isfile(model_id))
        or not _diffusers_available()
    ):
        return _fallback_generate(
            input_image_path=input_image_path,
            prompt=prompt,
            output_path=output_path,
            model_id=model_id,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            seed=seed,
            negative_prompt=negative_prompt,
            scheduler_name=scheduler_name,
            ip_adapter_scale=ip_adapter_scale,
            embed_metadata=embed_metadata,
            save_metadata_json=save_metadata_json,
        )

    # Optional staged path for Flux on multi-GPU: set IVG_FLUX_STAGED=1 to enable
    try:
        if os.environ.get("IVG_FLUX_STAGED") == "1":
            model_l = model_id.lower()
            if "flux" in model_l or "flux.1" in model_l:
                meta = {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt or "",
                    "seed": int(seed) if seed is not None else None,
                    "steps": int(num_inference_steps),
                    "guidance_scale": float(guidance_scale),
                    "strength": float(strength),
                    "model_id": model_id,
                    "redux_repo": os.environ.get("REDUX_REPO_ID"),
                    "device_map": os.environ.get("IVG_DEVICE_MAP"),
                    "staged": True,
                }
                return _generate_image_flux_staged(
                    input_image_path=input_image_path,
                    prompt=prompt,
                    output_path=output_path,
                    model_id=model_id,
                    device=device,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    strength=strength,
                    seed=seed,
                    embed_metadata=True,
                    metadata=meta,
                )
    except Exception:
        pass
    # Lazy import torch
    # If model_id points to a local model file (.gguf/.safetensors/.ckpt), route through local WebUI API.
    if model_id.lower().endswith(('.gguf', '.safetensors', '.ckpt')):
        if not os.path.isfile(model_id):
            raise ValueError(
                f"Local model file not found: '{model_id}'. Please download the checkpoint first "
                "(e.g., into external/stable-diffusion-webui/models/Stable-diffusion/...), then provide the full path."
            )
        
        model_name = os.path.basename(model_id)
        return _webui_img2img_call(
            input_image_path=input_image_path,
            prompt=prompt,
            output_path=output_path,
            steps=num_inference_steps,
            cfg_scale=guidance_scale,
            denoising_strength=strength,
            seed=seed,
            negative_prompt=negative_prompt,
            sampler_name=scheduler_name,
            model_override=model_name,
        )

    # Lazy import torch (only needed for diffusers path)
    import torch

    img = load_image(input_image_path)
    pipe = get_pipeline(model_id, device=device)

    # Optionally swap scheduler based on a friendly name (Flux requires FlowMatch scheduler)
    try:
        is_flux = pipe.__class__.__name__.lower().startswith("flux") or ("flux" in model_id.lower())
    except Exception:
        is_flux = False

    if scheduler_name:
        try:
            name = scheduler_name.strip().lower()
            if is_flux:
                # Flux pipelines expect schedulers that support custom sigmas (FlowMatch Euler)
                try:
                    from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
                    if hasattr(pipe, "scheduler") and pipe.scheduler is not None:
                        pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
                except Exception:
                    pass
            else:
                from diffusers import (
                    EulerAncestralDiscreteScheduler,
                    EulerDiscreteScheduler,
                    DDIMScheduler,
                    DPMSolverMultistepScheduler,
                    HeunDiscreteScheduler,
                    LMSDiscreteScheduler,
                    DEISMultistepScheduler,
                    UniPCMultistepScheduler,
                )
                sched_map = {
                    "euler a": EulerAncestralDiscreteScheduler,
                    "euler": EulerDiscreteScheduler,
                    "ddim": DDIMScheduler,
                    "dpm++ 2m": DPMSolverMultistepScheduler,
                    "dpmpp2m": DPMSolverMultistepScheduler,
                    "heun": HeunDiscreteScheduler,
                    "lms": LMSDiscreteScheduler,
                    "deis": DEISMultistepScheduler,
                    "unipc": UniPCMultistepScheduler,
                }
                S = sched_map.get(name)
                if S is not None and hasattr(pipe, "scheduler") and pipe.scheduler is not None:
                    pipe.scheduler = S.from_config(pipe.scheduler.config)
        except Exception:
            pass

    generator = None
    # Detect accelerate device_map usage
    use_dm = False
    try:
        dm = os.environ.get("IVG_DEVICE_MAP", "").strip().lower()
        if dm in {"balanced", "auto"}:
            import torch as _t
            use_dm = _t.cuda.is_available() and _t.cuda.device_count() > 1
    except Exception:
        use_dm = False

    def _first_param_device(module):
        try:
            for p in module.parameters():
                dev = getattr(p, 'device', None)
                if dev is not None and dev.type == 'cuda':
                    return str(dev)
        except Exception:
            return None
        return None

    # If device_map is used, pick a stable execution device (prefer VAE device)
    exec_device = None
    if use_dm:
        try:
            if hasattr(pipe, 'vae') and pipe.vae is not None:
                exec_device = _first_param_device(pipe.vae)
            if exec_device is None and hasattr(pipe, 'unet') and pipe.unet is not None:
                exec_device = _first_param_device(pipe.unet)
            if exec_device is None and hasattr(pipe, 'transformer') and pipe.transformer is not None:
                exec_device = _first_param_device(pipe.transformer)
        except Exception:
            exec_device = None
        # Set pipeline execution device hint if available
        try:
            if exec_device:
                pipe._execution_device = exec_device
        except Exception:
            pass

    if seed is not None:
        try:
            if use_dm:
                # Use global seeding and let diffusers allocate noise on the same device as latents
                torch.manual_seed(int(seed))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(seed))
                generator = None
            else:
                gen_device = device
                base_dev = getattr(pipe, 'device', None)
                if isinstance(base_dev, str) and base_dev.startswith('cuda'):
                    gen_device = base_dev
                generator = torch.Generator(device=gen_device).manual_seed(int(seed))
        except Exception:
            generator = torch.Generator().manual_seed(int(seed))

    # For CUDA, use autocast for better perf/VRAM usage
    def _run_pipe_with_scale(scale_override: float | None = None):
        # If an IP-Adapter repo was loaded, try to pass image and scale
        ip_repo = os.environ.get("IP_ADAPTER_REPO_ID")
        if ip_repo is not None and str(ip_repo).strip():
            try:
                kwargs = {
                    "prompt": prompt,
                    "image": img,
                    "strength": strength,
                    "guidance_scale": guidance_scale,
                    "num_inference_steps": num_inference_steps,
                    "generator": generator,
                    "negative_prompt": negative_prompt,
                    "ip_adapter_image": img,
                }
                use_scale = scale_override if scale_override is not None else ip_adapter_scale
                if use_scale is not None:
                    kwargs["ip_adapter_scale"] = float(use_scale)
                return pipe(**kwargs)
            except TypeError:
                # Pipeline doesn't accept ip_adapter args; fall back
                pass
        return pipe(
            prompt=prompt,
            image=img,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            negative_prompt=negative_prompt,
        )

    def _center_mask(size: tuple[int,int], softness: int = 32) -> Image.Image:
        w, h = size
        # Elliptical center-weight mask with gaussian falloff
        from PIL import ImageDraw, ImageFilter
        mask = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(mask)
        # ellipse covering ~70% of area
        pad_w, pad_h = int(0.15 * w), int(0.15 * h)
        draw.ellipse((pad_w, pad_h, w - pad_w, h - pad_h), fill=255)
        if softness > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(radius=softness))
        return mask

    def _face_mask(image: Image.Image, softness: int = 32) -> Image.Image | None:
        # Try mediapipe face detection if available; else None
        try:
            import mediapipe as mp
            mp_fd = mp.solutions.face_detection
            img_rgb = image.convert('RGB')
            w, h = img_rgb.size
            with mp_fd.FaceDetection(model_selection=1, min_detection_confidence=0.5) as det:
                import numpy as np
                arr = np.array(img_rgb)
                res = det.process(arr)
                if not res.detections:
                    return None
                # Build mask from union of detected faces with some expansion
                from PIL import ImageDraw, ImageFilter
                mask = Image.new('L', (w, h), 0)
                draw = ImageDraw.Draw(mask)
                for d in res.detections:
                    bb = d.location_data.relative_bounding_box
                    x0 = max(0, int(bb.xmin * w) - int(0.05 * w))
                    y0 = max(0, int(bb.ymin * h) - int(0.05 * h))
                    x1 = min(w, int((bb.xmin + bb.width) * w) + int(0.05 * w))
                    y1 = min(h, int((bb.ymin + bb.height) * h) + int(0.05 * h))
                    draw.ellipse((x0, y0, x1, y1), fill=255)
                if softness > 0:
                    mask = mask.filter(ImageFilter.GaussianBlur(radius=softness))
                return mask
        except Exception:
            return None

    def _normalize_mask_for_output(mask: Image.Image, target_size: tuple[int,int]) -> Image.Image:
        # Ensure mask is single channel 'L' and matches the output resolution
        if mask.mode != 'L':
            mask = mask.convert('L')
        if mask.size != target_size:
            # High quality resize with LANCZOS
            mask = mask.resize(target_size, Image.LANCZOS)
        return mask

    if device.startswith("cuda"):
        with torch.autocast(device):
            try:
                if ip_subject_aware and os.environ.get("IP_ADAPTER_REPO_ID"):
                    # Two-pass generate: high subject scale and lower background scale, then composite
                    hi_scale = float(ip_adapter_scale if ip_adapter_scale is not None else 0.8)
                    lo_scale = float(max(0.0, hi_scale * float(ip_bg_scale_factor)))
                    if seed is not None:
                        try:
                            torch.manual_seed(int(seed))
                            if torch.cuda.is_available():
                                torch.cuda.manual_seed_all(int(seed))
                        except Exception:
                            pass
                    res_hi = _run_pipe_with_scale(hi_scale)
                    if seed is not None:
                        try:
                            torch.manual_seed(int(seed))
                            if torch.cuda.is_available():
                                torch.cuda.manual_seed_all(int(seed))
                        except Exception:
                            pass
                    res_lo = _run_pipe_with_scale(lo_scale)
                    img_hi = res_hi.images[0]
                    img_lo = res_lo.images[0]
                    # Compute subject mask on reference image and resize to output size
                    m = _face_mask(img, ip_mask_softness) or _center_mask(img.size, ip_mask_softness)
                    m = _normalize_mask_for_output(m, img_hi.size)
                    # Ensure modes match
                    if img_hi.mode != 'RGB':
                        img_hi = img_hi.convert('RGB')
                    if img_lo.mode != 'RGB':
                        img_lo = img_lo.convert('RGB')
                    comp = img_lo.copy()
                    comp.paste(img_hi, mask=m)
                    result = type(res_hi)(images=[comp], nsfw_content_detected=getattr(res_hi, 'nsfw_content_detected', None))
                else:
                    result = _run_pipe_with_scale()
            except ValueError as ve:
                # Fallback for Flux when scheduler doesn't support custom sigmas
                msg = str(ve).lower()
                if "custom sigmas" in msg or "set_timesteps" in msg:
                    try:
                        from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
                        if hasattr(pipe, "scheduler") and pipe.scheduler is not None:
                            pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
                        result = _run_pipe()
                    except Exception:
                        raise
                else:
                    raise
    else:
        try:
            if ip_subject_aware and os.environ.get("IP_ADAPTER_REPO_ID"):
                hi_scale = float(ip_adapter_scale if ip_adapter_scale is not None else 0.8)
                lo_scale = float(max(0.0, hi_scale * float(ip_bg_scale_factor)))
                if seed is not None:
                    try:
                        torch.manual_seed(int(seed))
                    except Exception:
                        pass
                res_hi = _run_pipe_with_scale(hi_scale)
                if seed is not None:
                    try:
                        torch.manual_seed(int(seed))
                    except Exception:
                        pass
                res_lo = _run_pipe_with_scale(lo_scale)
                img_hi = res_hi.images[0]
                img_lo = res_lo.images[0]
                m = _face_mask(img, ip_mask_softness) or _center_mask(img.size, ip_mask_softness)
                m = _normalize_mask_for_output(m, img_hi.size)
                if img_hi.mode != 'RGB':
                    img_hi = img_hi.convert('RGB')
                if img_lo.mode != 'RGB':
                    img_lo = img_lo.convert('RGB')
                comp = img_lo.copy()
                comp.paste(img_hi, mask=m)
                result = type(res_hi)(images=[comp], nsfw_content_detected=getattr(res_hi, 'nsfw_content_detected', None))
            else:
                result = _run_pipe_with_scale()
        except ValueError as ve:
            msg = str(ve).lower()
            if "custom sigmas" in msg or "set_timesteps" in msg:
                try:
                    from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
                    if hasattr(pipe, "scheduler") and pipe.scheduler is not None:
                        pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
                    result = _run_pipe()
                except Exception:
                    raise
            else:
                raise

    out = result.images[0]
    # Build metadata
    meta = {
        "prompt": prompt,
        "negative_prompt": negative_prompt or "",
        "seed": int(seed) if seed is not None else None,
        "steps": int(num_inference_steps),
        "guidance_scale": float(guidance_scale),
        "strength": float(strength),
        "scheduler": (type(getattr(pipe, "scheduler", None)).__name__ if getattr(pipe, "scheduler", None) is not None else None),
        "model_id": model_id,
        "redux_repo": os.environ.get("REDUX_REPO_ID"),
        "ip_adapter_repo": os.environ.get("IP_ADAPTER_REPO_ID"),
        "ip_adapter_scale": float(ip_adapter_scale) if ip_adapter_scale is not None else None,
        "device_map": os.environ.get("IVG_DEVICE_MAP"),
    }
    if embed_metadata:
        try:
            info = PngInfo()
            info.add_text("ivg:meta", json.dumps(meta, ensure_ascii=False))
            out.save(output_path, pnginfo=info)
        except Exception:
            out.save(output_path)
    else:
        out.save(output_path)
    if save_metadata_json:
        try:
            sidecar = os.path.splitext(output_path)[0] + ".json"
            with open(sidecar, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    return output_path


def _generate_image_flux_staged(input_image_path: str,
                                prompt: str,
                                output_path: str,
                                model_id: str,
                                device: str = "cuda",
                                num_inference_steps: int = 20,
                                guidance_scale: float = 5.0,
                                strength: float = 0.75,
                                seed: int | None = None,
                                embed_metadata: bool = True,
                                metadata: dict | None = None) -> str:
    # Staged multi-GPU execution for Flux img2img without CPU offload.
    # - Text encoding: minimal pipeline, tokenizers implicitly on CPU.
    # - VAE encode/decode on secondary GPU.
    # - Transformer sharded across both GPUs via accelerate (device_map='auto').
    import torch, gc
    from diffusers import FluxImg2ImgPipeline, FluxPipeline
    from diffusers import AutoModel, AutoencoderKL
    try:
        from diffusers.image_processor import PreprocessorImage as _ImagePreproc
    except Exception:
        _ImagePreproc = None
    from PIL import Image as _PIL_Image

    assert device.startswith("cuda"), "Staged Flux path requires CUDA"
    n = torch.cuda.device_count()
    assert n >= 2, "Staged Flux path requires at least 2 GPUs"

    # Pick primary/secondary with optional override
    explicit_idx = None
    try:
        if ":" in device:
            explicit_idx = int(device.split(":", 1)[1])
    except Exception:
        explicit_idx = None
    env_idx = os.environ.get("IVG_PRIMARY_CUDA") or os.environ.get("PRIMARY_CUDA")
    if explicit_idx is None and env_idx is not None:
        try:
            explicit_idx = int(env_idx)
        except Exception:
            explicit_idx = None
    # Choose primary as explicit or max free
    free_bytes = [(torch.cuda.mem_get_info(i)[0], i) for i in range(n)]
    free_bytes.sort(reverse=True)
    if explicit_idx is not None and 0 <= explicit_idx < n:
        primary_idx = explicit_idx
        secondary_idx = next((i for (_, i) in free_bytes if i != primary_idx), (primary_idx + 1) % n)
    else:
        primary_idx = free_bytes[0][1]
        secondary_idx = free_bytes[1][1]
    primary = f"cuda:{primary_idx}"
    secondary = f"cuda:{secondary_idx}"

    torch_dtype = torch.float16
    gen = None
    if seed is not None:
        gen = torch.Generator(device=secondary).manual_seed(int(seed))

    def flush():
        gc.collect()
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    # Stage 0: load image on CPU
    img = _PIL_Image.open(input_image_path).convert("RGB")

    # Stage 1: Text encoding using a minimal Flux pipeline (no transformer/vae loaded)
    pipe_text = FluxPipeline.from_pretrained(
        model_id,
        transformer=None,
        vae=None,
        torch_dtype=torch_dtype,
        device_map={"tokenizer": "cpu", "tokenizer_2": "cpu", "text_encoder": secondary, "text_encoder_2": secondary},
    )
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, _ = pipe_text.encode_prompt(prompt=prompt, max_sequence_length=512)
        # Move embeds to primary for denoising stage
        try:
            prompt_embeds = prompt_embeds.to(primary)
            pooled_prompt_embeds = pooled_prompt_embeds.to(primary)
        except Exception:
            pass
    del pipe_text
    flush()

    # Stage 2: Image encode via VAE on secondary
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=torch_dtype).to(secondary)
    try:
        vae.enable_tiling()
    except Exception:
        pass
    # Preprocess to tensor; fall back to PIL->tensor if processor missing
    image_processor = _ImagePreproc() if _ImagePreproc else None
    with torch.no_grad():
        if image_processor is not None and hasattr(image_processor, 'preprocess'):
            img_t = image_processor.preprocess(img).to(device=secondary, dtype=torch_dtype)
        else:
            import torchvision.transforms as T
            t = T.Compose([T.ToTensor()])
            img_t = t(img).unsqueeze(0).to(device=secondary, dtype=torch_dtype)
        z = vae.encode(img_t).latent_dist.sample(generator=gen) * vae.config.scaling_factor
    flush()

    # Stage 3: Transformer sharded across both GPUs
    transformer = AutoModel.from_pretrained(
        model_id,
        subfolder="transformer",
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    pipe_core = FluxImg2ImgPipeline.from_pretrained(
        model_id,
        text_encoder=None, text_encoder_2=None,
        tokenizer=None, tokenizer_2=None,
        vae=None,
        transformer=transformer,
        torch_dtype=torch_dtype,
    )
    try:
        pipe_core.enable_attention_slicing("auto")
    except Exception:
        pass

    with torch.no_grad():
        out = pipe_core(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            image=z.to(primary),
            strength=float(strength),
            guidance_scale=float(guidance_scale),
            num_inference_steps=int(num_inference_steps),
            output_type="latent",
        )
        denoised_latents = out.images
    del pipe_core, transformer
    flush()

    # Stage 4: Decode on secondary
    with torch.no_grad():
        latents = denoised_latents.to(secondary) / vae.config.scaling_factor + vae.config.shift_factor
        imgs = vae.decode(latents, return_dict=False)[0]
        # Postprocess: clamp and convert to PIL
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        import numpy as np
        from PIL import Image as _PIL
        arr = (imgs[0].detach().float().cpu().permute(1,2,0).numpy() * 255).astype('uint8')
        im = _PIL.fromarray(arr)
        if embed_metadata and metadata:
            try:
                info = PngInfo()
                info.add_text("ivg:meta", json.dumps(metadata, ensure_ascii=False))
                im.save(output_path, pnginfo=info)
                return output_path
            except Exception:
                im.save(output_path)
                return output_path
        im.save(output_path)
        return output_path

