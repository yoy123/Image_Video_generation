import os
import tempfile
from typing import List
from PIL import Image
import imageio

from .generate import generate_image


def generate_frames_from_image(input_image_path: str,
                               prompt: str,
                               model_id: str,
                               num_frames: int = 8,
                               temp_dir: str | None = None,
                               device: str = "cuda",
                               base_seed: int | None = None) -> List[str]:
    """Generate a sequence of frames by calling the image generator multiple times.

    This naive approach varies the RNG seed across frames to create motion.
    """
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="ivg_frames_")
    frame_paths = []
    for i in range(num_frames):
        out_path = os.path.join(temp_dir, f"frame_{i:03d}.png")
        seed = (base_seed if base_seed is not None else 42) + i * 7
        # Slightly vary strength or prompt if desired to create temporal change
        generate_image(
            input_image_path=input_image_path,
            prompt=prompt,
            output_path=out_path,
            model_id=model_id,
            device=device,
            seed=seed,
            strength=0.6,
            num_inference_steps=25,
        )
        frame_paths.append(out_path)
    return frame_paths


def frames_to_video(frame_paths: List[str], out_path: str, fps: int = 8):
    writer = imageio.get_writer(out_path, fps=fps, macro_block_size=None)
    for p in frame_paths:
        img = Image.open(p).convert("RGB")
        writer.append_data(imageio.imread(p))
    writer.close()
    return out_path


def image_to_video(input_image_path: str,
                   prompt: str,
                   model_id: str,
                   out_path: str,
                   num_frames: int = 8,
                   fps: int = 8,
                   device: str = "cuda",
                   seed: int | None = None) -> str:
    frames = generate_frames_from_image(
        input_image_path=input_image_path,
        prompt=prompt,
        model_id=model_id,
        num_frames=num_frames,
        device=device,
        base_seed=seed,
    )
    return frames_to_video(frames, out_path, fps=fps)
