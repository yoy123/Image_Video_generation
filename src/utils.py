from PIL import Image


def load_image(path):
    """Load an image and convert to RGB PIL image."""
    return Image.open(path).convert("RGB")
