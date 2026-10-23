"""Image loading from files, http(s) and data URLs; JPEG data URLs for the VLM."""
import base64
import io
from pathlib import Path

import requests
from PIL import Image


class ImageError(RuntimeError):
    pass


def load_image(src: str, timeout: int = 20) -> Image.Image:
    try:
        if src.startswith("data:"):
            raw = base64.b64decode(src.split(",", 1)[1])
        elif src.startswith(("http://", "https://")):
            resp = requests.get(src, timeout=timeout)
            resp.raise_for_status()
            raw = resp.content
        else:
            raw = Path(src.removeprefix("file://")).read_bytes()
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise ImageError(f"cannot load image {src[:120]}: {e}") from e


def to_data_url(src: str, max_side: int = 768) -> str:
    img = load_image(src)
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
