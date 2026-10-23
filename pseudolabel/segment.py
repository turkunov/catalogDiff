"""Cut the item out of the photo before labeling: SAM3 by the category's text prompt, or BiRefNet's salient object.

The labeler then sees the item cropped to its box on a white background. When no usable mask is found the
original photo is kept.
"""
import hashlib
import json
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from pseudolabel.images import ImageError, load_image


class Segmenter(Protocol):
    name: str

    def mask(self, img: Image.Image) -> np.ndarray | None: ...


def apply_mask(img: Image.Image, mask: np.ndarray | None, pad: float = 0.05, min_area: float = 0.02) -> Image.Image:
    """The masked object on white, cropped to its box plus `pad` of the frame; `img` itself if the mask is too small."""
    if mask is None or mask.mean() < min_area:
        return img
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    py, px = int(pad * h), int(pad * w)
    y0, y1 = max(0, ys.min() - py), min(h, ys.max() + 1 + py)
    x0, x1 = max(0, xs.min() - px), min(w, xs.max() + 1 + px)
    arr = np.asarray(img.convert("RGB")).copy()
    arr[~mask] = 255
    return Image.fromarray(arr[y0:y1, x0:x1])


class Sam3Segmenter:
    """SAM3 promptable concept segmentation; the highest-scoring instance of the text prompt."""

    name = "sam3"

    def __init__(self, prompt: str, device: str = "cuda", checkpoint: str = "facebook/sam3", threshold: float = 0.4):
        import torch
        from transformers import Sam3Model, Sam3Processor

        self.prompt, self.device, self.threshold, self._torch = prompt, device, threshold, torch
        self.processor = Sam3Processor.from_pretrained(checkpoint)
        self.model = Sam3Model.from_pretrained(checkpoint).to(device).eval()

    def mask(self, img: Image.Image) -> np.ndarray | None:
        inputs = self.processor(images=img, text=self.prompt, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            outputs = self.model(**inputs)
        res = self.processor.post_process_instance_segmentation(
            outputs, threshold=self.threshold, mask_threshold=0.5, target_sizes=inputs.get("original_sizes").tolist())[0]
        if len(res["scores"]) == 0:
            return None
        return res["masks"][int(res["scores"].argmax())].cpu().numpy().astype(bool)


class BiRefNetSegmenter:
    """BiRefNet dichotomous (salient object) segmentation at 1024x1024."""

    name = "birefnet"
    SIZE = 1024
    MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])

    def __init__(self, device: str = "cuda", checkpoint: str = "ZhengPeng7/BiRefNet", threshold: float = 0.5):
        import torch
        from transformers import AutoModelForImageSegmentation

        self.device, self.threshold, self._torch = device, threshold, torch
        self.model = AutoModelForImageSegmentation.from_pretrained(checkpoint, trust_remote_code=True).to(device).eval()

    def mask(self, img: Image.Image) -> np.ndarray | None:
        torch = self._torch
        x = np.asarray(img.convert("RGB").resize((self.SIZE, self.SIZE), Image.BILINEAR), dtype=np.float32) / 255.0
        dtype = next(self.model.parameters()).dtype  # the checkpoint loads in float16
        x = torch.from_numpy(((x - self.MEAN) / self.STD).transpose(2, 0, 1)).unsqueeze(0).to(self.device, dtype)
        with torch.no_grad():
            pred = self.model(x)[-1].sigmoid()[0, 0].float().cpu().numpy()
        prob = np.asarray(Image.fromarray((pred * 255).astype(np.uint8)).resize(img.size, Image.BILINEAR)) / 255.0
        return prob > self.threshold


def make_segmenter(method: str, prompt: str, device: str = "cuda") -> Segmenter:
    if method == "sam3":
        return Sam3Segmenter(prompt, device)
    if method == "birefnet":
        return BiRefNetSegmenter(device)
    raise ValueError(f"unknown segmentation method {method!r}: sam3 or birefnet")


def segment_images(segmenter: Segmenter, paths: list[str], out_dir: Path) -> tuple[dict[str, str], dict]:
    """Cut-out JPEG per readable photo (cached by path); returns {photo: cut-out} and counts."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, str] = {}
    stats = {"segmented": 0, "kept_original": 0, "unreadable": 0, "cached": 0}
    for src in dict.fromkeys(paths):
        dst = out_dir / (hashlib.sha1(src.encode()).hexdigest()[:20] + ".jpeg")
        if dst.exists():
            mapping[src] = str(dst)
            stats["cached"] += 1
            continue
        try:
            img = load_image(src)
        except ImageError:
            stats["unreadable"] += 1
            continue
        out = apply_mask(img, segmenter.mask(img))
        stats["kept_original" if out is img else "segmented"] += 1
        out.save(dst, "JPEG", quality=95)
        mapping[src] = str(dst)
    (out_dir / "stats.json").write_text(json.dumps(stats), encoding="utf-8")
    return mapping, stats
