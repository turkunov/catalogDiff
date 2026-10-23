"""Image encoders (image tower only) and an on-disk embedding cache."""
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Protocol

import numpy as np

from pseudolabel.data import Item
from pseudolabel.images import ImageError, load_image


class ImageEncoder(Protocol):
    version: str

    def embed(self, images: list[str]) -> np.ndarray: ...


def _checkpoint_dir(checkpoint: str) -> Path:
    if Path(checkpoint).is_dir():
        return Path(checkpoint)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(checkpoint, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"]))


def _fingerprint(directory: Path) -> str:
    """Content identity of a checkpoint: the hub commit for hub snapshots, else a hash of config and weights."""
    directory = Path(directory)
    if directory.parent.name == "snapshots" and re.fullmatch(r"[0-9a-f]{40}", directory.name):
        return directory.name[:12]
    h = hashlib.sha256((directory / "config.json").read_bytes())
    for f in sorted([*directory.glob("*.safetensors"), *directory.glob("*.bin")]):
        h.update(f.name.encode())
        with f.open("rb") as fh:
            for block in iter(lambda: fh.read(1 << 24), b""):
                h.update(block)
    return h.hexdigest()[:12]


class SigLIP2Encoder:
    def __init__(self, checkpoint: str = "google/siglip2-so400m-patch14-384", device: str = "cuda", batch_size: int = 64):
        import torch
        from transformers import AutoModel, AutoProcessor

        directory = _checkpoint_dir(checkpoint)
        self.version = f"{checkpoint}@{_fingerprint(directory)}"
        self.device, self.batch_size, self._torch = device, batch_size, torch
        self.processor = AutoProcessor.from_pretrained(directory)
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.model = AutoModel.from_pretrained(directory, torch_dtype=dtype).to(device).eval()

    def embed(self, images: list[str]) -> np.ndarray:
        torch, out = self._torch, []
        for start in range(0, len(images), self.batch_size):
            batch = [load_image(src) for src in images[start:start + self.batch_size]]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            with torch.no_grad():
                result = self.model.get_image_features(**inputs)
            # transformers 5.x returns a model output with pooler_output; 4.x returns the tensor itself.
            feats = (result if isinstance(result, torch.Tensor) else result.pooler_output).float()
            out.append(torch.nn.functional.normalize(feats, dim=-1).cpu().numpy())
        return np.concatenate(out).astype(np.float32) if out else np.zeros((0, 0), np.float32)


def _cache_path(cache_dir: Path, version: str) -> Path:
    return Path(cache_dir) / (re.sub(r"[^A-Za-z0-9._@-]+", "_", version) + ".npz")


CHUNK = 256


def _embed_safely(encoder: ImageEncoder, images: list[str]) -> dict[str, np.ndarray]:
    """Embed in chunks; a chunk with an unreadable image is retried image by image and that image is skipped."""
    out: dict[str, np.ndarray] = {}
    for start in range(0, len(images), CHUNK):
        chunk = images[start:start + CHUNK]
        try:
            out.update(zip(chunk, encoder.embed(chunk)))
        except ImageError:
            for image in chunk:
                try:
                    out[image] = encoder.embed([image])[0]
                except ImageError:
                    continue
    return out


def _save_cache(path: Path, cache: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(cache)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, keys=np.array(json.dumps(keys)), vectors=np.stack([cache[k] for k in keys]))
    os.replace(tmp, path)


def embed_items(encoder: ImageEncoder, items: list[Item], cache_dir: Path) -> np.ndarray:
    """Embeddings for `items` in order; only uncached images are encoded.

    Items whose image cannot be read get a zero vector (never cached), so one broken image does not stop
    the iteration: its pairs are still labeled and the labeler records the image error.
    """
    path = _cache_path(cache_dir, encoder.version)
    cache: dict[str, np.ndarray] = {}
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            cache = dict(zip(json.loads(str(data["keys"])), data["vectors"]))
    missing = list(dict.fromkeys(i.image for i in items if i.image not in cache))
    if missing:
        new = _embed_safely(encoder, missing)
        if new:
            cache.update(new)
            _save_cache(path, cache)
    if not cache:
        raise ImageError("no image of the universe could be read")
    zeros = np.zeros_like(next(iter(cache.values())), dtype=np.float32)
    return np.stack([cache.get(i.image, zeros) for i in items]).astype(np.float32)
