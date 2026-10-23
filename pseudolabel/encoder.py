"""Image encoders (image tower only) and an on-disk embedding cache."""
import hashlib
import json
import re
from pathlib import Path
from typing import Protocol

import numpy as np

from pseudolabel.data import Item
from pseudolabel.images import load_image


class ImageEncoder(Protocol):
    version: str

    def embed(self, images: list[str]) -> np.ndarray: ...


def _checkpoint_dir(checkpoint: str) -> Path:
    if Path(checkpoint).is_dir():
        return Path(checkpoint)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(checkpoint, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"]))


def _fingerprint(directory: Path) -> str:
    h = hashlib.sha256((directory / "config.json").read_bytes())
    for f in sorted(directory.glob("*.safetensors")):
        h.update(f"{f.name}:{f.stat().st_size}".encode())
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
                feats = self.model.get_image_features(**inputs).float()
            out.append(torch.nn.functional.normalize(feats, dim=-1).cpu().numpy())
        return np.concatenate(out).astype(np.float32) if out else np.zeros((0, 0), np.float32)


def _cache_path(cache_dir: Path, version: str) -> Path:
    return Path(cache_dir) / (re.sub(r"[^A-Za-z0-9._@-]+", "_", version) + ".npz")


def embed_items(encoder: ImageEncoder, items: list[Item], cache_dir: Path) -> np.ndarray:
    """Embeddings for `items` in order; only images missing from the cache are encoded."""
    path = _cache_path(cache_dir, encoder.version)
    cache: dict[str, np.ndarray] = {}
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            cache = dict(zip(json.loads(str(data["keys"])), data["vectors"]))
    missing = list(dict.fromkeys(i.image for i in items if i.image not in cache))
    if missing:
        cache.update(zip(missing, encoder.embed(missing)))
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = list(cache)
        np.savez(path, keys=np.array(json.dumps(keys)), vectors=np.stack([cache[k] for k in keys]))
    return np.stack([cache[i.image] for i in items]).astype(np.float32)
