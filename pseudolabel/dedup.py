"""Same-photo shortcut: a pair whose photos are near-identical (perceptual hash) is class 5 without the VLM."""
import hashlib
from functools import lru_cache

import numpy as np
from PIL import Image

from pseudolabel.images import ImageError, load_image

HASH_SIZE, SCALE = 8, 4


@lru_cache(maxsize=4)
def _dct_matrix(n: int) -> np.ndarray:
    k, i = np.arange(n)[:, None], np.arange(n)[None, :]
    return np.cos(np.pi * (2 * i + 1) * k / (2 * n))


def phash(img: Image.Image) -> int:
    """64-bit DCT perceptual hash (as in the imagehash library)."""
    n = HASH_SIZE * SCALE
    g = np.asarray(img.convert("L").resize((n, n), Image.LANCZOS), dtype=np.float64)
    c = _dct_matrix(n)
    low = (c @ g @ c.T)[:HASH_SIZE, :HASH_SIZE]
    bits = (low > np.median(low)).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(x: int, y: int) -> int:
    return bin(x ^ y).count("1")


@lru_cache(maxsize=65536)
def _phash_of(src: str) -> int | None:
    try:
        return phash(load_image(src))
    except ImageError:
        return None


def photo_distance(a_src: str, b_src: str) -> int | None:
    """Hamming distance between the photos' hashes; None when a photo cannot be read."""
    ha, hb = _phash_of(a_src), _phash_of(b_src)
    return None if ha is None or hb is None else hamming(ha, hb)


def picked_for_check(a_id: str, b_id: str, share: float) -> bool:
    """Deterministic sample of shortcut pairs that still go to the VLM, to measure the shortcut's precision."""
    h = int(hashlib.sha1(f"{a_id}|{b_id}".encode()).hexdigest()[:8], 16)
    return h / 0xFFFFFFFF < share
