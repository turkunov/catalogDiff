import numpy as np
import pytest

from fakes import FakeEncoder
from helpers import make_jpeg
from pseudolabel.encoder import embed_items


def test_embed_items_uses_cache(tmp_path, items_with_images):
    enc = FakeEncoder({it.image: [1, i] for i, it in enumerate(items_with_images)})
    first = embed_items(enc, items_with_images, tmp_path)
    enc.embed = lambda imgs: (_ for _ in ()).throw(AssertionError("should be cached"))
    again = embed_items(enc, items_with_images, tmp_path)
    assert again.shape == (len(items_with_images), 2) and np.allclose(first, again)


def test_embed_items_computes_only_missing(tmp_path, items_with_images):
    enc = FakeEncoder({it.image: [1, i] for i, it in enumerate(items_with_images)})
    embed_items(enc, items_with_images[:2], tmp_path)
    enc.embedded.clear()
    embed_items(enc, items_with_images, tmp_path)
    assert enc.embedded == [items_with_images[2].image]


@pytest.mark.slow
def test_siglip2_smoke(tmp_path):
    from pseudolabel.encoder import SigLIP2Encoder

    enc = SigLIP2Encoder("google/siglip2-base-patch16-224", device="cpu")
    img = make_jpeg(tmp_path, size=(300, 200))
    e = enc.embed([str(img), str(img)])
    assert e.shape[0] == 2 and np.allclose(np.linalg.norm(e, axis=1), 1, atol=1e-4)
    assert float(e[0] @ e[1]) == pytest.approx(1.0, abs=1e-4)
    assert enc.version.startswith("google/siglip2-base-patch16-224@")


class LoadingEncoder:
    """Opens every image like SigLIP2 does; one unreadable image raises ImageError for its batch."""

    version = "loading@1"

    def embed(self, images):
        from pseudolabel.images import load_image

        return np.stack([np.asarray(load_image(i).resize((2, 1))).reshape(-1)[:2] + 1.0 for i in images]).astype(np.float32)


def test_embed_items_survives_broken_image(tmp_path, items_with_images):
    from pathlib import Path

    Path(items_with_images[1].image).unlink()
    emb = embed_items(LoadingEncoder(), items_with_images, tmp_path)
    assert emb.shape == (3, 2) and not emb[1].any() and emb[0].any()


def _checkpoint(dir_, weights: bytes):
    dir_.mkdir(parents=True)
    (dir_ / "config.json").write_text('{"model_type": "siglip"}')
    (dir_ / "model.safetensors").write_bytes(weights)
    return dir_


def test_fingerprint_depends_on_weight_contents(tmp_path):
    from pseudolabel.encoder import _fingerprint

    a = _checkpoint(tmp_path / "a", b"\x00" * 1024)
    b = _checkpoint(tmp_path / "b", b"\x01" * 1024)  # same size, different weights
    assert _fingerprint(a) != _fingerprint(b)
    (tmp_path / "c").mkdir()
    (tmp_path / "c" / "config.json").write_text('{"model_type": "siglip"}')
    (tmp_path / "c" / "pytorch_model.bin").write_bytes(b"\x02" * 10)
    assert _fingerprint(tmp_path / "c") != _fingerprint(a)


def test_fingerprint_uses_hub_commit(tmp_path):
    from pseudolabel.encoder import _fingerprint

    commit = "fc05daec18b0a78c049392ed2e771dde82bdf654"
    snap = _checkpoint(tmp_path / "models--google--siglip2" / "snapshots" / commit, b"\x00" * 8)
    assert _fingerprint(snap) == commit[:12]
