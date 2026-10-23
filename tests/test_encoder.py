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
