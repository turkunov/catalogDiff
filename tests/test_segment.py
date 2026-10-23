import numpy as np
import pytest
from PIL import Image

from helpers import make_jpeg


def _mask(h, w, box):
    m = np.zeros((h, w), dtype=bool)
    y0, y1, x0, x1 = box
    m[y0:y1, x0:x1] = True
    return m


def test_apply_mask_crops_and_whitens():
    from pseudolabel.segment import apply_mask

    img = Image.new("RGB", (100, 80), (10, 20, 30))
    mask = _mask(80, 100, (20, 60, 30, 70))
    mask[40, 50] = False  # a hole inside the object turns white
    out = apply_mask(img, mask, pad=0.0)
    assert out.size == (40, 40)
    arr = np.asarray(out)
    assert tuple(arr[0, 0]) == (10, 20, 30) and tuple(arr[20, 20]) == (255, 255, 255)


def test_apply_mask_pads_inside_the_frame():
    from pseudolabel.segment import apply_mask

    img = Image.new("RGB", (100, 100), (0, 0, 0))
    out = apply_mask(img, _mask(100, 100, (0, 50, 0, 50)), pad=0.1)
    assert out.size == (60, 60)


def test_apply_mask_falls_back_on_tiny_or_missing_mask():
    from pseudolabel.segment import apply_mask

    img = Image.new("RGB", (100, 100), (1, 2, 3))
    assert apply_mask(img, _mask(100, 100, (0, 1, 0, 1))) is img
    assert apply_mask(img, None) is img


class FakeSegmenter:
    name = "fake"

    def __init__(self):
        self.calls = 0

    def mask(self, img):
        self.calls += 1
        w, h = img.size
        return _mask(h, w, (h // 4, 3 * h // 4, w // 4, 3 * w // 4))


def test_segment_images_writes_once_and_maps_paths(tmp_path):
    from pseudolabel.segment import segment_images

    paths = [str(make_jpeg(tmp_path, size=(80, 80), name=f"{i}.jpeg")) for i in range(2)]
    seg = FakeSegmenter()
    mapping, stats = segment_images(seg, paths + [str(tmp_path / "missing.jpeg")], tmp_path / "seg")
    assert set(mapping) == set(paths) and all(Image.open(p).size == (48, 48) for p in mapping.values())
    assert stats["segmented"] == 2 and stats["unreadable"] == 1
    segment_images(seg, paths, tmp_path / "seg")
    assert seg.calls == 2


@pytest.mark.slow
@pytest.mark.parametrize("method", ["sam3", "birefnet"])
def test_real_segmenters_find_an_object(tmp_path, method):
    from pseudolabel.segment import make_segmenter

    img = Image.new("RGB", (256, 256), (255, 255, 255))
    img.paste((150, 30, 30), (64, 96, 192, 192))
    m = make_segmenter(method, prompt="box", device="cuda").mask(img)
    assert m is None or m.shape == (256, 256)
