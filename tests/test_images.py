import pytest

from helpers import make_jpeg
from pseudolabel.images import ImageError, load_image, to_data_url


def test_to_data_url_resizes(tmp_path):
    path = make_jpeg(tmp_path, size=(2000, 1000))
    img = load_image(to_data_url(str(path)))
    assert max(img.size) == 768


def test_to_data_url_does_not_upscale(tmp_path):
    img = load_image(to_data_url(str(make_jpeg(tmp_path, size=(300, 200)))))
    assert img.size == (300, 200)


def test_missing_image_raises(tmp_path):
    with pytest.raises(ImageError):
        to_data_url(str(tmp_path / "nope.jpeg"))
