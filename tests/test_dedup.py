import shutil

import numpy as np
from PIL import Image

from pseudolabel.data import Item


def _noise(path, seed, size=(160, 120)):
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, (12, 16, 3), dtype=np.uint8)  # blocky picture, robust to resizing
    Image.fromarray(small).resize(size, Image.NEAREST).save(path, "JPEG", quality=92)
    return str(path)


def test_phash_same_image_zero_and_resized_close(tmp_path):
    from pseudolabel.dedup import photo_distance

    a = _noise(tmp_path / "a.jpeg", 1)
    Image.open(a).resize((80, 60)).save(tmp_path / "small.jpeg", "JPEG", quality=80)
    assert photo_distance(a, a) == 0
    assert photo_distance(a, str(tmp_path / "small.jpeg")) <= 4


def test_phash_different_far_and_unreadable_none(tmp_path):
    from pseudolabel.dedup import photo_distance

    assert photo_distance(_noise(tmp_path / "a.jpeg", 1), _noise(tmp_path / "b.jpeg", 2)) > 16
    assert photo_distance(_noise(tmp_path / "a.jpeg", 1), str(tmp_path / "missing.jpeg")) is None


def _twin_items(tmp_path):
    a_img = _noise(tmp_path / "a.jpeg", 7)
    shutil.copy(a_img, tmp_path / "b.jpeg")
    kp = {"material_obivki": "велюр", "forma": "прямой", "osnovnoy_cvet": "серый"}
    return (Item("1", "a", "диван", "", kp, a_img), Item("2", "b", "диван", "", kp, str(tmp_path / "b.jpeg")))


def test_label_pair_same_photo_skips_vlm(labeler_with_fake, tmp_path, tmp_store):
    from pseudolabel.labeler import LabelerOptions

    a, b = _twin_items(tmp_path)
    lab, vlm = labeler_with_fake(lambda c, g, j: "{}", options=LabelerOptions(photo_dup_max_distance=6, photo_dup_check_share=0))
    r = lab.label_pair(a, b)
    assert r.cls == 5 and r.photo_dup_distance == 0 and vlm.calls == [] and r.confidence == 1.0
    stored = tmp_store.get_vlm_label("sofas", "1", "2", "fake-vlm", 1)
    assert stored.cls == 5 and "same photo" in stored.reasoning and stored.gen_params["photo_dup_distance"] == 0


def test_same_photo_check_share_still_asks_vlm(labeler_with_fake, tmp_path):
    from fakes import route_by_prompt
    from pseudolabel.labeler import LabelerOptions

    a, b = _twin_items(tmp_path)
    lab, vlm = labeler_with_fake(route_by_prompt(match='{"reasoning": "ok", "class": 4}'),
                                 options=LabelerOptions(photo_dup_max_distance=6, photo_dup_check_share=1.0))
    r = lab.label_pair(a, b)
    assert r.cls == 4 and r.photo_dup_distance == 0 and vlm.calls


def test_iteration_config_turns_on_the_shortcut():
    from dataclasses import asdict

    from pseudolabel.iteration import IterationConfig, labeler_options

    cfg = IterationConfig(profile_name="sofas", pairs_path="x", mode="pairs_file", photo_dup_max_distance=5)
    o = labeler_options(asdict(cfg))  # iterations keep their config as a dict
    assert o.photo_dup_max_distance == 5 and o.photo_dup_check_share == 0.05
    assert labeler_options({"profile_name": "sofas"}).photo_dup_max_distance is None  # configs saved before the option
