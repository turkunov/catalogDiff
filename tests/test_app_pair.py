from helpers import make_jpeg
from pseudolabel.app.tab_pair import label_custom, label_example


def test_label_example_without_threshold(app_ctx_fake_vlm):
    v = label_example(app_ctx_fake_vlm, "sofas", app_ctx_fake_vlm.first_example("sofas"), threshold=None)
    assert v.cls == 5 and v.binary == "Positive class threshold is not set" and v.error is None
    assert v.visual_a == {"color": "серый"}


def test_label_example_with_threshold(app_ctx_fake_vlm):
    v = label_example(app_ctx_fake_vlm, "sofas", app_ctx_fake_vlm.first_example("sofas"), threshold=4)
    assert v.binary == "positive"


def test_label_custom_pairs_do_not_share_cache(app_ctx_fake_vlm, tmp_path):
    a1 = {"title": "диван 1", "description": "", "key_params": {}, "image": str(make_jpeg(tmp_path, name="c1.jpeg"))}
    b1 = {"title": "диван 2", "description": "", "key_params": {}, "image": str(make_jpeg(tmp_path, name="c2.jpeg"))}
    b2 = {**b1, "title": "пуф", "image": str(make_jpeg(tmp_path, color=(1, 2, 3), name="c3.jpeg"))}
    label_custom(app_ctx_fake_vlm, "sofas", a1, b1, threshold=4)
    calls = len(app_ctx_fake_vlm.vlm.calls)
    label_custom(app_ctx_fake_vlm, "sofas", a1, b2, threshold=4)
    assert len(app_ctx_fake_vlm.vlm.calls) > calls
