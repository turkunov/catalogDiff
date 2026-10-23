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


def test_label_example_keeps_photos_without_segmentation(app_ctx_fake_vlm):
    from pseudolabel.profiles import load_profile

    ctx = app_ctx_fake_vlm
    ex = load_profile(ctx.settings.profiles_dir / "sofas.yaml").examples[0]
    v = label_example(ctx, "sofas", ex.name, threshold=4)
    assert (v.image_a, v.image_b) == (ex.a["image"], ex.b["image"])


def test_label_example_on_cut_out_photos(app_ctx_fake_vlm):
    from PIL import Image

    from test_segment import FakeSegmenter

    ctx = app_ctx_fake_vlm
    seg, seen = FakeSegmenter(), []
    ctx.segmenter_factory = lambda method, prompt: seen.append((method, prompt)) or seg
    plain = label_example(ctx, "sofas", ctx.first_example("sofas"), threshold=4)
    calls = len(ctx.vlm.calls)
    v = label_example(ctx, "sofas", ctx.first_example("sofas"), threshold=4, segmentation="birefnet")
    assert seen == [("birefnet", "sofa")] and seg.calls == 2
    assert "/segment/birefnet/" in v.image_a and Image.open(v.image_a).size != Image.open(plain.image_a).size
    assert len(ctx.vlm.calls) > calls  # cut-outs are labeled anew, not served from the original photos' cache
    assert v.cls == 5 and v.error is None


def test_custom_pair_with_sam3(app_ctx_fake_vlm, tmp_path):
    from test_segment import FakeSegmenter

    ctx = app_ctx_fake_vlm
    ctx.segmenter_factory = lambda method, prompt: FakeSegmenter()
    a = {"title": "диван 1", "description": "", "key_params": {}, "image": str(make_jpeg(tmp_path, size=(80, 80), name="s1.jpeg"))}
    b = {"title": "диван 2", "description": "", "key_params": {}, "image": str(make_jpeg(tmp_path, size=(80, 80), name="s2.jpeg"))}
    v = label_custom(ctx, "sofas", a, b, threshold=4, segmentation="sam3")
    assert "/segment/sam3/" in v.image_a and "/segment/sam3/" in v.image_b


def test_threshold_zero_or_empty_means_not_set():
    from pseudolabel.app.tab_pair import _threshold

    assert _threshold(0) is None and _threshold(None) is None and _threshold("") is None and _threshold(4) == 4


def test_summary_shows_confidence_and_flags_low(app_ctx_fake_vlm):
    from pseudolabel.app.tab_pair import PairView, _outputs

    sure = PairView({}, {}, {}, {}, "", 5, "positive", 3.0, None, confidence=0.98)
    unsure = PairView({}, {}, {}, {}, "", 4, "positive", 3.0, None, confidence=0.55)
    assert "confidence 0.98" in _outputs(sure)[0] and "Low confidence" not in _outputs(sure)[0]
    assert "Low confidence" in _outputs(unsure)[0]


def test_summary_says_when_the_answer_comes_from_cache():
    from pseudolabel.app.tab_pair import PairView, _outputs

    v = PairView({}, {}, {}, {}, "", 2, "negative", 57.9, None, confidence=1.0, cached=True)
    assert "from cache" in _outputs(v)[0]


def _ctx_with_eval_set(ctx, tmp_path):
    from dataclasses import replace

    from pseudolabel.profiles import dump_profile, load_profile
    from test_experiments import _eval_files

    pairs, labels = _eval_files(tmp_path)  # small_rows(): three pairs, reference classes 5, 2 and 3
    path = ctx.settings.profiles_dir / "sofas.yaml"
    p = load_profile(path)
    p.image_root = str(tmp_path / "images_small")
    path.write_text(dump_profile(p), encoding="utf-8")
    ctx.settings = replace(ctx.settings, demo_pairs_path=pairs, demo_labels_path=labels)
    return ctx


def test_demo_eval_pairs_by_class_with_previews(app_ctx_fake_vlm, tmp_path):
    from PIL import Image

    from pseudolabel.app.tab_pair import demo_eval_pairs

    ctx = _ctx_with_eval_set(app_ctx_fake_vlm, tmp_path)
    pairs = demo_eval_pairs(ctx, per_class=3)
    assert [p.reference for p in pairs] == [2, 3, 5] and [p.number for p in pairs] == [1, 2, 3]
    w, h = Image.open(pairs[0].preview).size
    assert w > h  # both photos side by side
    assert demo_eval_pairs(ctx, per_class=3) == pairs  # the same selection every time


def test_demo_eval_pairs_empty_without_files(app_ctx_fake_vlm, tmp_path):
    from dataclasses import replace

    from pseudolabel.app.tab_pair import demo_eval_pairs

    app_ctx_fake_vlm.settings = replace(app_ctx_fake_vlm.settings, demo_pairs_path=tmp_path / "none.csv")
    assert demo_eval_pairs(app_ctx_fake_vlm) == []


def test_label_eval_pair_runs_fresh_and_shows_reference(app_ctx_fake_vlm, tmp_path):
    from pseudolabel.app.tab_pair import _outputs, demo_eval_pairs, label_eval_pair

    ctx = _ctx_with_eval_set(app_ctx_fake_vlm, tmp_path)
    pair = next(p for p in demo_eval_pairs(ctx) if p.reference == 5)
    label_eval_pair(ctx, pair, threshold=4)
    calls = len(ctx.vlm.calls)
    v = label_eval_pair(ctx, pair, threshold=4)
    assert len(ctx.vlm.calls) >= calls + 3  # photo steps of both listings and the match, not served from cache
    assert v.reference == 5 and not v.cached and "reference 5" in _outputs(v)[0]
