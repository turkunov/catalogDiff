import json

import pandas as pd
import pytest

from fakes import FakeVLM, route_by_prompt
from helpers import make_jpeg, sofa_profile_with_root, small_rows


def test_quadratic_kappa_perfect_and_reversed():
    from pseudolabel.metrics import quadratic_kappa

    assert quadratic_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == pytest.approx(1.0)
    assert quadratic_kappa([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) == pytest.approx(-1.0)


def test_mcnemar_identical_is_one_and_strong_difference_is_small():
    from pseudolabel.metrics import mcnemar_p

    assert mcnemar_p([True, False, True], [True, False, True]) == 1.0
    assert mcnemar_p([False] * 20, [True] * 20) < 1e-5


def test_bootstrap_diff_contains_true_difference():
    from pseudolabel.metrics import bootstrap_diff

    lo, hi = bootstrap_diff([False] * 50 + [True] * 50, [True] * 100)
    assert lo <= 0.5 <= hi and lo > 0


def test_auroc_perfect_and_tied():
    from pseudolabel.metrics import auroc

    assert auroc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) == 1.0
    assert auroc([0.5, 0.5], [True, False]) == 0.5
    assert auroc([0.5, 0.6], [True, True]) is None


def _eval_files(tmp_path):
    rows = small_rows()
    for r in rows:
        make_jpeg(tmp_path / "images_small" / "avito_images", name=f"{r['q_image_id']}.jpeg")
        make_jpeg(tmp_path / "images_small" / "ozon_images", name=f"{r['d_image_id']}.jpeg")
    pairs = tmp_path / "pairs.csv"
    pd.DataFrame(rows).to_csv(pairs, index=False)
    labels = tmp_path / "labels.csv"
    pd.DataFrame([dict(a_id=str(r["q_item_id"]), b_id=str(r["d_item_id"]), **{"class": int(r["label"])}) for r in rows]
                 ).to_csv(labels, index=False)
    return pairs, labels


def test_run_variant_writes_jsonl_and_metrics_and_resumes(tmp_path, tmp_store):
    from pseudolabel.experiments import Variant, load_eval, run_variant

    profile = sofa_profile_with_root(tmp_path)
    pairs, labels = _eval_files(tmp_path)
    universe, human = load_eval(pairs, labels, profile)
    vlm = FakeVLM(route_by_prompt(match='{"reasoning": "ok", "class": 5}'))
    v = Variant(name="base", base_url="http://x/v1", model="fake-vlm", concurrency=2)
    m = run_variant(v, profile, universe, human, tmp_store, tmp_path / "out", threshold=4, vlm=vlm)
    rows = [json.loads(line) for line in (tmp_path / "out" / "base.jsonl").read_text().splitlines()]
    # small_rows(): three pairs with reference classes 5, 2 and 3
    assert len(rows) == 3 and {r["vlm"] for r in rows} == {5} and sorted(r["human"] for r in rows) == [2, 3, 5]
    assert m["n"] == 3 and m["exact"] == pytest.approx(1 / 3) and m["precision"] == pytest.approx(1 / 3)
    assert m["recall"] == 1.0 and m["failed"] == 0
    assert json.loads((tmp_path / "out" / "base.metrics.json").read_text())["exact"] == pytest.approx(1 / 3)
    n_calls = len(vlm.calls)
    run_variant(v, profile, universe, human, tmp_store, tmp_path / "out", threshold=4, vlm=vlm)
    assert len(vlm.calls) == n_calls  # pairs already in the JSONL are not labeled again


def test_run_variant_applies_profile_patch_and_options(tmp_path, tmp_store):
    from pseudolabel.experiments import Variant, load_eval, run_variant

    profile = sofa_profile_with_root(tmp_path)
    pairs, labels = _eval_files(tmp_path)
    universe, human = load_eval(pairs, labels, profile)
    vlm = FakeVLM(route_by_prompt(match='{"reasoning": "ok", "class": 2}'))
    v = Variant(name="one_step", base_url="http://x/v1", model="fake-vlm", profile_patch={"thinking": False},
                options={"visual_step": False, "text_step": False})
    run_variant(v, profile, universe, human, tmp_store, tmp_path / "out", threshold=4, vlm=vlm)
    assert len(vlm.calls) == 3 and all(c["gen"]["thinking"] is False for c in vlm.calls)


def test_skeptic_variant_reads_source_results(tmp_path, tmp_store):
    from pseudolabel.experiments import Variant, load_eval, run_variant
    from pseudolabel.prompts import SKEPTIC_MARKER

    profile = sofa_profile_with_root(tmp_path)
    pairs, labels = _eval_files(tmp_path)
    universe, human = load_eval(pairs, labels, profile)
    run_variant(Variant(name="base", base_url="x", model="fake-vlm"), profile, universe, human, tmp_store,
                tmp_path / "out", threshold=4, vlm=FakeVLM(route_by_prompt(match='{"reasoning": "ok", "class": 5}')))
    sk = FakeVLM(lambda c, g, j: '{"differences": "ножки", "class": 3}')
    m = run_variant(Variant(name="skeptic", base_url="x", model="fake-vlm", skeptic_of="base"), profile, universe, human,
                    tmp_store, tmp_path / "out", threshold=4, vlm=sk)
    assert len(sk.calls) == 3 and all(c["content"][-1]["text"].startswith(SKEPTIC_MARKER) for c in sk.calls)
    assert m["exact"] == pytest.approx(1 / 3) and m["recall"] == 0.0


def test_report_marks_baseline_and_deltas(tmp_path, tmp_store):
    from pseudolabel.experiments import Variant, load_eval, report, run_variant

    profile = sofa_profile_with_root(tmp_path)
    pairs, labels = _eval_files(tmp_path)
    universe, human = load_eval(pairs, labels, profile)
    for name, cls in (("base", 5), ("photo_only", 2)):
        run_variant(Variant(name=name, base_url="x", model="fake-vlm"), profile, universe, human, tmp_store,
                    tmp_path / "out", threshold=4, vlm=FakeVLM(route_by_prompt(match=f'{{"reasoning": "ok", "class": {cls}}}')))
    text = report(tmp_path / "out", baseline="base", threshold=4)
    assert "| base |" in text and "| photo_only |" in text and "baseline" in text
    assert "+0.00" in text or "0.00" in text


def test_cli_ablate_report_writes_markdown(tmp_path, tmp_store, monkeypatch):
    import yaml

    from pseudolabel.cli import main
    from pseudolabel.experiments import Variant, load_eval, run_variant

    profile = sofa_profile_with_root(tmp_path)
    pairs, labels = _eval_files(tmp_path)
    universe, human = load_eval(pairs, labels, profile)
    run_variant(Variant(name="base", base_url="x", model="fake-vlm"), profile, universe, human, tmp_store,
                tmp_path / "out", threshold=4, vlm=FakeVLM(route_by_prompt()))
    cfg = tmp_path / "abl.yaml"
    cfg.write_text(yaml.safe_dump({"profile": "sofas", "eval_pairs": str(pairs), "eval_labels": str(labels), "threshold": 4,
                                   "out_dir": str(tmp_path / "out"), "db": str(tmp_path / "abl.sqlite"), "variants": []}))
    monkeypatch.setenv("PL_WORK_DIR", str(tmp_path / "work"))
    assert main(["ablate-report", "--config", str(cfg)]) == 0
    assert "| base |" in (tmp_path / "out" / "report.md").read_text()
