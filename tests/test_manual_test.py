from collections import Counter

import pandas as pd
import pytest

from fakes import route_by_prompt
from pseudolabel.manual_test import import_human_labels, load_manual_test_csv, run_manual_test, select_manual_test
from pseudolabel.store import LabelRow


def test_select_manual_test_balanced_unique_q(universe_5_classes):
    rows, shortfall = select_manual_test(universe_5_classes, per_class=40, seed=0)
    assert len(rows) == 200 and shortfall == {}
    assert Counter(c for *_, c in rows) == {1: 40, 2: 40, 3: 40, 4: 40, 5: 40}
    assert len({a for a, _, _ in rows}) >= 120


def test_select_manual_test_shortfall(universe_5_classes_few_ones):
    rows, shortfall = select_manual_test(universe_5_classes_few_ones, per_class=40)
    assert shortfall == {1: 37} and len(rows) == 163


def test_load_manual_test_csv_from_labeling_tool(tmp_path):
    p = tmp_path / "set.csv"
    pd.DataFrame({"idx": [0, 1], "q_item_id": [11, 12], "d_item_id": [21, 22], "legacy_class": [5, 1]}).to_csv(p, index=False)
    assert load_manual_test_csv(p) == [("11", "21", 5), ("12", "22", 1)]


def test_import_human_labels_validates(tmp_store, tmp_path):
    p = tmp_path / "h.csv"
    p.write_text("a_id,b_id,class\n1,2,5\n1,3,7\n")
    with pytest.raises(ValueError, match="row 2"):
        import_human_labels(tmp_store, "sofas", p)
    assert tmp_store.latest_human_label("sofas", "1", "2") is None


def test_import_human_labels_from_labeling_tool_export(tmp_store, tmp_path):
    p = tmp_path / "h.csv"
    p.write_text("a_id,b_id,class,comment,updated_at\n1,2,5,похоже,2026-10-23T01:00:00+00:00\n1,3,2,,2026-10-23T01:01:00+00:00\n")
    assert import_human_labels(tmp_store, "sofas", p) == 2
    label = tmp_store.latest_human_label("sofas", "1", "2")
    assert label.cls == 5 and label.comment == "похоже"


def test_run_manual_test_metrics(labeler_with_fake, tmp_store, tiny_universe):
    lab, _ = labeler_with_fake(route_by_prompt(match='{"reasoning": "r", "class": 4}'))
    pairs = tiny_universe.pairs[:3]
    tmp_store.set_manual_test("sofas", [(a, b, 3) for a, b in pairs])
    for (a, b), human in zip(pairs, [4, 4, 2]):
        tmp_store.put_label(LabelRow("sofas", a, b, "human", human))
    m = run_manual_test(tmp_store, lab, tiny_universe, "sofas", threshold=4)
    assert m["n"] == 3 and m["exact"] == pytest.approx(2 / 3) and m["mean_latency_s"] >= 0
