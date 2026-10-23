import pandas as pd
import pytest

from pseudolabel.metrics import ThresholdRequired, is_disagreement, labeler_vs_human, origin_stats, require_threshold


def test_labeler_vs_human():
    df = pd.DataFrame({"human": [5, 4, 2, 3], "vlm": [5, 3, 2, None]})
    m = labeler_vs_human(df, threshold=4)
    assert m["n"] == 3 and m["exact"] == pytest.approx(2 / 3) and m["within1"] == pytest.approx(1.0)
    assert m["confusion"][3][2] == 1  # human 4 -> vlm 3
    assert m["vlm_failed"] == 1
    assert m["precision"] == pytest.approx(1.0) and m["recall"] == pytest.approx(0.5)


def test_labeler_vs_human_without_threshold():
    m = labeler_vs_human(pd.DataFrame({"human": [1], "vlm": [1]}))
    assert "precision" not in m and m["exact"] == 1.0


def test_require_threshold():
    with pytest.raises(ThresholdRequired, match="^Positive class threshold is not set$"):
        require_threshold(None)
    with pytest.raises(ValueError):
        require_threshold(1)
    assert require_threshold(4) == 4


def test_disagreement_definition():
    assert is_disagreement("top_1_10", 3, 4) and not is_disagreement("top_1_10", 4, 4)
    assert is_disagreement("title", 5, 4) and is_disagreement("top_101_1000", 4, 4)
    assert not is_disagreement("random", 5, 4) and not is_disagreement("top_1_10", None, 4)


def test_origin_stats():
    df = pd.DataFrame({"origin": ["top_1_10", "top_1_10", "top_11_100", "top_11_100"], "cls": [2, 5, 4, None]})
    s = origin_stats(df, threshold=4)
    assert s["top_1_10"]["n"] == 2 and s["top_1_10"]["class_counts"]["2"] == 1
    assert s["top_1_10"]["disagreements"] == 1 and s["top_1_10"]["positives"] == 1
    assert s["top_11_100"]["errors"] == 1 and s["top_11_100"]["disagreements"] == 1
    assert "positives" not in origin_stats(df, None)["top_1_10"]
