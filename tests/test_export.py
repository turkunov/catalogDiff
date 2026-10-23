import pandas as pd
import pytest

from pseudolabel.export import ThresholdRequired, build_training_set, write_export


def test_export_requires_threshold(tmp_store):
    with pytest.raises(ThresholdRequired, match="Positive class threshold is not set"):
        build_training_set(tmp_store, "sofas", threshold=None)


def test_export_binarization_and_human_override(store_with_labels):
    df = build_training_set(store_with_labels, "sofas", threshold=4).set_index(["a_id", "b_id"])
    assert df.loc[("1", "2"), "label"] == 1
    assert df.loc[("1", "3"), "label"] == 0
    assert df.loc[("1", "4"), "label_source"] == "human" and df.loc[("1", "4"), "raw_class"] == 5
    assert df.loc[("1", "4"), "label"] == 1
    assert df.loc[("1", "5"), "label_source"] == "auto_negative" and df.loc[("1", "5"), "label"] == 0
    assert pd.isna(df.loc[("1", "5"), "raw_class"])
    assert (df["weight"] == 1.0).all()
    assert ("9", "9") not in df.index
    assert df.loc[("1", "2"), "a_image"] == "/img/a1.jpeg" and df.loc[("1", "2"), "encoder_version"] == "fake@1"


def test_export_class_weights(store_with_labels):
    df = build_training_set(store_with_labels, "sofas", threshold=4, class_weights={3: 0.5})
    assert df.set_index(["a_id", "b_id"]).loc[("1", "3"), "weight"] == 0.5


def test_export_include_legacy(store_with_labels):
    df = build_training_set(store_with_labels, "sofas", threshold=4, include_legacy=True).set_index(["a_id", "b_id"])
    assert df.loc[("9", "9"), "label_source"] == "legacy" and df.loc[("9", "9"), "label"] == 1


def test_write_export(store_with_labels, tmp_path):
    df = build_training_set(store_with_labels, "sofas", threshold=4)
    pq, csv = write_export(df, tmp_path / "out", "sofas_t4")
    assert pq.exists() and csv.exists() and len(pd.read_parquet(pq)) == len(df)
