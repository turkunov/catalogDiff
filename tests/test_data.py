from helpers import sofa_profile_with_root, write_small_csv
from pseudolabel.data import load_universe


def test_load_universe_small_like(tmp_path):
    u = load_universe(sofa_profile_with_root(tmp_path), write_small_csv(tmp_path))
    assert len(u.pairs) == 3 and len(u.items_a) == 2
    item = u.items_b["380288750116"]
    assert item.key_params == {"material_obivki": None, "forma": "прямой", "osnovnoy_cvet": "бежевый"}
    assert item.image == str(tmp_path / "images_small" / "ozon_images" / "51504433441.jpeg")
    assert u.legacy_labels[("2272175750151", "380288750116")] == 5


def test_load_universe_multiline_descriptions(tmp_path):
    u = load_universe(sofa_profile_with_root(tmp_path), write_small_csv(tmp_path, multiline=True))
    assert len(u.pairs) == 3
    assert "строка2" in u.items_a["2272175750151"].description


def test_load_universe_category_filter(tmp_path):
    p = sofa_profile_with_root(tmp_path)
    p.input_b.category = "d_cat"
    p.categories_b = ["15018"]
    u = load_universe(p, write_small_csv(tmp_path, d_categories=["15018", "35623", "15018"]))
    assert len(u.pairs) == 2 and len(u.items_b) == 2
