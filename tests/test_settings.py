from pseudolabel.settings import load_settings


def test_defaults():
    s = load_settings({})
    assert s.vllm_model == "Qwen/Qwen3.5-9B"
    assert str(s.db_path).endswith("labels.sqlite")
    assert s.vlm_concurrency == 4


def test_override():
    assert load_settings({"PL_VLM_CONCURRENCY": "2"}).vlm_concurrency == 2


def test_demo_eval_pairs_defaults():
    s = load_settings({})
    assert str(s.demo_pairs_path) == "/data/eval160/pairs160.csv" and str(s.demo_labels_path) == "/data/eval160/labels160.csv"
    assert s.demo_profile == "sofas"
