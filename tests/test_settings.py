from pseudolabel.settings import load_settings


def test_defaults():
    s = load_settings({})
    assert s.vllm_model == "Qwen/Qwen3.5-9B"
    assert str(s.db_path).endswith("labels.sqlite")
    assert s.vlm_concurrency == 4


def test_override():
    assert load_settings({"PL_VLM_CONCURRENCY": "2"}).vlm_concurrency == 2
