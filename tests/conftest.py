import pytest

from helpers import PROFILES, make_jpeg
from pseudolabel.data import Item
from pseudolabel.profiles import load_profile


@pytest.fixture
def sofa_profile():
    return load_profile(PROFILES / "sofas.yaml")


@pytest.fixture
def tmp_store(tmp_path):
    from pseudolabel.store import Store

    return Store(tmp_path / "labels.sqlite")


@pytest.fixture
def sofa_item(tmp_path):
    return Item("2272175750151", "a", "диван прямой раскладной роденго", "диван от производителя, механизм еврокнижка",
                {"material_obivki": "велюр", "forma": None, "osnovnoy_cvet": "серый"},
                str(make_jpeg(tmp_path, name="a.jpeg")))


@pytest.fixture
def sofa_item_b(tmp_path):
    return Item("380288750116", "b", "диван кухонный роденго синий", "рогожка",
                {"material_obivki": "рогожка", "forma": "прямой", "osnovnoy_cvet": "синий"},
                str(make_jpeg(tmp_path, color=(20, 40, 200), name="b.jpeg")))


@pytest.fixture
def sofa_item_full_params(tmp_path):
    return Item("2300000000003", "a", "диван угловой", "угловой серый",
                {"material_obivki": "рогожка", "forma": "угловой", "osnovnoy_cvet": "серый"},
                str(make_jpeg(tmp_path, name="full.jpeg")))


@pytest.fixture
def missing_image_item(tmp_path):
    return Item("300000000009", "b", "диван", "", {"material_obivki": "велюр", "forma": "прямой", "osnovnoy_cvet": "серый"},
                str(tmp_path / "missing.jpeg"))


@pytest.fixture
def labeler_with_fake(tmp_store, sofa_profile):
    from fakes import FakeVLM
    from pseudolabel.labeler import Labeler

    def make(handler, profile=None):
        vlm = FakeVLM(handler)
        return Labeler(vlm, tmp_store, profile or sofa_profile, 1, retry_sleep_s=0), vlm

    return make


@pytest.fixture
def fake_openai_client():
    from fakes import FakeOpenAIClient

    return FakeOpenAIClient()


@pytest.fixture
def items_with_images(tmp_path):
    return [Item(str(i), "b", f"item {i}", "", {}, str(make_jpeg(tmp_path, name=f"it{i}.jpeg"))) for i in range(3)]


@pytest.fixture
def tiny_universe(tmp_path):
    from helpers import build_tiny_universe

    return build_tiny_universe(tmp_path)


@pytest.fixture
def tiny_embeddings(tiny_universe):
    from helpers import embeddings_for, tiny_vectors

    return embeddings_for(tiny_universe, tiny_vectors(tiny_universe))


@pytest.fixture
def store_with_labels(tmp_store, tmp_path):
    from pseudolabel.store import LabelRow

    s = tmp_store
    s.upsert_items("sofas", [Item(i, side, f"t{i}", "", {}, f"/img/{side}{i}.jpeg")
                             for side, ids in (("a", ["1", "9"]), ("b", ["2", "3", "4", "5", "9"])) for i in ids])
    it = s.create_iteration("sofas", 1, {"labeler_model": "m"}, "fake@1")
    s.add_iteration_pairs(it, [
        dict(a_id="1", b_id="2", origin="top_1_10", rank=1, cosine=0.9, selected_for_vlm=True),
        dict(a_id="1", b_id="3", origin="top_11_100", rank=20, cosine=0.6, selected_for_vlm=True),
        dict(a_id="1", b_id="4", origin="top_1_10", rank=2, cosine=0.8, selected_for_vlm=True),
        dict(a_id="1", b_id="5", origin="easy_negative", rank=900, cosine=0.1, selected_for_vlm=False),
    ])
    for b, cls in (("2", 4), ("3", 3), ("4", 2)):
        s.put_label(LabelRow("sofas", "1", b, "vlm", cls, model="m", profile_version=1))
    s.put_label(LabelRow("sofas", "1", "4", "human", 5))
    s.import_legacy("sofas", {("9", "9"): 5})
    return s


def _make_env(tmp_path, tmp_store, sofa_profile, universe, match='{"reasoning": "r", "class": 2}'):
    from types import SimpleNamespace

    from fakes import FakeEncoder, FakeVLM, prompt_text, route_by_prompt
    from helpers import tiny_vectors
    from pseudolabel.iteration import IterationConfig
    from pseudolabel.labeler import Labeler
    from pseudolabel.prompts import MATCH_MARKER

    version = tmp_store.register_profile_version(sofa_profile)
    vlm = FakeVLM(route_by_prompt(match=match))
    labeler = Labeler(vlm, tmp_store, sofa_profile, version, retry_sleep_s=0)

    def cfg(**kw):
        return IterationConfig(profile_name="sofas", pairs_path="tiny.csv", mode=kw.pop("mode", "pairs_file"),
                               labeler_model="fake-vlm", **kw)

    return SimpleNamespace(
        store=tmp_store, profile=sofa_profile, universe=universe, encoder=FakeEncoder(tiny_vectors(universe)),
        labeler=labeler, vlm=vlm, tmp=tmp_path / "cache", cfg=cfg,
        match_calls=lambda: sum(prompt_text(c["content"]).startswith(MATCH_MARKER) for c in vlm.calls))


@pytest.fixture
def env(tmp_path, tmp_store, sofa_profile, tiny_universe):
    return _make_env(tmp_path, tmp_store, sofa_profile, tiny_universe)


@pytest.fixture
def env_with_one_broken_image(tmp_path, tmp_store, sofa_profile, tiny_universe):
    from pathlib import Path

    for b in list(tiny_universe.items_b.values())[::2]:
        Path(b.image).unlink()
    return _make_env(tmp_path, tmp_store, sofa_profile, tiny_universe)


def _universe_by_class(counts: dict[int, int], n_q: int = 120):
    from pseudolabel.data import Universe

    items_a, items_b, pairs, legacy = {}, {}, [], {}
    i = 0
    for cls, n in counts.items():
        for _ in range(n):
            a, b = f"q{i % n_q}", f"d{i}"
            items_a.setdefault(a, Item(a, "a", a, "", {}, f"/img/{a}.jpeg"))
            items_b[b] = Item(b, "b", b, "", {}, f"/img/{b}.jpeg")
            pairs.append((a, b))
            legacy[(a, b)] = cls
            i += 1
    return Universe(items_a, items_b, pairs, legacy)


@pytest.fixture
def universe_5_classes():
    return _universe_by_class({c: 50 for c in range(1, 6)})


@pytest.fixture
def universe_5_classes_few_ones():
    return _universe_by_class({1: 3, 2: 50, 3: 50, 4: 50, 5: 50})


@pytest.fixture
def tmp_settings_env(tmp_path, monkeypatch):
    from helpers import PROFILES

    monkeypatch.setenv("PL_WORK_DIR", str(tmp_path / "work"))
    monkeypatch.setenv("PL_PROFILES_DIR", str(PROFILES))
    return tmp_path


def _app_ctx(tmp_path, handler=None):
    import shutil

    from fakes import FakeEncoder, FakeVLM, route_by_prompt
    from helpers import PROFILES
    from pseudolabel.app.context import AppContext
    from pseudolabel.profiles import Example, dump_profile, load_profile
    from pseudolabel.settings import load_settings
    from pseudolabel.store import Store

    profiles = tmp_path / "profiles"
    shutil.copytree(PROFILES, profiles)
    sofas = load_profile(profiles / "sofas.yaml")
    a_img, b_img = make_jpeg(tmp_path, name="ex_a.jpeg"), make_jpeg(tmp_path, color=(9, 9, 9), name="ex_b.jpeg")
    sofas.examples = [Example("same sofa", 5, {"title": "диван роденго", "description": "", "key_params": {}, "image": str(a_img)},
                              {"title": "диван кухонный роденго", "description": "", "key_params": {}, "image": str(b_img)})]
    (profiles / "sofas.yaml").write_text(dump_profile(sofas), encoding="utf-8")
    settings = load_settings({"PL_WORK_DIR": str(tmp_path / "work"), "PL_PROFILES_DIR": str(profiles)})
    vlm = FakeVLM(handler or route_by_prompt(match='{"reasoning": "same model", "class": 5}'))
    ctx = AppContext(settings, Store(settings.db_path), vlm_factory=lambda model=None: vlm,
                     encoder_factory=lambda checkpoint: FakeEncoder({}))
    ctx.first_example = lambda name: load_profile(profiles / f"{name}.yaml").examples[0].name
    ctx.vlm = vlm
    return ctx


@pytest.fixture
def app_ctx(tmp_path):
    return _app_ctx(tmp_path)


@pytest.fixture
def app_ctx_fake_vlm(tmp_path):
    return _app_ctx(tmp_path)


@pytest.fixture
def app_ctx_with_iteration(tmp_path, tiny_universe):
    from fakes import FakeEncoder, route_by_prompt
    from helpers import tiny_vectors
    from pseudolabel.iteration import IterationConfig, finalize_iteration, plan_iteration, run_iteration

    ctx = _app_ctx(tmp_path, route_by_prompt(match='{"reasoning": "r", "class": 2}'))
    profile = ctx.profile("sofas")
    encoder = FakeEncoder(tiny_vectors(tiny_universe))
    cfg = IterationConfig(profile_name="sofas", pairs_path="tiny.csv", mode="pairs_file", labeler_model="fake-vlm",
                          budget=8, positive_threshold=4)
    it = plan_iteration(cfg, ctx.store, profile, tiny_universe, encoder, tmp_path / "cache")
    run_iteration(it, ctx.store, ctx.labeler(profile, ctx.store.get_iteration(it)["profile_version"]))
    finalize_iteration(it, ctx.store)
    ctx.iteration_id = it
    ctx.disagreement_queue = f"disagreements:{it}"
    return ctx
