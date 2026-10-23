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
