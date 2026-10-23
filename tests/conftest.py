import pytest

from helpers import PROFILES
from pseudolabel.profiles import load_profile


@pytest.fixture
def sofa_profile():
    return load_profile(PROFILES / "sofas.yaml")


@pytest.fixture
def tmp_store(tmp_path):
    from pseudolabel.store import Store

    return Store(tmp_path / "labels.sqlite")


@pytest.fixture
def sofa_item():
    from pseudolabel.data import Item

    return Item("2272175750151", "a", "диван прямой раскладной роденго", "диван от производителя, механизм еврокнижка",
                {"material_obivki": "велюр", "forma": None, "osnovnoy_cvet": "серый"}, "/nonexistent/a.jpeg")


@pytest.fixture
def sofa_item_b():
    from pseudolabel.data import Item

    return Item("380288750116", "b", "диван кухонный роденго синий", "рогожка",
                {"material_obivki": "рогожка", "forma": "прямой", "osnovnoy_cvet": "синий"}, "/nonexistent/b.jpeg")
