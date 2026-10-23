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
