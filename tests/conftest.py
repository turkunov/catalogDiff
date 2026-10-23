import pytest

from helpers import PROFILES
from pseudolabel.profiles import load_profile


@pytest.fixture
def sofa_profile():
    return load_profile(PROFILES / "sofas.yaml")
