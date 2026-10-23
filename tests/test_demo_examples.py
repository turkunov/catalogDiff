import pytest

from helpers import PROFILES, ROOT
from pseudolabel.profiles import load_profile


@pytest.mark.parametrize("name", ["sofas"])
def test_demo_examples_present(name):
    p = load_profile(PROFILES / f"{name}.yaml")
    assert len(p.examples) == 3 and max(e.expected_class for e in p.examples) == 5
    assert min(e.expected_class for e in p.examples) <= 2
    for e in p.examples:
        for side in (e.a, e.b):
            assert (ROOT / side["image"]).exists() or side["image"].startswith("/data/")
