"""Shared test helpers (importable from tests; fixtures live in conftest.py)."""
import json
from pathlib import Path

import pandas as pd
from PIL import Image

from pseudolabel.profiles import Profile, load_profile

ROOT = Path(__file__).parents[1]
PROFILES = ROOT / "profiles"


def sofa_profile_with_root(tmp_path: Path) -> Profile:
    p = load_profile(PROFILES / "sofas.yaml")
    p.image_root = str(tmp_path / "images_small")
    return p


def make_jpeg(tmp_path: Path, size=(64, 48), color=(200, 120, 40), name: str | None = None) -> Path:
    path = Path(tmp_path) / (name or f"img_{size[0]}x{size[1]}_{color[0]}.jpeg")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path, "JPEG")
    return path


def small_rows(multiline: bool = False) -> list[dict]:
    kp = lambda m, f, c: json.dumps({"material_obivki": m, "forma": f, "osnovnoy_cvet": c}, ensure_ascii=False)  # noqa: E731
    desc1 = 'строка1\r\nстрока2 с "кавычками"' if multiline else "диван от производителя"
    return [
        dict(q_item_id=2272175750151, d_item_id=380288750116, q_title="диван прямой раскладной",
             d_title="диван раскладной бежевый", q_key_params=kp("велюр", "прямой", "бежевый"),
             d_key_params=kp(None, "прямой", "бежевый"), q_description=desc1, d_description="",
             q_image_url="https://x/1", d_image_url="https://x/2", q_image_id=50374145674,
             d_image_id=51504433441, label=5.0, img_cossim=0.8),
        dict(q_item_id=2272175750151, d_item_id=300000000002, q_title="диван прямой раскладной",
             d_title="угловой диван синий", q_key_params=kp("велюр", "прямой", "бежевый"),
             d_key_params=kp("рогожка", "угловой", "синий"), q_description=desc1, d_description="синий",
             q_image_url="https://x/1", d_image_url="https://x/3", q_image_id=50374145674,
             d_image_id=51500000002, label=2.0, img_cossim=0.5),
        dict(q_item_id=2300000000003, d_item_id=300000000003, q_title="диван угловой",
             d_title="диван угловой серый", q_key_params=kp("рогожка", "угловой", "серый"),
             d_key_params=kp("рогожка", "угловой", "серый"), q_description="угловой", d_description="серый",
             q_image_url="https://x/4", d_image_url="https://x/5", q_image_id=50000000003,
             d_image_id=51500000003, label=3.0, img_cossim=0.7),
    ]


def write_small_csv(tmp_path: Path, multiline: bool = False, d_categories: list[str] | None = None) -> Path:
    df = pd.DataFrame(small_rows(multiline))
    if d_categories is not None:
        df["d_cat"] = d_categories
    path = Path(tmp_path) / "pairs.csv"
    df.to_csv(path, index=False)
    return path
