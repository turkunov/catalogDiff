"""Pairs files → items of both sides, pairs and legacy labels, as described by a profile."""
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pseudolabel.profiles import Profile, SideInput


@dataclass(frozen=True)
class Item:
    id: str
    side: str
    title: str
    description: str
    key_params: dict[str, str | None]
    image: str


@dataclass
class Universe:
    items_a: dict[str, Item]
    items_b: dict[str, Item]
    pairs: list[tuple[str, str]]
    legacy_labels: dict[tuple[str, str], int]


def _text(v) -> str:
    return "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)


def _key_params(raw, keys: list[str]) -> dict[str, str | None]:
    parsed = raw if isinstance(raw, dict) else {}
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {}
    out = {}
    for k in keys:
        v = parsed.get(k) if isinstance(parsed, dict) else None
        out[k] = None if v is None or str(v).strip() == "" else str(v)
    return out


def _resolve_image(image: str, root: str | Path) -> str:
    if not image or image.startswith(("http://", "https://", "data:", "/")) or not str(root):
        return image
    return str(Path(root) / image)


def _item(row: dict, side: str, spec: SideInput, keys: list[str], root: str) -> Item:
    return Item(
        id=str(row[spec.id]),
        side=side,
        title=_text(row.get(spec.title)),
        description=_text(row.get(spec.description)),
        key_params=_key_params(row.get(spec.key_params) if spec.key_params else None, keys),
        image=_resolve_image(spec.image.format(**row), root),
    )


def _read(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, lineterminator="\n")


def load_universe(profile: Profile, pairs_path: Path) -> Universe:
    df = _read(pairs_path)
    a, b = profile.input_a, profile.input_b
    for spec, allowed in ((a, profile.categories_a), (b, profile.categories_b)):
        if spec.category and allowed:
            df = df[df[spec.category].astype(str).isin(set(allowed))]
    df = df.drop_duplicates([a.id, b.id], keep="first")
    keys = [k.key for k in profile.key_params]
    items_a: dict[str, Item] = {}
    items_b: dict[str, Item] = {}
    pairs: list[tuple[str, str]] = []
    legacy: dict[tuple[str, str], int] = {}
    label_col = profile.legacy_label_column if profile.legacy_label_column in df.columns else None
    for row in df.to_dict("records"):
        ia = items_a.setdefault(str(row[a.id]), _item(row, "a", a, keys, profile.image_root))
        ib = items_b.setdefault(str(row[b.id]), _item(row, "b", b, keys, profile.image_root))
        pairs.append((ia.id, ib.id))
        if label_col and not pd.isna(row[label_col]):
            legacy[(ia.id, ib.id)] = int(row[label_col])
    return Universe(items_a=items_a, items_b=items_b, pairs=pairs, legacy_labels=legacy)


def item_from_example(side: str, d: dict, base_dir: Path) -> Item:
    return Item(
        id=str(d.get("id", f"example-{side}")),
        side=side,
        title=_text(d.get("title")),
        description=_text(d.get("description")),
        key_params={k: (None if v in (None, "") else str(v)) for k, v in (d.get("key_params") or {}).items()},
        image=_resolve_image(str(d.get("image", "")), base_dir),
    )
