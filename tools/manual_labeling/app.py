#!/usr/bin/env python3
"""Blind manual labeling of the fixed manual-test set (classes 1-5).

Every click is saved to SQLite immediately and appended to an audit log.
The legacy class and anything a VLM produced are never sent to the browser.
Export: GET /export.csv -> a_id,b_id,class,comment,updated_at.
"""
import json
import os
import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from pydantic import BaseModel

PAIRS_CSV = Path(os.getenv("PAIRS_CSV", "/ml/out/manual_test_pairs.csv"))
IMAGES_ROOT = Path(os.getenv("IMAGES_ROOT", "/data/images_small"))
OUT_DIR = Path(os.getenv("OUT_DIR", "/ml/out"))
HERE = Path(__file__).parent

PAIRS = pd.read_csv(PAIRS_CSV, lineterminator="\n", dtype={"q_item_id": str, "d_item_id": str})
DB_PATH = OUT_DIR / "manual_labels.sqlite"
AUDIT_PATH = OUT_DIR / "manual_labels_audit.jsonl"
LOCK = threading.Lock()


def db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute(
        "CREATE TABLE IF NOT EXISTS labels (a_id TEXT, b_id TEXT, class INTEGER, comment TEXT,"
        " updated_at TEXT, PRIMARY KEY (a_id, b_id))"
    )
    return con


def labels() -> dict[tuple[str, str], tuple[int, str]]:
    with db() as con:
        return {(a, b): (c, m) for a, b, c, m in con.execute("SELECT a_id, b_id, class, comment FROM labels")}


def clean(v) -> str:
    return "" if pd.isna(v) else str(v)


def params(v) -> dict:
    try:
        return {k: val for k, val in json.loads(v).items()}
    except Exception:
        return {}


class LabelIn(BaseModel):
    idx: int
    cls: int | None  # None clears the label
    comment: str = ""


app = FastAPI()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (HERE / "index.html").read_text(encoding="utf-8")


@app.get("/api/legend", response_class=PlainTextResponse)
def legend() -> str:
    return (HERE / "legend.md").read_text(encoding="utf-8")


@app.get("/api/status")
def status() -> list[dict]:
    lab = labels()
    return [
        {"idx": int(r.idx), "cls": lab.get((r.q_item_id, r.d_item_id), (None, ""))[0]}
        for r in PAIRS.itertuples()
    ]


@app.get("/api/pair/{idx}")
def pair(idx: int) -> dict:
    if not 0 <= idx < len(PAIRS):
        raise HTTPException(404)
    r = PAIRS.iloc[idx]
    cls, comment = labels().get((r.q_item_id, r.d_item_id), (None, ""))
    side = lambda p, folder: {  # noqa: E731
        "title": clean(r[f"{p}_title"]),
        "description": clean(r[f"{p}_description"]),
        "params": params(r[f"{p}_key_params"]),
        "image": f"/img/{folder}/{int(r[f'{p}_image_id'])}",
    }
    return {"idx": idx, "total": len(PAIRS), "a": side("q", "avito_images"), "b": side("d", "ozon_images"),
            "cls": cls, "comment": comment}


@app.post("/api/label")
def save(body: LabelIn) -> dict:
    if not 0 <= body.idx < len(PAIRS) or (body.cls is not None and body.cls not in range(1, 6)):
        raise HTTPException(400, "bad idx or class")
    r = PAIRS.iloc[body.idx]
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with LOCK, db() as con:
        if body.cls is None:
            con.execute("DELETE FROM labels WHERE a_id=? AND b_id=?", (r.q_item_id, r.d_item_id))
        else:
            con.execute(
                "INSERT INTO labels VALUES (?,?,?,?,?) ON CONFLICT(a_id,b_id) DO UPDATE SET"
                " class=excluded.class, comment=excluded.comment, updated_at=excluded.updated_at",
                (r.q_item_id, r.d_item_id, body.cls, body.comment, now),
            )
        with AUDIT_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": now, "a_id": r.q_item_id, "b_id": r.d_item_id, "cls": body.cls,
                                "comment": body.comment}, ensure_ascii=False) + "\n")
    return {"ok": True}


@app.get("/img/{folder}/{image_id}")
def image(folder: str, image_id: str) -> FileResponse:
    if folder not in {"avito_images", "ozon_images"} or not re.fullmatch(r"\d+", image_id):
        raise HTTPException(404)
    path = IMAGES_ROOT / folder / f"{image_id}.jpeg"
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path)


@app.get("/export.csv", response_class=PlainTextResponse)
def export() -> str:
    with db() as con:
        df = pd.read_sql("SELECT a_id, b_id, class, comment, updated_at FROM labels ORDER BY updated_at", con)
    return df.to_csv(index=False)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    db().close()
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7870")))
