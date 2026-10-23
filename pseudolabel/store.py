"""SQLite store: profile versions, item snapshots and annotations, labels, iterations, queues."""
import json
import sqlite3
import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from pseudolabel.data import Item
from pseudolabel.profiles import Profile, dump_profile, parse_profile, profile_hash

SCHEMA = """
CREATE TABLE IF NOT EXISTS profile_versions (
    name TEXT, version INTEGER, hash TEXT, yaml TEXT, created_at TEXT,
    PRIMARY KEY (name, version), UNIQUE (name, hash));
CREATE TABLE IF NOT EXISTS items (
    profile TEXT, side TEXT, item_id TEXT, title TEXT, description TEXT, key_params TEXT, image TEXT,
    PRIMARY KEY (profile, side, item_id));
CREATE TABLE IF NOT EXISTS item_annotations (
    profile TEXT, side TEXT, item_id TEXT, kind TEXT, model TEXT, profile_version INTEGER,
    result TEXT, raw TEXT, error TEXT, latency_s REAL, created_at TEXT,
    PRIMARY KEY (profile, side, item_id, kind, model, profile_version));
CREATE TABLE IF NOT EXISTS labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT, profile TEXT, a_id TEXT, b_id TEXT, source TEXT, cls INTEGER,
    reasoning TEXT, raw TEXT, error TEXT, latency_s REAL, comment TEXT, model TEXT, profile_version INTEGER,
    gen_params TEXT, created_at TEXT);
CREATE INDEX IF NOT EXISTS labels_pair ON labels (profile, a_id, b_id, source);
CREATE TABLE IF NOT EXISTS iterations (
    id INTEGER PRIMARY KEY AUTOINCREMENT, profile TEXT, profile_version INTEGER, config TEXT,
    encoder_version TEXT, status TEXT, report TEXT, created_at TEXT, finished_at TEXT);
CREATE TABLE IF NOT EXISTS iteration_pairs (
    iteration_id INTEGER, a_id TEXT, b_id TEXT, origin TEXT, rank INTEGER, cosine REAL,
    selected_for_vlm INTEGER, status TEXT, PRIMARY KEY (iteration_id, a_id, b_id));
CREATE TABLE IF NOT EXISTS manual_test (
    profile TEXT, a_id TEXT, b_id TEXT, legacy_class INTEGER, PRIMARY KEY (profile, a_id, b_id));
CREATE TABLE IF NOT EXISTS queues (
    queue TEXT, profile TEXT, a_id TEXT, b_id TEXT, position INTEGER, PRIMARY KEY (queue, a_id, b_id));
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class LabelRow:
    profile: str
    a_id: str
    b_id: str
    source: str
    cls: int | None
    reasoning: str = ""
    raw: str = ""
    error: str | None = None
    latency_s: float | None = None
    comment: str = ""
    model: str = ""
    profile_version: int | None = None
    gen_params: dict = field(default_factory=dict)
    created_at: str = ""


_LABEL_COLS = "profile, a_id, b_id, source, cls, reasoning, raw, error, latency_s, comment, model, profile_version, gen_params, created_at"


def _label_from_row(r: sqlite3.Row) -> LabelRow:
    d = dict(r)
    d["gen_params"] = json.loads(d["gen_params"] or "{}")
    return LabelRow(**{k: d[k] for k in _LABEL_COLS.split(", ")})


class Store:
    def __init__(self, path: Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(path, check_same_thread=False)
        self._con.row_factory = sqlite3.Row
        self._con.execute("PRAGMA journal_mode=WAL")
        self._lock = threading.Lock()
        with self._lock:
            self._con.executescript(SCHEMA)

    def _write(self, sql: str, params: Iterable = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._con.execute(sql, tuple(params))
            self._con.commit()
            return cur

    def _write_many(self, sql: str, rows: list[tuple]) -> None:
        with self._lock:
            self._con.executemany(sql, rows)
            self._con.commit()

    def _read(self, sql: str, params: Iterable = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._con.execute(sql, tuple(params)).fetchall()

    # profile versions
    def register_profile_version(self, p: Profile) -> int:
        h = profile_hash(p)
        with self._lock:
            row = self._con.execute("SELECT version FROM profile_versions WHERE name=? AND hash=?", (p.name, h)).fetchone()
            if row:
                return row[0]
            version = (self._con.execute("SELECT MAX(version) FROM profile_versions WHERE name=?", (p.name,)).fetchone()[0] or 0) + 1
            self._con.execute("INSERT INTO profile_versions VALUES (?,?,?,?,?)", (p.name, version, h, dump_profile(p), _now()))
            self._con.commit()
            return version

    def profile_at(self, name: str, version: int) -> Profile:
        """The profile exactly as it was registered under `version`."""
        rows = self._read("SELECT yaml FROM profile_versions WHERE name=? AND version=?", (name, version))
        if not rows:
            raise KeyError(f"profile {name!r} has no version {version}")
        return parse_profile(rows[0][0])

    def profile_hash_at(self, name: str, version: int) -> str | None:
        rows = self._read("SELECT hash FROM profile_versions WHERE name=? AND version=?", (name, version))
        return rows[0][0] if rows else None

    def profile_versions(self, name: str) -> list[tuple[int, str]]:
        return [(r[0], r[1]) for r in self._read("SELECT version, created_at FROM profile_versions WHERE name=? ORDER BY version", (name,))]

    # items and per-item annotations
    def upsert_items(self, profile: str, items: Iterable[Item]) -> None:
        rows = [(profile, i.side, i.id, i.title, i.description, json.dumps(i.key_params, ensure_ascii=False), i.image) for i in items]
        self._write_many("INSERT OR REPLACE INTO items VALUES (?,?,?,?,?,?,?)", rows)

    def get_item(self, profile: str, side: str, item_id: str) -> Item | None:
        rows = self._read("SELECT * FROM items WHERE profile=? AND side=? AND item_id=?", (profile, side, item_id))
        if not rows:
            return None
        r = rows[0]
        return Item(r["item_id"], r["side"], r["title"], r["description"], json.loads(r["key_params"]), r["image"])

    def get_annotation(self, profile, side, item_id, kind, model, profile_version) -> dict | None:
        rows = self._read(
            "SELECT result FROM item_annotations WHERE profile=? AND side=? AND item_id=? AND kind=? AND model=?"
            " AND profile_version=? AND error IS NULL", (profile, side, item_id, kind, model, profile_version))
        return json.loads(rows[0][0]) if rows else None

    def put_annotation(self, profile, side, item_id, kind, model, profile_version, *, result: dict, raw: str,
                       error: str | None, latency_s: float | None) -> None:
        self._write("INSERT OR REPLACE INTO item_annotations VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (profile, side, item_id, kind, model, profile_version, json.dumps(result, ensure_ascii=False),
                     raw, error, latency_s, _now()))

    def annotations(self, profile: str, side: str, item_id: str) -> list[dict]:
        rows = self._read("SELECT kind, model, profile_version, result, error, latency_s, created_at FROM item_annotations"
                          " WHERE profile=? AND side=? AND item_id=? ORDER BY created_at", (profile, side, item_id))
        return [{**dict(r), "result": json.loads(r["result"] or "{}")} for r in rows]

    # labels
    def put_label(self, row: LabelRow) -> int:
        created = row.created_at or _now()
        cur = self._write(
            f"INSERT INTO labels ({_LABEL_COLS}) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (row.profile, row.a_id, row.b_id, row.source, row.cls, row.reasoning, row.raw, row.error, row.latency_s,
             row.comment, row.model, row.profile_version, json.dumps(row.gen_params), created))
        return cur.lastrowid

    def get_vlm_label(self, profile, a_id, b_id, model, profile_version) -> LabelRow | None:
        rows = self._read(
            f"SELECT {_LABEL_COLS} FROM labels WHERE profile=? AND a_id=? AND b_id=? AND source='vlm' AND model=?"
            " AND profile_version=? AND error IS NULL ORDER BY id DESC LIMIT 1", (profile, a_id, b_id, model, profile_version))
        return _label_from_row(rows[0]) if rows else None

    def latest_human_label(self, profile, a_id, b_id) -> LabelRow | None:
        rows = self._read(
            f"SELECT {_LABEL_COLS} FROM labels WHERE profile=? AND a_id=? AND b_id=? AND source='human'"
            " ORDER BY id DESC LIMIT 1", (profile, a_id, b_id))
        return _label_from_row(rows[0]) if rows else None

    def import_legacy(self, profile: str, labels: dict[tuple[str, str], int]) -> int:
        existing = {(r[0], r[1]) for r in self._read("SELECT a_id, b_id FROM labels WHERE profile=? AND source='legacy'", (profile,))}
        rows = [(profile, a, b, "legacy", int(c), _now()) for (a, b), c in labels.items() if (a, b) not in existing]
        self._write_many("INSERT INTO labels (profile, a_id, b_id, source, cls, created_at) VALUES (?,?,?,?,?,?)", rows)
        return len(rows)

    # iterations
    def create_iteration(self, profile: str, profile_version: int, config: dict, encoder_version: str) -> int:
        cur = self._write(
            "INSERT INTO iterations (profile, profile_version, config, encoder_version, status, created_at)"
            " VALUES (?,?,?,?,?,?)", (profile, profile_version, json.dumps(config, ensure_ascii=False), encoder_version, "running", _now()))
        return cur.lastrowid

    def add_iteration_pairs(self, iteration_id: int, rows: list[dict]) -> None:
        self._write_many(
            "INSERT OR REPLACE INTO iteration_pairs VALUES (?,?,?,?,?,?,?,?)",
            [(iteration_id, r["a_id"], r["b_id"], r["origin"], r.get("rank"), r.get("cosine"),
              int(bool(r["selected_for_vlm"])), "pending" if r["selected_for_vlm"] else "not_selected") for r in rows])

    def iteration_pairs(self, iteration_id: int, status: str | None = None) -> pd.DataFrame:
        sql, params = "SELECT * FROM iteration_pairs WHERE iteration_id=?", [iteration_id]
        if status:
            sql, params = sql + " AND status=?", params + [status]
        with self._lock:
            return pd.read_sql(sql, self._con, params=params)

    def set_iteration_pair_status(self, iteration_id: int, a_id: str, b_id: str, status: str) -> None:
        self._write("UPDATE iteration_pairs SET status=? WHERE iteration_id=? AND a_id=? AND b_id=?", (status, iteration_id, a_id, b_id))

    def pair_origins(self, profile: str, a_id: str, b_id: str) -> pd.DataFrame:
        with self._lock:
            return pd.read_sql(
                "SELECT ip.iteration_id, ip.origin, ip.rank, ip.cosine, ip.status FROM iteration_pairs ip"
                " JOIN iterations it ON it.id = ip.iteration_id WHERE it.profile=? AND ip.a_id=? AND ip.b_id=?"
                " ORDER BY ip.iteration_id", self._con, params=[profile, a_id, b_id])

    def finish_iteration(self, iteration_id: int, status: str, report: dict) -> None:
        self._write("UPDATE iterations SET status=?, report=?, finished_at=? WHERE id=?",
                    (status, json.dumps(report, ensure_ascii=False, default=str), _now(), iteration_id))

    def get_iteration(self, iteration_id: int) -> dict:
        r = dict(self._read("SELECT * FROM iterations WHERE id=?", (iteration_id,))[0])
        r["config"] = json.loads(r["config"] or "{}")
        r["report"] = json.loads(r["report"]) if r["report"] else None
        return r

    def list_iterations(self, profile: str) -> pd.DataFrame:
        with self._lock:
            return pd.read_sql("SELECT * FROM iterations WHERE profile=? ORDER BY id", self._con, params=[profile])

    # manual test and review queues
    def set_manual_test(self, profile: str, rows: list[tuple[str, str, int]]) -> None:
        with self._lock:
            self._con.execute("DELETE FROM manual_test WHERE profile=?", (profile,))
            self._con.executemany("INSERT INTO manual_test VALUES (?,?,?,?)", [(profile, a, b, c) for a, b, c in rows])
            self._con.commit()

    def manual_test_pairs(self, profile: str) -> list[tuple[str, str, int]]:
        return [(r[0], r[1], r[2]) for r in self._read("SELECT a_id, b_id, legacy_class FROM manual_test WHERE profile=? ORDER BY rowid", (profile,))]

    def queue_put(self, queue: str, profile: str, pairs: list[tuple[str, str]]) -> None:
        start = self._read("SELECT COALESCE(MAX(position), -1) FROM queues WHERE queue=?", (queue,))[0][0] + 1
        self._write_many("INSERT OR IGNORE INTO queues VALUES (?,?,?,?,?)",
                         [(queue, profile, a, b, start + i) for i, (a, b) in enumerate(pairs)])

    def queue_pairs(self, queue: str) -> list[tuple[str, str]]:
        return [(r[0], r[1]) for r in self._read("SELECT a_id, b_id FROM queues WHERE queue=? ORDER BY position", (queue,))]

    def queue_profile(self, queue: str) -> str | None:
        rows = self._read("SELECT profile FROM queues WHERE queue=? LIMIT 1", (queue,))
        return rows[0][0] if rows else None

    def queues(self, profile: str) -> list[str]:
        return [r[0] for r in self._read("SELECT DISTINCT queue FROM queues WHERE profile=? ORDER BY queue", (profile,))]

    # browsing
    def query_labels(self, profile: str, **filters) -> pd.DataFrame:
        """Labels joined with their iteration origin (the given iteration, or the latest one with the pair)."""
        iteration_id = filters.pop("iteration_id", None)
        disagreement = filters.pop("disagreement", None)
        ip_filter = "ip.iteration_id = ?" if iteration_id is not None else (
            "ip.iteration_id = (SELECT MAX(ip2.iteration_id) FROM iteration_pairs ip2 JOIN iterations it2"
            " ON it2.id = ip2.iteration_id WHERE it2.profile = l.profile AND ip2.a_id = l.a_id AND ip2.b_id = l.b_id)")
        sql = (
            "SELECT l.a_id, l.b_id, l.source, l.cls, l.reasoning, l.model, l.profile_version, ip.origin, ip.rank,"
            " ip.cosine, ip.iteration_id, l.error, l.comment, l.latency_s, l.created_at FROM labels l"
            f" LEFT JOIN iteration_pairs ip ON ip.a_id = l.a_id AND ip.b_id = l.b_id AND {ip_filter}"
            " WHERE l.profile = ?")
        params: list = ([iteration_id] if iteration_id is not None else []) + [profile]
        if iteration_id is not None:
            sql += " AND ip.iteration_id IS NOT NULL"
        for col in ("origin", "cls", "model", "profile_version", "source"):
            if filters.get(col) is not None:
                sql += f" AND {'ip' if col == 'origin' else 'l'}.{col} = ?"
                params.append(filters[col])
        if disagreement:
            sql += (" AND EXISTS (SELECT 1 FROM queues q WHERE q.profile = l.profile AND q.a_id = l.a_id"
                    " AND q.b_id = l.b_id AND q.queue LIKE 'disagreements:%')")
        with self._lock:
            return pd.read_sql(sql + " ORDER BY l.id", self._con, params=params)
