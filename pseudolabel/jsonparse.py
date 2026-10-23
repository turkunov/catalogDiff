"""Lenient extraction of JSON objects from model output."""
import ast
import json
from typing import Any

_DECODER = json.JSONDecoder()


def _strip_thinking(text: str) -> str:
    return text.rsplit("</think>", 1)[1] if "</think>" in text else text


def _objects(text: str) -> list[dict]:
    """Every top-level JSON object in `text`, scanning from each '{' (prose with braces is skipped)."""
    out, i = [], text.find("{")
    while i != -1:
        try:
            value, end = _DECODER.raw_decode(text, i)
        except json.JSONDecodeError:
            value, end = None, i + 1
        if isinstance(value, dict):
            out.append(value)
            i = text.find("{", end)
        else:
            i = text.find("{", end)
    return out


def _literal(text: str) -> dict | None:
    """Python-literal fallback for answers with True/False/None or single quotes."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        value: Any = ast.literal_eval(text[start:end + 1])
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def parse_first_json_object(text: str) -> dict:
    """The first JSON object in `text` after any `</think>`, or {}."""
    if not text:
        return {}
    body = _strip_thinking(text)
    objects = _objects(body)
    return objects[0] if objects else (_literal(body) or {})


def find_json_object(text: str, keys: tuple[str, ...]) -> dict:
    """The last object having one of `keys` (case-insensitive), else the first object, else {}."""
    if not text:
        return {}
    body = _strip_thinking(text)
    wanted = {k.lower() for k in keys}
    objects = _objects(body) or [o for o in [_literal(body)] if o]
    for obj in reversed(objects):
        if wanted & {str(k).lower() for k in obj}:
            return obj
    return objects[0] if objects else {}
