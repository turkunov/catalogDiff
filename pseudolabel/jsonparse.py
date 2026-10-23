"""Lenient extraction of the first JSON object from model output."""
import ast
import json
import re
from typing import Any


def _as_dict(value: Any) -> dict | None:
    return value if isinstance(value, dict) else None


def parse_first_json_object(text: str) -> dict:
    """Return the first JSON object in `text`, or {} if there is none.

    Reasoning before `</think>` is dropped. Candidates are tried in order: the whole text,
    fenced ```json blocks, then the first `{...}` span; each with json.loads and then
    ast.literal_eval (for True/False/None).
    """
    if not text:
        return {}
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    candidates = [text.strip()]
    candidates += [b.strip() for b in re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text) if b.strip()]
    span = re.search(r"\{[\s\S]*\}", text)
    if span:
        candidates.append(span.group(0))
    for candidate in candidates:
        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = _as_dict(loader(candidate))
            except Exception:
                continue
            if parsed is not None:
                return parsed
    return {}
