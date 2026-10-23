"""Test doubles for the VLM, the OpenAI client and the image encoder."""
from collections.abc import Callable
from types import SimpleNamespace

from pseudolabel.prompts import MATCH_MARKER, TEXT_MARKER, VISUAL_MARKER


class FakeVLM:
    def __init__(self, handler: Callable[[list[dict], dict, bool], str], model: str = "fake-vlm"):
        self.handler = handler
        self.model = model
        self.calls: list[dict] = []

    def complete(self, content: list[dict], gen: dict, json_mode: bool) -> str:
        self.calls.append({"content": content, "gen": gen, "json_mode": json_mode})
        return self.handler(content, gen, json_mode)


def prompt_text(content: list[dict]) -> str:
    return " ".join(p.get("text", "") for p in content if p["type"] == "text")


def route_by_prompt(visual: str = '{"attributes": {"color": "серый"}}', text: str = '{"params": {}}',
                    match: str = '{"reasoning": "ok", "class": 3}') -> Callable[[list[dict], dict, bool], str]:
    def handler(content, gen, json_mode):
        t = prompt_text(content)
        if t.startswith(VISUAL_MARKER):
            return visual
        if t.startswith(TEXT_MARKER):
            return text
        if t.startswith(MATCH_MARKER):
            return match
        raise AssertionError("unknown prompt")

    return handler


class FakeOpenAIClient:
    """Mimics client.chat.completions.create; rejects response_format on the first call."""

    def __init__(self):
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if "response_format" in kwargs and len(self.calls) == 1:
            raise RuntimeError("response_format not supported")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])
