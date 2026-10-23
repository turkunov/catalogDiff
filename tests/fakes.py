"""Test doubles for the VLM, the OpenAI client and the image encoder."""
from collections.abc import Callable
from types import SimpleNamespace

from pseudolabel.prompts import MATCH_MARKER, TEXT_MARKER, VISUAL_MARKER


class FakeVLM:
    """The handler returns the answer text, or a Completion to also return logprobs and token counts."""

    def __init__(self, handler: Callable[[list[dict], dict, bool], object], model: str = "fake-vlm"):
        self.handler = handler
        self.model = model
        self.calls: list[dict] = []

    def complete(self, content: list[dict], gen: dict, json_mode: bool) -> str:
        return self.complete_detailed(content, gen, json_mode).text

    def complete_detailed(self, content: list[dict], gen: dict, json_mode: bool, top_logprobs: int = 0):
        from pseudolabel.vlm import Completion

        self.calls.append({"content": content, "gen": gen, "json_mode": json_mode, "top_logprobs": top_logprobs})
        out = self.handler(content, gen, json_mode)
        return out if isinstance(out, Completion) else Completion(out, None, None)


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


def token_logprobs(tokens: list[str], top: dict[int, list[tuple[str, float]]] | None = None) -> list:
    """Logprob entries shaped like the OpenAI client's; `top` maps a token index to its alternatives."""
    top = top or {}
    return [SimpleNamespace(token=t, logprob=0.0,
                            top_logprobs=[SimpleNamespace(token=a, logprob=lp) for a, lp in top.get(i, [(t, 0.0)])])
            for i, t in enumerate(tokens)]


class FakeOpenAIClient:
    """Mimics client.chat.completions.create; rejects response_format on the first call.

    With `logprobs` set, answers carry them; with `reject_logprobs`, requests asking for logprobs fail.
    """

    def __init__(self, content: str = "ok", logprobs: list | None = None, reject_logprobs: bool = False):
        self.calls: list[dict] = []
        self.content, self.logprobs, self.reject_logprobs = content, logprobs, reject_logprobs
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if "response_format" in kwargs and len(self.calls) == 1:
            raise RuntimeError("response_format not supported")
        if kwargs.get("logprobs") and self.reject_logprobs:
            raise RuntimeError("logprobs not supported")
        lp = SimpleNamespace(content=self.logprobs) if kwargs.get("logprobs") and self.logprobs is not None else None
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.content), logprobs=lp)],
                               usage=SimpleNamespace(completion_tokens=7))


class FakeEncoder:
    """Returns fixed vectors per image path, L2-normalized."""

    def __init__(self, vectors: dict, version: str = "fake@1"):
        import numpy as np

        self.version = version
        self.vectors = {k: np.asarray(v, dtype=np.float32) / np.linalg.norm(v) for k, v in vectors.items()}
        self.embedded: list[str] = []

    def embed(self, images: list[str]):
        import numpy as np

        self.embedded += images
        return np.stack([self.vectors[i] for i in images]).astype(np.float32)
