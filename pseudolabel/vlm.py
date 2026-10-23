"""VLM client over the OpenAI-compatible API of vLLM."""
import math
import re
from dataclasses import dataclass
from typing import Protocol

GEN_KEYS = ("temperature", "top_p", "max_tokens")


@dataclass
class Completion:
    text: str
    logprobs: list[dict] | None  # [{"token": str, "top_logprobs": [{"token": str, "logprob": float}]}]
    completion_tokens: int | None


class VLMClient(Protocol):
    model: str

    def complete(self, content: list[dict], gen: dict, json_mode: bool) -> str: ...


_CLASS_KEY = re.compile(r'"(class|класс)"\s*:\s*$', re.IGNORECASE)
_DIGIT = re.compile(r"^\s*([1-5])\s*$")


def class_probs(logprobs: list[dict] | None) -> dict[int, float] | None:
    """Distribution over classes 1..5 at the class digit of the answer; thinking before </think> is skipped."""
    if not logprobs:
        return None
    tokens = [e["token"] for e in logprobs]
    start = 0
    for i, t in enumerate(tokens):
        if "</think>" in t:
            start = i + 1
    text = ""
    for i in range(start, len(tokens)):
        if _DIGIT.match(tokens[i]) and _CLASS_KEY.search(text):
            mass = dict.fromkeys(range(1, 6), 0.0)
            for alt in logprobs[i]["top_logprobs"]:
                m = _DIGIT.match(alt["token"])
                if m:
                    mass[int(m.group(1))] += math.exp(alt["logprob"])
            total = sum(mass.values())
            return {k: v / total for k, v in mass.items()} if total > 0 else None
        text += tokens[i]
    return None


def _plain_logprobs(choice) -> list[dict] | None:
    lp = getattr(choice, "logprobs", None)
    content = getattr(lp, "content", None) if lp is not None else None
    if not content:
        return None
    return [{"token": e.token, "top_logprobs": [{"token": a.token, "logprob": a.logprob} for a in (e.top_logprobs or [])]}
            for e in content]


class OpenAIVLM:
    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY", timeout: int = 600, client=None):
        if client is None:
            from openai import OpenAI

            client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.client = client
        self.model = model

    def complete(self, content: list[dict], gen: dict, json_mode: bool) -> str:
        return self.complete_detailed(content, gen, json_mode).text

    def complete_detailed(self, content: list[dict], gen: dict, json_mode: bool, top_logprobs: int = 0) -> Completion:
        """One answer; logprobs are requested when top_logprobs > 0 and dropped if the server refuses them."""
        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "extra_body": {"chat_template_kwargs": {"enable_thinking": bool(gen.get("thinking", True))}},
            **{k: gen[k] for k in GEN_KEYS if k in gen},
        }
        if top_logprobs:
            try:
                return self._create({**kwargs, "logprobs": True, "top_logprobs": top_logprobs}, json_mode)
            except Exception:
                pass  # server or model without logprobs: answer without them
        return self._create(kwargs, json_mode)

    def _create(self, kwargs: dict, json_mode: bool) -> Completion:
        resp = None
        if json_mode:
            try:
                resp = self.client.chat.completions.create(**kwargs, response_format={"type": "json_object"})
            except Exception:
                pass  # server or model without JSON mode: retry as plain completion
        if resp is None:
            resp = self.client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        usage = getattr(resp, "usage", None)
        return Completion(choice.message.content or "", _plain_logprobs(choice), getattr(usage, "completion_tokens", None))
