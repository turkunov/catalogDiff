"""VLM client over the OpenAI-compatible API of vLLM."""
from typing import Protocol

GEN_KEYS = ("temperature", "top_p", "max_tokens")


class VLMClient(Protocol):
    model: str

    def complete(self, content: list[dict], gen: dict, json_mode: bool) -> str: ...


class OpenAIVLM:
    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY", timeout: int = 600, client=None):
        if client is None:
            from openai import OpenAI

            client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.client = client
        self.model = model

    def complete(self, content: list[dict], gen: dict, json_mode: bool) -> str:
        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "extra_body": {"chat_template_kwargs": {"enable_thinking": bool(gen.get("thinking", True))}},
            **{k: gen[k] for k in GEN_KEYS if k in gen},
        }
        if json_mode:
            try:
                resp = self.client.chat.completions.create(**kwargs, response_format={"type": "json_object"})
                return resp.choices[0].message.content or ""
            except Exception:
                pass  # server or model without JSON mode: retry as plain completion
        resp = self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""
