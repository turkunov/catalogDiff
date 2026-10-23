from pseudolabel.vlm import OpenAIVLM


def test_openai_vlm_passes_thinking_and_falls_back(fake_openai_client):
    vlm = OpenAIVLM("http://x/v1", "m", client=fake_openai_client)
    out = vlm.complete([{"type": "text", "text": "hi"}], {"temperature": 0.1, "max_tokens": 10, "thinking": False}, json_mode=True)
    assert out == "ok"
    calls = fake_openai_client.calls
    assert calls[0]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert calls[0]["temperature"] == 0.1 and "thinking" not in calls[0]
    assert "response_format" in calls[0] and "response_format" not in calls[1]


def _lp(tokens, top=None):
    """Logprob entries as complete_detailed returns them (plain dicts)."""
    top = top or {}
    return [{"token": t, "top_logprobs": [{"token": a, "logprob": lp} for a, lp in top.get(i, [(t, 0.0)])]}
            for i, t in enumerate(tokens)]


def test_class_probs_ignores_digits_inside_thinking():
    import math

    from pseudolabel.vlm import class_probs

    tokens = ["<think>", '"class', '":', " ", "2", " maybe", "</think>", '{"', "class", '":', " ", "4", "}"]
    probs = class_probs(_lp(tokens, {4: [("2", -0.01)], 11: [("4", -0.1), ("5", -2.5), ("3", -3.0), ("x", -1.0)]}))
    z = math.exp(-0.1) + math.exp(-2.5) + math.exp(-3.0)
    assert set(probs) == {1, 2, 3, 4, 5}
    assert abs(probs[4] - math.exp(-0.1) / z) < 1e-9 and probs[1] == 0.0 and abs(sum(probs.values()) - 1) < 1e-9


def test_class_probs_merges_spaced_digit_tokens_and_russian_key():
    from pseudolabel.vlm import class_probs

    probs = class_probs(_lp(['{"', "Класс", '":', " 5"], {3: [(" 5", -0.2), ("5", -0.2), (" 4", -5.0)]}))
    assert probs[5] > 0.99 and probs[4] > 0


def test_class_probs_none_without_class_digit():
    from pseudolabel.vlm import class_probs

    assert class_probs(None) is None
    assert class_probs(_lp(['{"', "reasoning", '":', ' "ok"', "}"])) is None


def test_complete_detailed_requests_logprobs_and_returns_them():
    from fakes import FakeOpenAIClient, token_logprobs

    client = FakeOpenAIClient(content='{"class": 4}', logprobs=token_logprobs(['{"', "class", '":', " ", "4", "}"]))
    vlm = OpenAIVLM("http://x/v1", "m", client=client)
    out = vlm.complete_detailed([{"type": "text", "text": "hi"}], {"thinking": True}, json_mode=False, top_logprobs=5)
    assert out.text == '{"class": 4}' and out.completion_tokens == 7
    assert client.calls[0]["logprobs"] is True and client.calls[0]["top_logprobs"] == 5
    assert [e["token"] for e in out.logprobs][4] == "4" and out.logprobs[4]["top_logprobs"][0]["token"] == "4"


def test_complete_detailed_falls_back_without_logprobs():
    from fakes import FakeOpenAIClient

    client = FakeOpenAIClient(content="plain", reject_logprobs=True)
    vlm = OpenAIVLM("http://x/v1", "m", client=client)
    out = vlm.complete_detailed([{"type": "text", "text": "hi"}], {}, json_mode=False, top_logprobs=5)
    assert out.text == "plain" and out.logprobs is None
    assert "logprobs" not in client.calls[-1]


def test_top_k_goes_to_extra_body(fake_openai_client):
    vlm = OpenAIVLM("http://x/v1", "m", client=fake_openai_client)
    vlm.complete([{"type": "text", "text": "hi"}], {"temperature": 0.6, "top_p": 0.95, "top_k": 20}, json_mode=False)
    call = fake_openai_client.calls[-1]
    assert call["extra_body"]["top_k"] == 20 and "top_k" not in call and call["top_p"] == 0.95


class ScriptedClient:
    """Answers from a list in order; each item is (content, finish_reason)."""

    def __init__(self, answers):
        from types import SimpleNamespace

        self.ns, self.answers, self.calls = SimpleNamespace, list(answers), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        content, finish = self.answers.pop(0)
        ns = self.ns
        return ns(choices=[ns(message=ns(content=content), finish_reason=finish, logprobs=None)],
                  usage=ns(completion_tokens=len(content)))


def test_budget_forcing_closes_cut_thinking_and_asks_for_the_answer():
    client = ScriptedClient([("думаю долго, класс 4 подходит", "length"), ('{"class": 4}', "stop")])
    vlm = OpenAIVLM("http://x/v1", "m", client=client)
    out = vlm.complete_detailed([{"type": "text", "text": "hi"}], {"thinking": True, "max_tokens": 50}, json_mode=False)
    assert out.text.endswith('</think>\n\n{"class": 4}') and out.completion_tokens == len("думаю долго, класс 4 подходит") + 12
    forced = client.calls[1]
    assert forced["messages"][-1] == {"role": "assistant", "content": "думаю долго, класс 4 подходит\n</think>\n\n"}
    assert forced["extra_body"]["continue_final_message"] is True and forced["extra_body"]["add_generation_prompt"] is False
    assert forced["max_tokens"] == 512


def test_no_budget_forcing_without_thinking_or_when_finished():
    client = ScriptedClient([("cut", "length"), ('{"class": 2}', "stop")])
    vlm = OpenAIVLM("http://x/v1", "m", client=client)
    assert vlm.complete_detailed([{"type": "text", "text": "hi"}], {"thinking": False}, json_mode=False).text == "cut"
    assert vlm.complete_detailed([{"type": "text", "text": "hi"}], {"thinking": True}, json_mode=False).text == '{"class": 2}'
    assert len(client.calls) == 2
