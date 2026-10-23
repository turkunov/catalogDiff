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
