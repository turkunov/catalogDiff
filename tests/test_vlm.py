from pseudolabel.vlm import OpenAIVLM


def test_openai_vlm_passes_thinking_and_falls_back(fake_openai_client):
    vlm = OpenAIVLM("http://x/v1", "m", client=fake_openai_client)
    out = vlm.complete([{"type": "text", "text": "hi"}], {"temperature": 0.1, "max_tokens": 10, "thinking": False}, json_mode=True)
    assert out == "ok"
    calls = fake_openai_client.calls
    assert calls[0]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert calls[0]["temperature"] == 0.1 and "thinking" not in calls[0]
    assert "response_format" in calls[0] and "response_format" not in calls[1]
