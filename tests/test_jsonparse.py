from pseudolabel.jsonparse import parse_first_json_object


def test_plain_json():
    assert parse_first_json_object('{"class": 4}') == {"class": 4}


def test_fenced_after_thinking():
    text = 'Reasoning... {"draft": 1} </think>\n```json\n{"class": 3, "reasoning": "x"}\n```'
    assert parse_first_json_object(text) == {"class": 3, "reasoning": "x"}


def test_python_literals():
    assert parse_first_json_object("{'a': True, 'b': None}") == {"a": True, "b": None}


def test_garbage_returns_empty():
    assert parse_first_json_object("no json here") == {}
