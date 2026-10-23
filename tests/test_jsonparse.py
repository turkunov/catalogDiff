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


def test_braces_in_prose_before_json():
    from pseudolabel.jsonparse import find_json_object

    text = 'Attributes {color} differ. {"reasoning": "r", "class": 2}'
    assert find_json_object(text, ("class",)) == {"reasoning": "r", "class": 2}


def test_two_objects_takes_last_with_key():
    from pseudolabel.jsonparse import find_json_object

    text = '{"class": 3} draft... final: {"reasoning": "ok", "class": 4}'
    assert find_json_object(text, ("class",))["class"] == 4


def test_find_json_object_key_case_insensitive():
    from pseudolabel.jsonparse import find_json_object

    assert find_json_object('{"Class": 5}', ("class",)) == {"Class": 5}
