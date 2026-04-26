from src.agent.robust_json import extract_first_json, extract_json_dict, parse_yes_no

def test_plain_object():
    assert extract_first_json('{"a": 1}') == {"a": 1}

def test_prose_wrapped():
    raw = 'Sure, here is the result: {"video_angle": "test"} let me know if you need more'
    assert extract_first_json(raw) == {"video_angle": "test"}

def test_code_fence():
    raw = "```json\n{\"x\": 2}\n```"
    assert extract_first_json(raw) == {"x": 2}

def test_multiple_objects_returns_first():
    raw = '{"first": 1} {"second": 2}'
    assert extract_first_json(raw) == {"first": 1}

def test_trailing_garbage_after_close():
    # Greedy regex \{.*\} would break on this; raw_decode handles it
    raw = '{"a": {"b": 1}} and more garbage'
    assert extract_first_json(raw) == {"a": {"b": 1}}

def test_invalid_returns_none():
    assert extract_first_json("not json at all") is None

def test_required_keys_missing():
    assert extract_json_dict('{"a": 1}', required_keys=["a", "b"]) is None

def test_required_keys_present():
    assert extract_json_dict('{"a": 1, "b": 2}', required_keys=["a", "b"]) == {"a": 1, "b": 2}

def test_yes_no_permissive():
    assert parse_yes_no("yes") is True
    assert parse_yes_no("YES") is True
    assert parse_yes_no("yes.") is True
    assert parse_yes_no(" yes, definitely") is True
    assert parse_yes_no("no") is False
    assert parse_yes_no("maybe") is False
    assert parse_yes_no("") is False
    assert parse_yes_no(None) is False
