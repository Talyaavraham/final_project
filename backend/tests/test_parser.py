from query_parser import parse, parse_rules


def test_english_subject_first():
    p = parse_rules("find a man in a red shirt walking with a dog")
    assert p["objects"] == ["person", "dog"]
    assert p["colors"] == ["red"]
    assert p["actions"] == ["walking"]
    assert p["timeRange"] is None
    assert p["unknown"] == [] and p["semantic"] == ""


def test_hebrew_with_prefixes():
    p = parse_rules("מצא איש שרץ עם הכלב")
    assert p["objects"] == ["person", "dog"]
    assert p["actions"] == ["running"]


def test_riding_adds_person():
    p = parse_rules("someone riding a bicycle")
    assert p["objects"][0] == "person" and "bicycle" in p["objects"]


def test_time_ranges():
    assert parse_rules("cars between 0:10 and 1:30")["timeRange"] == [10, 90]
    assert parse_rules("cars after 2:00")["timeRange"] == [120, None]
    assert parse_rules("כלב לפני 1:00")["timeRange"] == [0, 60]
    assert parse_rules("a bus at 1:30")["timeRange"] == [85, 95]


def test_unknown_words_become_semantic_query():
    p = parse_rules("a woman with a stroller")
    assert p["objects"] == ["person"]
    assert "stroller" in p["unknown"]
    assert p["semantic"] == "a woman with a stroller"


def test_parse_falls_back_to_rules_without_llm(monkeypatch):
    import query_parser
    monkeypatch.setattr(query_parser, "llm_available", lambda: False)
    assert parse("a red car")["source"] == "rules"


def test_parse_uses_llm_result_when_available(monkeypatch):
    import query_parser
    monkeypatch.setattr(query_parser, "llm_available", lambda: True)
    monkeypatch.setattr(query_parser, "parse_llm",
                        lambda text, classes: {"objects": ["person"], "colors": [],
                                               "actions": [], "timeRange": None,
                                               "semantic": "x", "unknown": [],
                                               "source": "llm"})
    assert parse("anything")["source"] == "llm"


def test_parse_llm_handles_missing_sdk(monkeypatch):
    import builtins
    import query_parser
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "anthropic":
            raise ImportError
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert query_parser.parse_llm("a dog", ["dog"]) is None


def test_game_classes_hebrew_and_english():
    p = parse_rules("דמות ליד עץ ושמש")
    assert p["objects"] == ["character", "tree", "sun"]
    assert p["colors"] == []
    p = parse_rules("the player under the sun")
    assert p["objects"] == ["character", "sun"]


def test_person_falls_back_to_character_when_video_has_no_person():
    p = parse_rules("איש רץ", classes=["character", "tree", "fruit"])
    assert p["objects"] == ["character"]
    assert p["actions"] == ["running"]
    # a video with real people keeps "person"
    assert parse_rules("איש רץ", classes=["person", "dog"])["objects"] == ["person"]
    assert parse_rules("איש רץ")["objects"] == ["person"]


def test_colour_binds_to_the_object_next_to_it():
    # Hebrew: adjective after the noun
    p = parse_rules("דמות ליד עץ עם תפוח אדום")
    assert p["objects"] == ["fruit", "character", "tree"] and p["colors"] == ["red"]
    # English: adjective before the noun
    p = parse_rules("a character next to a yellow fruit")
    assert p["objects"] == ["fruit", "character"] and p["colors"] == ["yellow"]
    # colour of the subject itself stays as is
    p = parse_rules("red car next to a bus")
    assert p["objects"] == ["car", "bus"]


def test_security_classes_are_objects_when_the_video_has_them():
    q = parse_rules("a person with a gun", classes=["person", "gun", "car"])
    assert q["objects"] == ["person", "gun"] and q["unknown"] == []
    q = parse_rules("אדם עם סכין", classes=["person", "knife"])
    assert q["objects"] == ["person", "knife"]
    assert parse_rules("fire", classes=["fire"])["objects"] == ["fire"]


def test_absent_security_class_becomes_a_clip_question():
    q = parse_rules("a person with a gun", classes=["person", "car"])
    assert q["objects"] == ["person"]
    assert q["unknown"] == ["gun"] and q["semantic"] == "a person with a gun"
    # nothing left to filter on -> stays an object (the search falls back to scenes)
    assert parse_rules("fire", classes=["person"])["objects"] == ["fire"]
