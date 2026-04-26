from src.agent.entity_sanitizer import (
    forbidden_entities,
    is_contaminated,
    sanitize_list,
)

def test_topic_self_excluded():
    fb = forbidden_entities("Re:Zero", topic_aliases=["Re Zero", "Rezero"])
    # "Re:Zero" and its aliases should not appear in the forbidden list
    assert "Re:Zero" not in fb
    assert "Re Zero" not in fb

def test_topic_foreign_listed():
    fb = forbidden_entities("Re:Zero")
    assert "Jujutsu Kaisen" in fb
    assert "JJK" in fb
    assert "Naruto" in fb

def test_contamination_detected():
    fb = forbidden_entities("Re:Zero")
    contam, hits = is_contaminated("Al and Subaru side by side JJK", fb)
    assert contam is True
    assert "JJK" in hits

def test_contamination_clean():
    fb = forbidden_entities("Re:Zero")
    contam, hits = is_contaminated("Subaru uses return by death", fb)
    assert contam is False
    assert hits == []

def test_sanitize_list():
    fb = forbidden_entities("Re:Zero")
    items = [
        "Subaru Natsuki despair",
        "Al and Subaru side by side JJK",
        "Re Zero magic circle glowing",
        "Hakari Kinji JJK gambling",
    ]
    kept, dropped = sanitize_list(items, fb)
    assert "Subaru Natsuki despair" in kept
    assert "Re Zero magic circle glowing" in kept
    assert "Al and Subaru side by side JJK" in dropped
    assert "Hakari Kinji JJK gambling" in dropped

def test_whole_word_match():
    # "naruto" substring should not false-positive on "narutosaves"
    fb = ["Naruto"]
    contam, _ = is_contaminated("narutosaves the village", fb)
    assert contam is False
