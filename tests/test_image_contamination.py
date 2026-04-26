from src.agent.entity_sanitizer import forbidden_entities, sanitize_list


def test_rezero_jjk_keywords_dropped():
    fb = forbidden_entities("Re:Zero", topic_aliases=["Re Zero", "Rezero"])
    raw = [
        "Subaru Natsuki despair",
        "Re Zero magic circle glowing",
        "Al and Subaru side by side JJK",
        "Hakari Kinji JJK gambling",
        "Charles Bernard manga JJK",
    ]
    kept, dropped = sanitize_list(raw, fb)
    assert "Subaru Natsuki despair" in kept
    assert "Re Zero magic circle glowing" in kept
    assert all("JJK" in d for d in dropped)
    assert len(dropped) == 3


def test_clean_keywords_unchanged():
    fb = forbidden_entities("Naruto")
    raw = ["Naruto fight scene", "Naruto Shippuden battle"]
    kept, dropped = sanitize_list(raw, fb)
    assert len(kept) == 2
    assert dropped == []


def test_synthesize_keywords_from_topic():
    from src.agent.image_agent import _synthesize_keywords

    class _P:
        topic = "Re:Zero"
        topic_category = "anime"
        entity_aliases = ["Re Zero"]

    block = {"text": "Subaru Natsuki faced the Witch of Envy in that dream."}
    kw = _synthesize_keywords(_P(), block, 0)
    assert any("Re:Zero" in k for k in kw)
    assert any("Subaru" in k or "Natsuki" in k or "Witch" in k for k in kw)
