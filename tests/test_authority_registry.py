from src.agent.authority_registry import classify, tier_weight

def test_wikipedia_tier1():
    assert classify("https://en.wikipedia.org/wiki/Re:Zero") == 1

def test_anilist_tier1():
    assert classify("https://anilist.co/anime/21355") == 1

def test_screenrant_tier2():
    assert classify("https://screenrant.com/re-zero-darkest-moments") == 2

def test_reddit_tier3():
    assert classify("https://www.reddit.com/r/Re_Zero/comments/abc") == 3

def test_fandom_with_matching_alias_tier1():
    assert classify("https://rezero.fandom.com/wiki/Subaru", topic_aliases=["rezero", "Re:Zero"]) == 1

def test_fandom_without_alias_tier2():
    # Fandom but no alias match — still tier 2, never random tier 4
    assert classify("https://naruto.fandom.com/wiki/Kakashi", topic_aliases=["rezero"]) == 2

def test_fan_blog_tier3():
    assert classify("https://randomblog.blogspot.com/anime-theory") == 3

def test_unknown_tier4():
    assert classify("https://mystiqora.com/the-rezero-paradox") == 4

def test_weights_default():
    assert tier_weight(1) == 1.50
    assert tier_weight(4) == 0.25

def test_weights_from_config():
    cfg = {"tier_1_weight": 2.0, "tier_3_weight": 0.1}
    assert tier_weight(1, cfg) == 2.0
    assert tier_weight(3, cfg) == 0.1
