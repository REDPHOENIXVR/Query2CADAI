from src.cache import make_cache_key

def test_llm_cache_key_different_prompts():
    k1 = make_cache_key("prompt1", "gpt-4")
    k2 = make_cache_key("prompt2", "gpt-4")
    assert k1 != k2

def test_llm_cache_key_same():
    k1 = make_cache_key("prompt", "gpt-4")
    k2 = make_cache_key("prompt", "gpt-4")
    assert k1 == k2