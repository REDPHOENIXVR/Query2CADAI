import shelve
import hashlib
import os
from src.logger import get_logger

logger = get_logger("cache")

CACHE_PATH = "data/cache/shelve_cache"
os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)

DEFAULT_TEMP = 0.7

def make_cache_key(prompt, model):
    m = hashlib.sha256()
    m.update(prompt.encode("utf-8"))
    m.update(str(model).encode("utf-8"))
    return m.hexdigest()

def cached_get_answers(
    model,
    prompt,
    api_key=None,
    temp=DEFAULT_TEMP,
    base_url=None,
    *args,
    **kwargs
):
    """
    Cached wrapper for get_answers. Compatible with legacy calls.
    Supports model, prompt, api_key, temp, base_url, plus *args/**kwargs for forward compatibility.
    temp is included in cache key to distinguish between different temperature settings.
    """
    key_prompt = f"{prompt}__temp={temp}"
    key = make_cache_key(key_prompt, model)
    with shelve.open(CACHE_PATH) as db:
        if key in db:
            logger.info(f"Cache hit for model={model} temp={temp}")
            return db[key]
        else:
            logger.info(f"Cache miss for model={model} temp={temp}, generating...")
            from src.llm import get_answers
            result = get_answers(model, api_key, prompt, temp, base_url)
            db[key] = result
            return result