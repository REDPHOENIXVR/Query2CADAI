import shelve
import hashlib
import os
from src.logger import get_logger

logger = get_logger("cache")

CACHE_PATH = "data/cache/shelve_cache"
os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)

def make_cache_key(prompt, model):
    m = hashlib.sha256()
    m.update(prompt.encode("utf-8"))
    m.update(str(model).encode("utf-8"))
    return m.hexdigest()

def cached_get_answers(model, prompt, *args, **kwargs):
    key = make_cache_key(prompt, model)
    with shelve.open(CACHE_PATH) as db:
        if key in db:
            logger.info(f"Cache hit for model={model}")
            return db[key]
        else:
            logger.info(f"Cache miss for model={model}, generating...")
            from src.llm import get_answers
            result = get_answers(model, prompt, *args, **kwargs)
            db[key] = result
            return result