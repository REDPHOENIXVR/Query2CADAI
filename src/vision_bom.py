import os
import json
import time
import logging

def _lazy_import_openai():
    try:
        import openai
        return openai
    except ImportError:
        logging.warning("openai not installed; vision_bom will return dummy BOM.")
        return None

def load_image(image_input):
    try:
        from PIL import Image
        if isinstance(image_input, bytes):
            from io import BytesIO
            return Image.open(BytesIO(image_input))
        elif isinstance(image_input, str):
            # treat as file path
            return Image.open(image_input)
        else:
            raise ValueError("Unsupported image_input type.")
    except Exception:
        # Real fallback: create a blank image
        from PIL import Image
        return Image.new("RGB", (128, 128), color="gray")

def extract_bom(image_bytes, prompt_hint=""):
    # Check for OpenAI API key presence
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_PROXY_KEY")
    if not api_key:
        # Return dummy
        bom = {
            "head": {"type": "sphere", "material": "plastic"},
            "torso": {"type": "box", "material": "metal"},
            "legs": [
                {"type": "cylinder", "side": "left", "material": "metal"},
                {"type": "cylinder", "side": "right", "material": "metal"}
            ]
        }
    else:
        openai = _lazy_import_openai()
        if not openai:
            return {}
        prompt = (
            "Given this image, extract a Bill of Materials (BOM) for a humanoid robot as a JSON dict. "
            "Keys: head, torso, legs (list), arms (optional). Each is a dict with type/material/tags. "
            + (prompt_hint or "")
        )
        # Simulate vision API call for now (stub)
        bom = {
            "head": {"type": "sphere", "material": "plastic"},
            "torso": {"type": "box", "material": "metal"},
            "legs": [
                {"type": "cylinder", "side": "left", "material": "metal"},
                {"type": "cylinder", "side": "right", "material": "metal"}
            ]
        }
        # Real code would call vision model

    # Validate keys
    if not (isinstance(bom, dict) and "head" in bom and "torso" in bom and "legs" in bom and isinstance(bom["legs"], list)):
        raise ValueError("BOM must contain at least head, torso, legs (list)")
    ts = int(time.time())
    os.makedirs("results/bom", exist_ok=True)
    outpath = f"results/bom/{ts}.json"
    with open(outpath, "w") as f:
        json.dump(bom, f, indent=2)
    return bom