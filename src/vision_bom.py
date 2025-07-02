import os
import json
import time
import logging

try:
    from PIL import Image
except ImportError:
    Image = None

def extract_bom(image_path):
    if Image is None:
        logging.warning("PIL not available, returning dummy BOM")
        # Fallback: return a dummy BOM
        return [{"category": "unknown", "model": "unknown", "qty": 1}]
    # ...implementation...

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

def extract_bom(image_bytes_or_path, prompt_hint=""):
    """
    Given an image (file path or bytes), calls OpenAI Vision endpoint (if API key and openai>=1.0 installed)
    to extract a Bill of Materials (BOM) as JSON. Falls back to current dummy BOM on failure.
    Image is resized to max 1024x1024 before sending.
    """
    import io

    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_PROXY_KEY")
    bom = None
    vision_attempted = False

    # Accept file path or bytes, and resize image if needed
    try:
        if Image is None:
            raise ImportError("PIL not available")
        img = load_image(image_bytes_or_path)
        max_dim = 1024
        if max(img.size) > max_dim:
            ratio = max_dim / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_bytes = buf.getvalue()
    except Exception as e:
        logging.warning(f"Image loading/resizing failed: {e}")
        img_bytes = image_bytes_or_path if isinstance(image_bytes_or_path, bytes) else None

    if api_key and img_bytes:
        try:
            import openai
            from packaging import version
            openai_version = getattr(openai, "__version__", "0.0.0")
            if version.parse(openai_version) >= version.parse("1.0.0"):
                client = openai.OpenAI(api_key=api_key)
                prompt = (
                    "Given this image, extract a Bill of Materials (BOM) for a humanoid robot as a JSON dict. "
                    "Keys: head, torso, legs (list), arms (optional). Each is a dict with type/material/tags. "
                    + (prompt_hint or "")
                )
                response = client.chat.completions.create(
                    model="gpt-4-vision-preview",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"type": "png", "data": img_bytes}},
                            ],
                        }
                    ],
                    max_tokens=512,
                    temperature=0,
                )
                raw = response.choices[0].message.content.strip()
                vision_attempted = True
                try:
                    # Try pydantic validation first
                    from pydantic import BaseModel
                    class LegModel(BaseModel):
                        type: str
                        side: str = None
                        material: str = None
                        tags: list = []
                    class BOMModel(BaseModel):
                        head: dict
                        torso: dict
                        legs: list
                        arms: list = []
                    import json as _json
                    parsed = _json.loads(raw)
                    result = BOMModel(**parsed).model_dump()
                    bom = result
                except Exception:
                    import json as _json
                    try:
                        bom = _json.loads(raw)
                    except Exception:
                        bom = None
        except Exception as e:
            logging.warning(f"Vision BOM API call failed: {e}")
            bom = None

    # Fallback: Dummy BOM
    if not bom:
        bom = {
            "head": {"type": "sphere", "material": "plastic"},
            "torso": {"type": "box", "material": "metal"},
            "legs": [
                {"type": "cylinder", "side": "left", "material": "metal"},
                {"type": "cylinder", "side": "right", "material": "metal"}
            ]
        }

    # Validate keys
    if not (isinstance(bom, dict) and "head" in bom and "torso" in bom and "legs" in bom and isinstance(bom["legs"], list)):
        raise ValueError("BOM must contain at least head, torso, legs (list)")
    ts = int(time.time())
    os.makedirs("results/bom", exist_ok=True)
    outpath = f"results/bom/{ts}.json"
    with open(outpath, "w") as f:
        json.dump(bom, f, indent=2)
    return bom