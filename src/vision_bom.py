import os
import json
import time
import logging
import base64

# Configurable Vision Model (default: gpt-4o, override with OPENAI_VISION_MODEL)
VISION_MODEL = os.environ.get("OPENAI_VISION_MODEL", "gpt-4o")
VISION_MODEL_FALLBACKS = ["gpt-4o-mini", "gpt-4o"]
VISION_MODEL_DEPRECATED = {"gpt-4-vision-preview", "gpt-4-vision", "gpt-4-vision-preview-1106"}

def _get_vision_model(requested=None):
    """
    Returns the model name for Vision API.
    Logs a warning if the model is likely deprecated.
    """
    model = requested if requested is not None else VISION_MODEL
    if model in VISION_MODEL_DEPRECATED:
        logging.warning(f"OpenAI Vision model '{model}' is deprecated or unsupported. Consider using 'gpt-4o'.")
    return model

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

from src.bom_utils import validate_bom

def extract_bom(image_bytes_or_path, prompt_hint=""):
    """
    Given an image (file path or bytes), calls OpenAI Vision endpoint (if API key and openai>=1.0 installed)
    to extract a Bill of Materials (BOM) as JSON. Falls back to current dummy BOM on failure.
    Image is resized to max 1024x1024 before sending.
    Returns: (corrected_bom, warnings)
    """
    import io

    # Only use OPENAI_API_KEY for OpenAI Vision endpoint (never OPENROUTER_API_KEY)
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_PROXY_KEY")
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

    def call_openai_vision(model_name):
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

                # --- Begin fix for OpenAI Vision API image payload ---
                b64 = base64.b64encode(img_bytes).decode('ascii')
                data_url = f"data:image/png;base64,{b64}"
                data_url_size = len(data_url.encode('utf-8'))
                if data_url_size > 19 * 1024 * 1024:  # 19 MB safeguard, API limit is 20 MB
                    logging.warning(
                        f"Image data URL size is {data_url_size/1024/1024:.2f} MB (>19 MB, may fail with OpenAI Vision API!)"
                    )
                # --- End fix ---

                response = client.chat.completions.create(
                    model=_get_vision_model(model_name),
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": data_url}},
                            ],
                        }
                    ],
                    max_tokens=512,
                    temperature=0,
                )
                raw = response.choices[0].message.content.strip()
                return raw, None
        except Exception as e:
            return None, e
        return None, Exception("OpenAI Vision API unavailable or version < 1.0.0")

    # Try OpenAI Vision call(s) if possible
    tried_models = []
    last_error = None
    if api_key and img_bytes:
        # Always try user-specified model first (VISION_MODEL)
        models_to_try = [VISION_MODEL]
        # For fallback, only try fallback models not already tried (and only if the default was used)
        if VISION_MODEL not in VISION_MODEL_FALLBACKS:
            fallback_models = VISION_MODEL_FALLBACKS.copy()
        else:
            # if VISION_MODEL was already a fallback, try the others except it
            fallback_models = [m for m in VISION_MODEL_FALLBACKS if m != VISION_MODEL]

        # If the user did not override the model, allow fallback attempts
        allow_fallback = (os.environ.get("OPENAI_VISION_MODEL") is None)

        def is_404_error(exc):
            return hasattr(exc, "status_code") and getattr(exc, "status_code", None) == 404 or \
                "404" in str(exc) or "not found" in str(exc).lower()

        for model_name in models_to_try:
            tried_models.append(model_name)
            raw, err = call_openai_vision(model_name)
            if raw is not None:
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
                if bom:
                    break  # Success
            last_error = err
            # If we get a 404 and fallback is allowed, try fallback models
            if err and is_404_error(err) and allow_fallback:
                for fallback_model in fallback_models:
                    if fallback_model not in tried_models:
                        logging.warning(
                            f"Model '{model_name}' returned 404. Retrying with fallback model '{fallback_model}'."
                        )
                        tried_models.append(fallback_model)
                        raw, err = call_openai_vision(fallback_model)
                        if raw is not None:
                            vision_attempted = True
                            try:
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
                            if bom:
                                break
                        last_error = err
                # Done all fallbacks
            if bom:
                break

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

    # === BOM VALIDATION & WARNING LOGIC ===
    corrected_bom, warnings = validate_bom(bom)

    ts = int(time.time())
    os.makedirs("results/bom", exist_ok=True)
    outpath = f"results/bom/{ts}.json"
    with open(outpath, "w") as f:
        json.dump(corrected_bom, f, indent=2)
    # Save warnings to a dedicated file if any
    if warnings:
        warn_path = f"results/bom/{ts}_warnings.txt"
        with open(warn_path, "w") as wf:
            for w in warnings:
                wf.write(w + "\n")
    return corrected_bom, warnings