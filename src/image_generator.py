import os
import uuid
import logging
import base64

def _lazy_import_openai():
    try:
        import openai
        return openai
    except ImportError:
        logging.warning("OpenAI package not available. Falling back to placeholder image generation.")
        return None

def generate_image(prompt: str, size: str = "1024x1024", outdir: str = "results/images") -> str:
    """
    Generates an image from a text prompt using OpenAI's API if available, otherwise creates a placeholder PNG.
    Returns the local file path to the generated image.
    """
    os.makedirs(outdir, exist_ok=True)
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    openai_mod = _lazy_import_openai()
    image_model = os.environ.get("OPENAI_IMAGE_MODEL", "dall-e-3")
    if openai_mod and openai_api_key:
        try:
            import requests
            client = openai_mod.OpenAI(api_key=openai_api_key)
            # Truncate prompt if necessary
            max_chars = int(os.environ.get("OPENAI_IMAGE_PROMPT_MAX_CHARS", 4000))
            truncated_prompt = None
            if len(prompt) > max_chars:
                logging.info(
                    f"Prompt length {len(prompt)} exceeds maximum {max_chars}, truncating."
                )
                truncated_prompt = prompt[: max_chars - 3] + "..."
                logging.info(
                    f"Using truncated prompt of length {len(truncated_prompt)} for image generation."
                )

            prompt_to_use = truncated_prompt if truncated_prompt is not None else prompt

            logging.info(f"Generating image for prompt: {prompt_to_use!r} (model: {image_model}, size: {size})")
            response = client.images.generate(
                model=image_model,
                prompt=prompt_to_use,
                n=1,
                size=size,
                response_format="url"
            )
            image_url = response.data[0].url
            img_data = requests.get(image_url).content
            fname = f"{uuid.uuid4().hex}.png"
            outpath = os.path.join(outdir, fname)
            with open(outpath, "wb") as f:
                f.write(img_data)
            logging.info(f"Image generated and saved to: {outpath}")
            return outpath
        except Exception as e:
            logging.warning(f"OpenAI image generation failed, using placeholder. Reason: {e}")
    else:
        logging.warning("OpenAI image generation unavailable. Using placeholder image.")

    # Placeholder fallback using Pillow
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        raise RuntimeError("Pillow is required for placeholder image generation")

    img = Image.new("RGB", (512, 512), (180, 180, 180))
    draw = ImageDraw.Draw(img)
    font = None
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
    text = prompt.strip() or "(no prompt)"
    lines = []
    max_width = 480
    # Simple word wrap
    for word in text.split():
        if not lines or draw.textlength(lines[-1] + " " + word, font=font) > max_width:
            lines.append(word)
        else:
            lines[-1] += " " + word
    y = 220 - (len(lines) * 15)
    for line in lines:
        w = draw.textlength(line, font=font)
        draw.text(((512 - w) / 2, y), line, fill=(80, 80, 80), font=font)
        y += 30

    placeholder_fname = f"placeholder_{uuid.uuid4().hex[:8]}.png"
    outpath = os.path.join(outdir, placeholder_fname)
    img.save(outpath, "PNG")
    logging.info(f"Placeholder image saved to: {outpath}")
    return outpath