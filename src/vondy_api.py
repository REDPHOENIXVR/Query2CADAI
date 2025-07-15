"""
Vondy API integration module.

Provides synchronous and asynchronous functions to generate CAD files from prompts via the Vondy API.

Environment Variables:
    VONDY_BASE_URL: Base URL for the Vondy API (default: https://www.vondy.com/api/cad/generate)
    VONDY_API_KEY: API key for authenticating with Vondy API.
"""

import os
import uuid
import logging
import requests
from pathlib import Path
from typing import Optional

from src.logger import logger

BASE_URL = os.getenv("VONDY_BASE_URL", "https://www.vondy.com/api/cad/generate")
API_KEY = os.getenv("VONDY_API_KEY", "")

def _get_filename_from_cd(cd_header: Optional[str], default_ext: str = ".zip") -> str:
    """
    Parse Content-Disposition header to extract filename, or generate one if absent.
    """
    if cd_header:
        import re
        match = re.search(r'filename="?([^";]+)"?', cd_header)
        if match:
            return match.group(1)
    return str(uuid.uuid4()) + default_ext

def generate_cad(
    prompt: str,
    output_dir: str,
    *,
    timeout: int = 60,
    retries: int = 3
) -> str:
    """
    Generate a CAD file from a text prompt using the Vondy API.

    Args:
        prompt (str): The CAD design prompt.
        output_dir (str): Directory to save the downloaded file.
        timeout (int, optional): Request timeout in seconds. Defaults to 60.
        retries (int, optional): Number of times to retry on failure. Defaults to 3.

    Returns:
        str: Path to the downloaded CAD file.

    Raises:
        RuntimeError: If unable to generate or download the CAD file.
    """
    if not API_KEY:
        logger.error("VONDY_API_KEY environment variable not set.")
        raise RuntimeError("VONDY_API_KEY not set")

    headers = {"Authorization": f"Bearer {API_KEY}"}
    payload = {"prompt": prompt}
    session = requests.Session()

    for attempt in range(1, retries + 1):
        try:
            logger.info(f"Requesting CAD generation from Vondy (attempt {attempt})...")
            resp = session.post(BASE_URL, json=payload, headers=headers, timeout=timeout)
            resp.raise_for_status()
            json_resp = resp.json()
            download_url = json_resp.get("download_url")
            if not download_url:
                raise RuntimeError("No download_url in Vondy response")
            logger.info(f"Downloading generated CAD file from {download_url}")
            get_resp = session.get(download_url, headers=headers, timeout=timeout, stream=True)
            get_resp.raise_for_status()
            filename = _get_filename_from_cd(get_resp.headers.get("Content-Disposition"))
            output_path = Path(output_dir) / filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                for chunk in get_resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"CAD file saved to {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Vondy CAD generation failed: {e}")
            if attempt == retries:
                raise RuntimeError(f"Vondy CAD generation failed after {retries} attempts: {e}")
    # Should not reach here
    raise RuntimeError("Vondy CAD generation failed unexpectedly.")

async def generate_cad_async(
    prompt: str,
    output_dir: str,
    *,
    timeout: int = 60,
    retries: int = 3
) -> str:
    """
    Asynchronous version of generate_cad using aiohttp if available, else falls back to sync.

    Args:
        prompt (str): The CAD design prompt.
        output_dir (str): Directory to save the downloaded file.
        timeout (int, optional): Request timeout in seconds. Defaults to 60.
        retries (int, optional): Number of times to retry on failure. Defaults to 3.

    Returns:
        str: Path to the downloaded CAD file.

    Raises:
        RuntimeError: If unable to generate or download the CAD file.
    """
    try:
        import aiohttp
        import asyncio

        if not API_KEY:
            logger.error("VONDY_API_KEY environment variable not set.")
            raise RuntimeError("VONDY_API_KEY not set")

        headers = {"Authorization": f"Bearer {API_KEY}"}
        payload = {"prompt": prompt}

        for attempt in range(1, retries + 1):
            try:
                logger.info(f"[async] Requesting CAD generation from Vondy (attempt {attempt})...")
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                    async with session.post(BASE_URL, json=payload, headers=headers) as resp:
                        resp.raise_for_status()
                        json_resp = await resp.json()
                        download_url = json_resp.get("download_url")
                        if not download_url:
                            raise RuntimeError("No download_url in Vondy response")
                    logger.info(f"[async] Downloading generated CAD file from {download_url}")
                    async with session.get(download_url, headers=headers) as get_resp:
                        get_resp.raise_for_status()
                        cd = get_resp.headers.get("Content-Disposition")
                        filename = _get_filename_from_cd(cd)
                        output_path = Path(output_dir) / filename
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(output_path, "wb") as f:
                            async for chunk in get_resp.content.iter_chunked(8192):
                                f.write(chunk)
                logger.info(f"[async] CAD file saved to {output_path}")
                return str(output_path)
            except Exception as e:
                logger.error(f"[async] Vondy CAD generation failed: {e}")
                if attempt == retries:
                    raise RuntimeError(f"[async] Vondy CAD generation failed after {retries} attempts: {e}")
        raise RuntimeError("[async] Vondy CAD generation failed unexpectedly.")
    except ImportError:
        logger.warning("aiohttp not installed; falling back to synchronous generate_cad()")
        # fallback to synchronous version in a thread
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, generate_cad, prompt, output_dir, timeout, retries
        )