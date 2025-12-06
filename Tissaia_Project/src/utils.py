import time
import random
import io
import base64
import requests
import logging
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def encode_image_optimized(path, max_size=3500, quality=93):
    with Image.open(path) as img:
        if img.mode != "RGB": img = img.convert("RGB")
        if max(img.size) > max_size: img.thumbnail((max_size, max_size))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

def make_request_with_retry(url, json_payload, headers, max_retries=5):
    for i in range(max_retries):
        try:
            r = requests.post(url, json=json_payload, headers=headers, timeout=600)
            if r.status_code in [200, 404, 403, 400]:
                return r

            # Retryable errors: 429 (Rate Limit), 500+, 502, 503, 504
            if r.status_code == 429 or r.status_code >= 500:
                # Exponential Backoff with Jitter
                # Base 2 seconds, multiplier 2^i
                sleep_time = (2 ** i) + random.uniform(0, 1)
                logger.warning(f"Request failed with {r.status_code}. Retrying in {sleep_time:.2f}s (Attempt {i+1}/{max_retries})")
                time.sleep(sleep_time)
                continue

            return r
        except Exception as e:
            sleep_time = (2 ** i) + random.uniform(0, 1)
            logger.warning(f"Request exception: {str(e)}. Retrying in {sleep_time:.2f}s")
            time.sleep(sleep_time)

    logger.error(f"Failed to get valid response from {url} after {max_retries} retries.")
    return None
