import time, random, io, base64, requests
from PIL import Image

def encode_image_optimized(path, max_size=3500, quality=93):
    with Image.open(path) as img:
        if img.mode != "RGB": img = img.convert("RGB")
        if max(img.size) > max_size: img.thumbnail((max_size, max_size))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

def make_request_with_retry(url, json_payload, headers, max_retries=5):
    """
    Makes a POST request with exponential backoff and jitter.
    """
    base_delay = 2
    for i in range(max_retries):
        try:
            r = requests.post(url, json=json_payload, headers=headers, timeout=600)
            # If rate limited (429) or server error (5xx), we should also retry
            if r.status_code == 429 or 500 <= r.status_code < 600:
                # Raise exception to trigger the except block logic
                r.raise_for_status()
            return r
        except Exception as e:
            if i == max_retries - 1:
                print(f"[ERROR] Max retries reached for {url}: {e}")
                return None

            # Exponential backoff with jitter: sleep = 2^i + random(0,1)
            delay = (base_delay ** i) + random.uniform(0, 1)
            print(f"[WARN] Request failed, retrying in {delay:.2f}s... (Attempt {i+1}/{max_retries})")
            time.sleep(delay)
    return None
