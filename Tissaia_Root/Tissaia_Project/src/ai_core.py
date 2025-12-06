import json, re, io, base64, os, time
from PIL import Image
from src.config import API_KEY, MODEL_RESTORATION
from src.utils import make_request_with_retry
from src.graphics import apply_super_sharpen

def get_url(m):
    clean = m.replace('models/', '')
    return f"https://generativelanguage.googleapis.com/v1beta/models/{clean}:generateContent"

def smart_crop_10_percent(img):
    w, h = img.size
    margin_w, margin_h = max(10, int(w*0.1)), max(10, int(h*0.1))
    return img.crop((margin_w, margin_h, w-margin_w, h-margin_h))

def restore_image_generative(pil_img, save_path):
    url = get_url(MODEL_RESTORATION)
    safe_img = smart_crop_10_percent(pil_img)
    buf = io.BytesIO()
    safe_img.save(buf, format="JPEG", quality=95)
    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    prompt = (
        "ROLE: Forensic Specialist. "
        "TASK: Restore vintage photo. REMOVE: Dust, scratches. FACE: Sharp eyes, texture. STYLE: Kodak Portra 400. "
        "OUTPUT: Return the restored image in standard API response structure."
    )

    payload = {
        "contents": [{"parts": [{"text": prompt}, {"inlineData": {"mimeType": "image/jpeg", "data": b64}}]}],
        "generationConfig": {
            "temperature": 0.35,
            "maxOutputTokens": 8192,
            # "responseMimeType": "application/json" # Not all models support this yet, keeping it safe
        }
    }

    try:
        r = make_request_with_retry(url, payload, {"Content-Type": "application/json", "x-goog-api-key": API_KEY})
        if r and r.status_code == 200:
            data = r.json()
            if "candidates" in data and len(data["candidates"]) > 0:
                candidate = data["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    parts = candidate["content"]["parts"]
                    # Look for inlineData part
                    for part in parts:
                        if "inlineData" in part:
                             img_data = base64.b64decode(part["inlineData"]["data"])
                             final = apply_super_sharpen(Image.open(io.BytesIO(img_data)))
                             final.save(save_path)
                             return True
            print(f"[ERROR] Invalid API response structure: {data.keys()}")
        elif r:
            print(f"[ERROR] API Error {r.status_code}: {r.text}")
    except Exception as e:
        print(f"[ERROR] restore_image_generative exception: {e}")
    return False
