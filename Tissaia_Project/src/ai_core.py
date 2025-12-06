import json
import re
import io
import base64
import os
import time
from PIL import Image
from src.config import API_KEY, MODEL_RESTORATION
from src.utils import make_request_with_retry
from src.graphics import apply_super_sharpen

def get_url(m):
    clean_model = m.replace('models/', '')
    return f"https://generativelanguage.googleapis.com/v1beta/models/{clean_model}:generateContent"

def smart_crop_10_percent(img):
    w, h = img.size
    margin_w = max(10, int(w * 0.10))
    margin_h = max(10, int(h * 0.10))
    box = (margin_w, margin_h, w - margin_w, h - margin_h)
    try: return img.crop(box)
    except: return img

def restore_image_generative(pil_img, save_path):
    # STRICT MODE: Gemini 3 Pro Image Preview ONLY.
    url = get_url(MODEL_RESTORATION)
    
    # 1. Prepare Input
    safe_img = smart_crop_10_percent(pil_img)
    buf = io.BytesIO()
    safe_img.save(buf, format="JPEG", quality=95)
    b64_data = base64.b64encode(buf.getvalue()).decode('utf-8')

    # 2. PROMPT INŻYNIERYJNY
    prompt = \"\"\"
    ROLE: Forensic Photo Restoration Specialist using Gemini 3 Vision.
    INPUT: A vintage photo crop (center 90%).
    
    TASK: Reconstruct high-fidelity image.
    
    1. DIGITAL HYGIENE: Remove all dust, scratches, tears, and scan artifacts.
    2. FORENSIC FACE RECONSTRUCTION: Sharp, symmetric eyes. Realistic skin texture.
    3. HDR STUDIO REMASTERING: "Kodak Portra 400" aesthetic. Soft studio lighting.
    
    OUTPUT: Return ONLY the raw restored image.
    \"\"\"

    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inlineData": {"mimeType": "image/jpeg", "data": b64_data}}
            ]
        }],
        "generationConfig": {
            "temperature": 0.35,
            "topK": 32,
            "topP": 0.95,
            "maxOutputTokens": 8192
        }
    }

    headers = {"Content-Type": "application/json", "x-goog-api-key": API_KEY}
    
    try:
        response = make_request_with_retry(url, payload, headers)
        
        if response and response.status_code == 200:
            data = response.json()
            if "candidates" in data and data["candidates"][0]["content"]["parts"]:
                for part in data["candidates"][0]["content"]["parts"]:
                    if "inlineData" in part:
                        img_data = base64.b64decode(part["inlineData"]["data"])
                        restored_img = Image.open(io.BytesIO(img_data))
                        
                        final_img = apply_super_sharpen(restored_img)
                        final_img.save(save_path)
                        return True
        elif response:
            print(f"API ERROR {response.status_code}: {response.text[:200]}")
            
    except Exception as e:
        print(f"CRITICAL EXCEPTION: {str(e)}")

    return False