import json, re, io, base64, cv2
import numpy as np
from PIL import Image
from src.config import API_KEY, MODEL_DETECTION, MODEL_RESTORATION
from src.utils import make_request_with_retry, encode_image_optimized
from src.graphics import aggressive_trim_borders, apply_super_sharpen

def get_url(m): return f"https://generativelanguage.googleapis.com/v1beta/models/{m.replace('models/','')}:generateContent"

def clean_json_response(text):
    try:
        return json.loads(re.sub(r"```json|```", "", text).strip())
    except: return None

def restore_locally(pil_img, save_path):
    try:
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        denoised = cv2.fastNlMeansDenoisingColored(cv_img, None, 10, 10, 7, 21)
        detail = cv2.detailEnhance(denoised, sigma_s=10, sigma_r=0.15)
        final_pil = Image.fromarray(cv2.cvtColor(detail, cv2.COLOR_BGR2RGB))
        final_pil = apply_super_sharpen(aggressive_trim_borders(final_pil))
        final_pil.save(save_path)
        return True
    except Exception as e:
        print(f"Local restore failed: {e}")
        return False

def detect_rotation_strict(path):
    url = get_url(MODEL_DETECTION)
    try:
        with Image.open(path) as img:
            img.thumbnail((800,800))
            buf=io.BytesIO(); img.save(buf,format="JPEG"); b64=base64.b64encode(buf.getvalue()).decode('utf-8')
        p = 'Look at people. Task: Direction of TOP OF HEADS? Options: "UP","DOWN","LEFT","RIGHT". JSON: {"direction": "UP"}'
        r = make_request_with_retry(url, {"contents":[{"parts":[{"text":p},{"inlineData":{"mimeType":"image/jpeg","data":b64}}]}],"generationConfig":{"responseMimeType":"application/json"}}, {"Content-Type":"application/json","x-goog-api-key":API_KEY})
        if r and r.status_code==200:
            j = clean_json_response(r.json()["candidates"][0]["content"]["parts"][0]["text"])
            if j: return {"UP":0, "DOWN":180, "RIGHT":90, "LEFT":270}.get(j.get("direction","UP"), 0)
    except: pass
    return 0

def detect_corners(path):
    url = get_url(MODEL_DETECTION)
    try:
        b64 = encode_image_optimized(path, 1500)
        p = 'Analyze scan. Detect 4 EXACT CORNERS. JSON: {"photos": [{"corners":[[x,y],[x,y],[x,y],[x,y]]}]}. Scale 0-1000.'
        r = make_request_with_retry(url, {"contents":[{"parts":[{"text":p},{"inlineData":{"mimeType":"image/jpeg","data":b64}}]}],"generationConfig":{"responseMimeType":"application/json"}}, {"Content-Type":"application/json","x-goog-api-key":API_KEY})
        if r and r.status_code==200:
            return clean_json_response(r.json()["candidates"][0]["content"]["parts"][0]["text"])
    except: pass
    return None

def restore_final(path, out):
    # Phase 1: Try AI (Gemini 3)
    url = get_url(MODEL_RESTORATION)
    try:
        b64 = encode_image_optimized(path, 4000)
        p = "Restore photo: Fix geometry, remove dust, sharpen faces. Return IMAGE."
        r = make_request_with_retry(url, {"contents":[{"parts":[{"text":p},{"inlineData":{"mimeType":"image/jpeg","data":b64}}]}],"generationConfig":{"temperature":0.35}}, {"Content-Type":"application/json","x-goog-api-key":API_KEY})
        
        if r and r.status_code==200:
            data = r.json()
            if "candidates" in data:
                parts = data["candidates"][0]["content"]["parts"]
                for x in parts:
                    if "inlineData" in x:
                        img = Image.open(io.BytesIO(base64.b64decode(x["inlineData"]["data"])))
                        img = apply_super_sharpen(aggressive_trim_borders(img))
                        img.save(out)
                        print("AI Restore SUCCESS")
                        return True
    except: pass
    
    # Phase 2: Local Fallback
    try:
        with Image.open(path) as img:
            return restore_locally(img, out)
    except: return False