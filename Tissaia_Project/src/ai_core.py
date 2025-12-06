import json, re, io, base64
from PIL import Image
from src.config import API_KEY, MODEL_DETECTION, MODEL_RESTORATION
from src.utils import make_request_with_retry, encode_image_optimized
from src.graphics import aggressive_trim_borders, apply_super_sharpen

def get_url(m): return f"https://generativelanguage.googleapis.com/v1beta/models/{m.replace('models/','')}:generateContent"

def clean_json_response(text):
    try: return json.loads(re.sub(r"```json|```", "", text).strip())
    except: return None

def detect_rotation_strict(path):
    url = get_url(MODEL_DETECTION)
    try:
        with Image.open(path) as img:
            img.thumbnail((800,800))
            buf=io.BytesIO(); img.save(buf,format="JPEG"); b64=base64.b64encode(buf.getvalue()).decode('utf-8')
        p = "Is the image UPSIDE_DOWN? Answer JSON: {\"is_upside_down\": true/false}"
        r = make_request_with_retry(url, {"contents":[{"parts":[{"text":p},{"inlineData":{"mimeType":"image/jpeg","data":b64}}]}],"generationConfig":{"responseMimeType":"application/json"}}, {"Content-Type":"application/json","x-goog-api-key":API_KEY})
        if r and r.status_code==200:
            j = clean_json_response(r.json()["candidates"][0]["content"]["parts"][0]["text"])
            if j and j.get("is_upside_down", False): return 180
    except: pass
    return 0

def restore_final(pil_img, out_path):
    try:
        img = aggressive_trim_borders(pil_img)
        img = apply_super_sharpen(img)
        img.save(out_path)
        return True
    except: return False