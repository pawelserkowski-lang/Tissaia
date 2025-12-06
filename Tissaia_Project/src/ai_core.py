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
        p = "Analyze image orientation. JSON: {\"rotation_needed_degrees\": 0 or 90 or 180 or 270}"
        r = make_request_with_retry(url, {"contents":[{"parts":[{"text":p},{"inlineData":{"mimeType":"image/jpeg","data":b64}}]}],"generationConfig":{"responseMimeType":"application/json"}}, {"Content-Type":"application/json","x-goog-api-key":API_KEY})
        if r and r.status_code==200:
            j = clean_json_response(r.json()["candidates"][0]["content"]["parts"][0]["text"])
            return int(j.get("rotation_needed_degrees", 0))
    except: pass
    return 0

def restore_final(pil_img, out_path):
    try:
        img = aggressive_trim_borders(pil_img)
        buf=io.BytesIO(); img.save(buf,format="JPEG"); b64=base64.b64encode(buf.getvalue()).decode('utf-8')
        url = get_url(MODEL_RESTORATION)
        p = "Restore this photo. Remove scratches, dust, fix lighting. Keep natural skin texture. Return ONLY the image."
        r = make_request_with_retry(url, {"contents":[{"parts":[{"text":p},{"inlineData":{"mimeType":"image/jpeg","data":b64}}]}],"generationConfig":{"temperature":0.3}}, {"Content-Type":"application/json","x-goog-api-key":API_KEY})
        if r and r.status_code==200:
            raw = r.json()["candidates"][0]["content"]["parts"]
            for x in raw:
                if "inlineData" in x:
                    rec_img = Image.open(io.BytesIO(base64.b64decode(x["inlineData"]["data"])))
                    rec_img = apply_super_sharpen(rec_img)
                    rec_img.save(out_path)
                    return True
    except: pass
    return False