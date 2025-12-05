import json, re, io, base64
from PIL import Image
from src.config import API_KEY, MODEL_DETECTION, MODEL_RESTORATION
from src.utils import make_request_with_retry, encode_image_optimized
from src.graphics import aggressive_trim_borders, apply_super_sharpen

def get_url(m): return f"https://generativelanguage.googleapis.com/v1beta/models/{m.replace('models/','')}:generateContent"

def detect_rotation_strict(path):
    # Semantyczna detekcja: Gdzie jest gora glowy?
    url = get_url(MODEL_DETECTION)
    try:
        with Image.open(path) as img:
            img.thumbnail((800,800))
            buf=io.BytesIO(); img.save(buf,format="JPEG"); b64=base64.b64encode(buf.getvalue()).decode('utf-8')
        p = "Look at people. Task: Direction of TOP OF HEADS? Options: 'UP','DOWN','LEFT','RIGHT'. JSON: {\"direction\": \"UP\"}"
        r = make_request_with_retry(url, {"contents":[{"parts":[{"text":p},{"inlineData":{"mimeType":"image/jpeg","data":b64}}]}],"generationConfig":{"responseMimeType":"application/json"}}, {"Content-Type":"application/json","x-goog-api-key":API_KEY})
        if r and r.status_code==200:
            d = json.loads(r.json()["candidates"][0]["content"]["parts"][0]["text"]).get("direction","UP")
            return {"UP":0, "DOWN":180, "RIGHT":90, "LEFT":270}.get(d, 0)
    except: pass
    return 0

def detect_corners(path):
    url = get_url(MODEL_DETECTION)
    try:
        b64 = encode_image_optimized(path, 1500)
        p = "Analyze scan. Detect 4 EXACT CORNERS. JSON: {\"photos\": [{\"corners\":[[x,y],[x,y],[x,y],[x,y]]}]}. Scale 0-1000."
        r = make_request_with_retry(url, {"contents":[{"parts":[{"text":p},{"inlineData":{"mimeType":"image/jpeg","data":b64}}]}],"generationConfig":{"responseMimeType":"application/json"}}, {"Content-Type":"application/json","x-goog-api-key":API_KEY})
        if r and r.status_code==200:
            return json.loads(re.sub(r"```json|```","", r.json()["candidates"][0]["content"]["parts"][0]["text"]).strip())
    except: pass
    return None

def restore_final(path, out):
    url = get_url(MODEL_RESTORATION)
    try:
        b64 = encode_image_optimized(path, 4000)
        p = "Restore photo: Fix geometry (inpaint corners), remove flash glare/dust, sharpen faces, natural skin. Full image, no borders."
        r = make_request_with_retry(url, {"contents":[{"parts":[{"text":p},{"inlineData":{"mimeType":"image/jpeg","data":b64}}]}],"generationConfig":{"temperature":0.35}}, {"Content-Type":"application/json","x-goog-api-key":API_KEY})
        if r and r.status_code==200:
            data = r.json()
            if "candidates" in data:
                raw = data["candidates"][0]["content"]["parts"]
                for x in raw:
                    if "inlineData" in x:
                        img = Image.open(io.BytesIO(base64.b64decode(x["inlineData"]["data"])))
                        img = apply_super_sharpen(aggressive_trim_borders(img))
                        img.save(out)
                        return True
    except: pass
    return False