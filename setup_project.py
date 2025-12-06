import os, sys, base64, zlib

# Tissaia Project - One-Click Installer (WARLORD_V23_ANTI_STRIP)
# Status: AGGRESSIVE FILTERING.
# Fixes: 
# 1. "Strip Killer 2.0": Raised minimum area threshold to 7% (0.07) to eliminate all strips.
# 2. Retains "Precise FloodFill" and "Rotated Cuts".

def w(path, content):
    d = os.path.dirname(path)
    if d and not os.path.exists(d): os.makedirs(d)
    with open(path, 'w', encoding='utf-8') as f: f.write(content.strip())
    print(f"[+] {path}")

def main():
    base = "Tissaia_Project"
    
    # 1. DEPENDENCIES
    w(f"{base}/requirements.txt", """requests>=2.31.0
python-dotenv>=1.0.0
Pillow>=10.0.0
numpy>=1.24.0
opencv-python-headless>=4.8.0
""")
    
    if not os.path.exists(f"{base}/.env"):
        w(f"{base}/.env", "GOOGLE_API_KEY=WKLEJ_SWOJ_KLUCZ_TUTAJ")
    
    # 2. DOCKER
    w(f"{base}/Dockerfile", r"""
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libgl1 libglx-mesa0 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN useradd -m -u 1000 appuser
COPY . .
RUN chown -R appuser:appuser /app
USER appuser
CMD ["python", "main.py"]
""")

    w(f"{base}/START.bat", r"""
@echo off
echo [WARLORD] Tissaia V23 (Anti-Strip Protocol)...
if not exist .env ( echo [ERROR] Brak .env! & pause & exit )
docker compose up --build
pause
""")
    
    w(f"{base}/docker-compose.yml", r"""
services:
  app:
    build: .
    image: tissaia-app
    volumes:
      - .:/app
    env_file:
      - .env
    user: "1000:1000"
""")

    # 3. CONFIG
    w(f"{base}/src/config.py", r"""
import os
from dotenv import load_dotenv

load_dotenv()
if not os.environ.get("GOOGLE_API_KEY"):
    try:
        with open(".env", "r") as f:
            for line in f:
                if line.strip().startswith("GOOGLE_API_KEY="):
                    k = line.strip().split("=", 1)[1]
                    os.environ["GOOGLE_API_KEY"] = k.strip().strip('"').strip("'")
    except: pass

API_KEY = os.environ.get("GOOGLE_API_KEY")
MODEL_DETECTION = "models/gemini-2.0-flash" 
MODEL_RESTORATION = "models/gemini-2.0-flash"

MAX_WORKERS = 8
INPUT_ZIP = "zdjecia.zip"
OUTPUT_DIR = "odnowione_final"
TEMP_DIR = "temp_input"
""")

    # 4. UTILS
    w(f"{base}/src/utils.py", r"""
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
    for i in range(max_retries):
        try:
            r = requests.post(url, json=json_payload, headers=headers, timeout=600)
            if r.status_code == 404: 
                if "v1beta" in url: url = url.replace("v1beta", "v1alpha"); continue
                return r
            if r.status_code in [429, 500, 503]:
                wait = (i + 1) * 2 + random.random()
                time.sleep(wait)
                continue
            return r
        except: time.sleep(2)
    return None
""")

    # 5. GRAPHICS (V23: 7% THRESHOLD)
    w(f"{base}/src/graphics.py", r'''
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageChops

def cv2_to_pil(img_cv):
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

def crop_min_area_rect(img, rect):
    box = cv2.boxPoints(rect)
    box = np.int32(box) 
    
    width = int(rect[1][0])
    height = int(rect[1][1])

    src_pts = box.astype("float32")
    s = src_pts.sum(axis=1)
    tl = src_pts[np.argmin(s)]
    br = src_pts[np.argmax(s)]
    diff = np.diff(src_pts, axis=1)
    tr = src_pts[np.argmin(diff)]
    bl = src_pts[np.argmax(diff)]
    
    ordered_src = np.array([tl, tr, br, bl], dtype="float32")

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst_pts = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(ordered_src, dst_pts)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    return warped

def is_valid_cut(w, h, total_area_img):
    area = w * h
    ratio = w / float(h) if h > 0 else 0
    
    # V23: Aggressive Strip Killer
    # Minimum 7% of total scan area to be considered a photo.
    # For a scan with 8 photos, each is ~12.5%, so 7% is safe.
    if area < (total_area_img * 0.07): return False 
    
    if ratio < 0.2 or ratio > 5.0: return False
    
    if min(w, h) < 50: return False
    
    return True

def remove_inner_rectangles(candidates):
    if not candidates: return []
    candidates = sorted(candidates, key=lambda x: x[1][2]*x[1][3], reverse=True)
    kept = []
    
    for i, current in enumerate(candidates):
        rect_data, (x1, y1, w1, h1) = current
        is_inner = False
        area1 = w1 * h1
        
        for _, (x2, y2, w2, h2) in kept:
            ix1 = max(x1, x2); iy1 = max(y1, y2)
            ix2 = min(x1+w1, x2+w2); iy2 = min(y1+h1, y2+h2)
            if ix1 < ix2 and iy1 < iy2:
                inter_area = (ix2 - ix1) * (iy2 - iy1)
                if inter_area / float(area1) > 0.85: 
                    is_inner = True
                    break
        if not is_inner:
            kept.append(current)
    return kept

def strategy_flood_fill_precise(img):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 21, 4)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open, iterations=2)
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def strategy_flood_fill_heavy(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 25, 6)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def find_cuts_robust(pil_image_path):
    try:
        img_orig = cv2.imread(pil_image_path)
        if img_orig is None: return [], None
        
        scale = 1.0
        h_orig, w_orig = img_orig.shape[:2]
        if max(h_orig, w_orig) > 2000:
            scale = 0.5
            img = cv2.resize(img_orig, (0,0), fx=scale, fy=scale)
        else:
            img = img_orig.copy()
            
        total_area = img.shape[0] * img.shape[1]
        
        strategies = [
            ("FLOOD_PRECISE", lambda i: strategy_flood_fill_precise(i)),
            ("FLOOD_HEAVY", lambda i: strategy_flood_fill_heavy(i))
        ]
        
        best_candidates = []
        debug_img = img.copy()
        
        for name, method in strategies:
            print(f"   [STRATEGY] Testing {name}...")
            contours = method(img)
            candidates = []
            
            for c in contours:
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)
                box = np.int32(box)
                x, y, w, h = cv2.boundingRect(box) 
                
                if is_valid_cut(w, h, total_area):
                    candidates.append((rect, (x, y, w, h)))
            
            clean = remove_inner_rectangles(candidates)
            
            if len(clean) >= 1:
                best_candidates = clean
                print(f"   [ACCEPTED] {name} found {len(best_candidates)} photos.")
                for (rect, _) in best_candidates:
                    box = cv2.boxPoints(rect)
                    box = np.int32(box)
                    cv2.drawContours(debug_img, [box], 0, (0, 255, 0), 4)
                    cx, cy = int(rect[0][0]), int(rect[0][1])
                    cv2.putText(debug_img, name, (cx-20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                break 
        
        final_cuts = []
        
        for (rect, _) in best_candidates:
            (cx, cy), (w, h), ang = rect
            cx /= scale; cy /= scale
            w /= scale; h /= scale
            scaled_rect = ((cx, cy), (w, h), ang)
            cut = crop_min_area_rect(img_orig, scaled_rect)
            final_cuts.append(cv2_to_pil(cut))
            
        return final_cuts, cv2_to_pil(debug_img)

    except Exception as e:
        print(f"[CV ERROR] {e}")
        return [], None

def aggressive_trim_borders(img):
    try:
        bg = Image.new(img.mode, img.size, img.getpixel((0,0)))
        diff = ImageChops.difference(img, bg)
        diff = ImageChops.add(diff, diff, 2.0, -100)
        bbox = diff.getbbox()
        if bbox: return img.crop(bbox)
    except: pass
    return img

def apply_super_sharpen(img):
    img = ImageEnhance.Contrast(img).enhance(1.1)
    img = ImageEnhance.Sharpness(img).enhance(1.3)
    return img
''')

    # 6. AI CORE
    w(f"{base}/src/ai_core.py", r'''
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
''')

    # 7. WORKFLOW
    w(f"{base}/src/workflow.py", r'''
import os, random
from PIL import Image
from src.config import OUTPUT_DIR
from src.ai_core import detect_rotation_strict, restore_final
from src.graphics import find_cuts_robust

def process(fpath, fname):
    log = []
    temps = []
    try:
        # 1. Rotacja skanu
        ang = detect_rotation_strict(fpath)
        cur = fpath
        if ang != 0:
            rot_name = f"tmp_rot_{random.randint(111,999)}_{fname}"
            Image.open(fpath).rotate(-ang, expand=True).save(rot_name)
            cur = rot_name
            temps.append(rot_name)
        
        # 2. Wycinanie V23 (Anti-Strip)
        print(f">> Analyzing: {fname} ...")
        cuts, map_img = find_cuts_robust(cur)
        
        if map_img:
            map_path = os.path.join(OUTPUT_DIR, f"DEBUG_MAP_{fname}")
            map_img.save(map_path)
            log.append(f"MAP: {map_path}")

        if not cuts:
            log.append(f"WARN: No cuts for {fname}. Using full image.")
            cuts = [Image.open(cur)]
        
        # 3. Renowacja
        for i, cut in enumerate(cuts):
            suf = f"_{i+1}" if len(cuts)>1 else ""
            out = os.path.join(OUTPUT_DIR, f"restored_{os.path.splitext(fname)[0]}{suf}.png")
            
            print(f"   >> Restoring {fname} ({i+1}/{len(cuts)})...")
            if restore_final(cut, out): log.append(f"OK: {out}")
            else: log.append(f"ERR: {fname} part {i+1}")
            
    except Exception as e: log.append(f"CRASH {fname}: {e}")
    finally:
        for t in temps: 
            if os.path.exists(t): 
                try: os.remove(t)
                except: pass
    return log
''')

    # 8. MAIN
    w(f"{base}/main.py", r'''
import os, zipfile, shutil, concurrent.futures
from src.config import INPUT_ZIP, OUTPUT_DIR, TEMP_DIR, MAX_WORKERS
from src.workflow import process

if __name__ == "__main__":
    print("=== TISSAIA V23: ANTI-STRIP ===")
    
    if os.path.exists(OUTPUT_DIR):
        print("[INIT] Cleaning old artifacts...")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    
    if os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR)
    
    if not os.path.exists(INPUT_ZIP): 
        print(f"CRITICAL: Nie znaleziono {INPUT_ZIP}!")
        exit(1)
        
    print("[INIT] Unzipping...")
    with zipfile.ZipFile(INPUT_ZIP,'r') as z: z.extractall(TEMP_DIR)
    
    fs = [os.path.join(r,f) for r,_,x in os.walk(TEMP_DIR) for f in x if f.lower().endswith(('jpg','png','jpeg'))]
    print(f"[START] Processing {len(fs)} scans with {MAX_WORKERS} threads.")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        ft = {ex.submit(process, f, os.path.basename(f)):f for f in fs}
        for f in concurrent.futures.as_completed(ft):
            for l in f.result(): print(l)
            
    try: shutil.rmtree(TEMP_DIR)
    except: pass
    print("\n[DONE] Mission Complete.")
''')

    print(f"\n[+] Tissaia V23 Installed. \n1. python setup_project.py\n2. Tissaia_Project\\START.bat")

if __name__ == "__main__":
    main()