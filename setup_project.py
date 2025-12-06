import os, sys, base64, zlib

# Tissaia Project - Compact Installer (WARLORD_V14_WATERSHED)
# Fixes: Separation logic via Distance Transform & Watershed Algorithm

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
        w(f"{base}/.env", "GOOGLE_API_KEY=TUTAJ_WKLEJ_SWOJ_KLUCZ")
    
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
    
    w(f"{base}/.dockerignore", """
__pycache__
*.pyc
.git
.gitignore
venv
env
temp_input
odnowione_final
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

    w(f"{base}/START.bat", r"""
@echo off
echo [WARLORD] Uruchamianie procedury Tissaia V14...
if not exist .env ( echo [ERROR] Brak .env! & pause & exit )
docker compose up --build
pause
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
MODEL_DETECTION = "models/gemini-1.5-pro"
MODEL_RESTORATION = "models/gemini-1.5-pro"
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
                time.sleep((i + 1) * 2)
                continue
            return r
        except: time.sleep(2)
    return None
""")

    # 5. GRAPHICS (V14 WATERSHED)
    w(f"{base}/src/graphics.py", r'''
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter, ImageChops

def cv2_to_pil(img_cv):
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

def find_cuts_watershed(pil_image_path):
    """
    V14 Strategy: Marker-Controlled Watershed Segmentation
    The only reliable way to separate touching objects (photos).
    """
    try:
        img = cv2.imread(pil_image_path)
        if img is None: return [], None
        original = img.copy()
        
        # 1. Binarization (Otsu is usually safest for scans)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Invert: Photos should be White, Background Black
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 2. Noise Removal (Opening)
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # 3. Sure Background area (Dilate)
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # 4. Sure Foreground area (Distance Transform)
        # This finds the "centers" of the photos, far away from edges
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        # Threshold: We take only the peaks (centers)
        ret, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        
        # 5. Unknown region (Edges)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # 6. Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)
        # Add 1 to all markers so that sure background is not 0, but 1
        markers = markers + 1
        # Mark the region of unknown with 0
        markers[unknown == 255] = 0
        
        # 7. Watershed
        markers = cv2.watershed(img, markers)
        
        # Extract cuts based on markers
        cuts = []
        map_img = original.copy()
        img_area = img.shape[0] * img.shape[1]
        
        # Unique markers (skip -1 for boundaries and 1 for background)
        unique_markers = np.unique(markers)
        
        i = 1
        for m in unique_markers:
            if m <= 1: continue # Skip background and edges
            
            # Create mask for this marker
            mask = np.zeros(gray.shape, dtype="uint8")
            mask[markers == m] = 255
            
            # Find contour of this segment
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: continue
            
            c = contours[0]
            area = cv2.contourArea(c)
            
            # Filter: Min 2% area
            if area < (img_area * 0.02): continue
            
            x, y, w, h = cv2.boundingRect(c)
            
            # Filter Strips
            ratio = w / float(h)
            if ratio > 15 or ratio < 0.05: continue
            
            # Padding
            pad = 10
            x = max(0, x - pad)
            y = max(0, y - pad)
            w = min(img.shape[1] - x, w + 2*pad)
            h = min(img.shape[0] - y, h + 2*pad)
            
            cut = img[y:y+h, x:x+w]
            cuts.append(cv2_to_pil(cut))
            
            # Debug Map: Draw bounding rect and Center
            cv2.rectangle(map_img, (x, y), (x+w, y+h), (0, 255, 0), 4)
            cv2.putText(map_img, f"#{i}", (x+int(w/2)-20, y+int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
            i += 1
            
        return cuts, cv2_to_pil(map_img)

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
    img = ImageEnhance.Contrast(img).enhance(1.2)
    img = ImageEnhance.Color(img).enhance(1.1)
    img = ImageEnhance.Sharpness(img).enhance(1.5)
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
''')

    # 7. WORKFLOW
    w(f"{base}/src/workflow.py", r'''
import os, random
from PIL import Image, ImageDraw
from src.config import OUTPUT_DIR
from src.ai_core import detect_rotation_strict, restore_final
from src.graphics import find_cuts_watershed

def process(fpath, fname):
    log = []
    temps = []
    try:
        # 1. Rotacja
        ang = detect_rotation_strict(fpath)
        cur = fpath
        if ang != 0:
            rot_name = f"tmp_rot_{random.randint(111,999)}_{fname}"
            Image.open(fpath).rotate(ang, expand=True).save(rot_name)
            cur = rot_name
            temps.append(rot_name)
        
        # 2. Wycinanie V14
        cuts, map_img = find_cuts_watershed(cur)
        
        if map_img:
            map_path = os.path.join(OUTPUT_DIR, f"MAP_{fname}")
            map_img.save(map_path)
            log.append(f"MAP: {map_path}")
        else:
            try:
                img_debug = Image.open(cur)
                d = ImageDraw.Draw(img_debug)
                d.rectangle([0,0,img_debug.width-1, img_debug.height-1], outline="red", width=10)
                map_path = os.path.join(OUTPUT_DIR, f"MAP_FAIL_{fname}")
                img_debug.save(map_path)
            except: pass

        if not cuts:
            log.append(f"WARN: Brak wycinkow dla {fname}, przetwarzam calosc.")
            cuts = [Image.open(cur)]
        
        # 3. Renowacja
        for i, cut in enumerate(cuts):
            suf = f"_{i+1}" if len(cuts)>1 else ""
            out = os.path.join(OUTPUT_DIR, f"restored_{os.path.splitext(fname)[0]}{suf}.png")
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
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    if os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR)
    
    if not os.path.exists(INPUT_ZIP): 
        print(f"CRITICAL: Nie znaleziono {INPUT_ZIP}!")
        exit(1)
        
    print("[INIT] Rozpakowywanie...")
    with zipfile.ZipFile(INPUT_ZIP,'r') as z: z.extractall(TEMP_DIR)
    
    fs = [os.path.join(r,f) for r,_,x in os.walk(TEMP_DIR) for f in x if f.lower().endswith(('jpg','png','jpeg'))]
    print(f"Start: {len(fs)} files, {MAX_WORKERS} threads.")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        ft = {ex.submit(process, f, os.path.basename(f)):f for f in fs}
        for f in concurrent.futures.as_completed(ft):
            for l in f.result(): print(l)
            
    try: shutil.rmtree(TEMP_DIR)
    except: pass
    print("DONE.")
''')

    print(f"\n[+] Patched V14 (Watershed). \nRun 'python setup_project.py' then 'Tissaia_Project\\START.bat'")

if __name__ == "__main__":
    main()