import os

def write_file(path, content):
    """Zapisuje plik, tworząc niezbędne katalogi."""
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content.strip())
    print(f"[+] Utworzono: {path}")

def main():
    base_dir = "Tissaia_Project"
    
    print(f"--- TISSAIA GENESIS PROTOCOL (FINAL FIX) ---")
    print(f"Tworzenie struktury w: {os.path.abspath(base_dir)}")

    # 1. STRUKTURA KATALOGÓW
    directories = [
        f"{base_dir}/src",
        f"{base_dir}/temp_input",
        f"{base_dir}/odnowione_final",
    ]
    for d in directories:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"[+] Katalog: {d}")

    # 2. DOCKER CONFIG
    
    dockerfile_content = r"""
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
"""
    write_file(f"{base_dir}/Dockerfile", dockerfile_content)

    docker_compose_content = r"""
services:
  tissaia-backend:
    build: .
    image: tissaia-v14-backend
    container_name: tissaia_core
    ports:
      - "8000:8000"
    volumes:
      - ./temp_input:/app/temp_input
      - ./odnowione_final:/app/odnowione_final
      - ./src:/app/src
      - ./api_server.py:/app/api_server.py
    environment:
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
    restart: unless-stopped
    command: uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
"""
    write_file(f"{base_dir}/docker-compose.yml", docker_compose_content)

    requirements_content = """
fastapi>=0.100.0
uvicorn>=0.23.0
python-multipart>=0.0.6
requests>=2.31.0
python-dotenv>=1.0.0
Pillow>=10.0.0
numpy>=1.24.0
opencv-python-headless>=4.8.0
tqdm>=4.66.0
colorama>=0.4.6
"""
    write_file(f"{base_dir}/requirements.txt", requirements_content)
    write_file(f"{base_dir}/.env", "GOOGLE_API_KEY=YOUR_KEY_HERE_IF_NOT_IN_ENV_VARS")

    # 3. SOURCE CODE (SRC)

    write_file(f"{base_dir}/src/__init__.py", "")

    config_content = r"""
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
MODEL_DETECTION = "models/gemini-3-pro-preview"
MODEL_RESTORATION = "models/gemini-3-pro-image-preview"
MAX_WORKERS = 4
INPUT_ZIP = "zdjecia.zip"
OUTPUT_DIR = "odnowione_final"
TEMP_DIR = "temp_input"

print(f"\n[CONFIG] API Key Present: {'YES' if API_KEY else 'NO'}")
print(f"[CONFIG] Engine: {MODEL_RESTORATION}")
"""
    write_file(f"{base_dir}/src/config.py", config_content)

    utils_content = r"""
import time
import random
import io
import base64
import requests
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
            if r.status_code in [404, 403, 400]: return r
            if r.status_code in [429, 500, 503]:
                time.sleep((i + 1) * 2)
                continue
            return r
        except: time.sleep(2)
    return None
"""
    write_file(f"{base_dir}/src/utils.py", utils_content)

    # Używamy potrójnych cudzysłowów ostrożnie.
    graphics_content = r"""
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter

def apply_super_sharpen(img):
    img = ImageEnhance.Contrast(img).enhance(1.1)
    img = ImageEnhance.Color(img).enhance(1.1)
    img = ImageEnhance.Sharpness(img).enhance(1.3)
    return img

def perform_watershed(img, dist_ratio, k, i):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((k, k), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=i)
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist, dist_ratio * dist.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        markers = cv2.watershed(img, markers)
        
        extracted = []
        unique = np.unique(markers)
        for m in unique:
            if m <= 1: continue
            mask = np.zeros(gray.shape, dtype="uint8")
            mask[markers == m] = 255
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts: continue
            c = cnts[0]
            if cv2.contourArea(c) < (img.shape[0]*img.shape[1]*0.015): continue
            x,y,w,h = cv2.boundingRect(c)
            crop = img[y:y+h, x:x+w]
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            extracted.append(Image.fromarray(crop_rgb))
        return extracted
    except: return []

def perform_glue(img):
    try:
        blurred = cv2.GaussianBlur(img, (9, 9), 0)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((9, 9), np.uint8)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=5)
        cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        extracted = []
        for c in cnts:
            if cv2.contourArea(c) < (img.shape[0]*img.shape[1]*0.015): continue
            x,y,w,h = cv2.boundingRect(c)
            crop = img[y:y+h, x:x+w]
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            extracted.append(Image.fromarray(crop_rgb))
        return extracted
    except: return []
"""
    write_file(f"{base_dir}/src/graphics.py", graphics_content)

    # Poprawione wcięcia w prompt (Python w Pythonie to zawsze zabawa)
    ai_core_content = r"""
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
"""
    write_file(f"{base_dir}/src/ai_core.py", ai_core_content)

    api_server_content = r"""
import os
import shutil
import cv2
import uvicorn
from typing import List
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

# Importy lokalne
from src.config import MODEL_RESTORATION, TEMP_DIR, OUTPUT_DIR
from src.ai_core import restore_image_generative
from src.graphics import perform_watershed, perform_glue

app = FastAPI(title="Tissaia V14 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app.mount("/temp", StaticFiles(directory=TEMP_DIR), name="temp")
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")

STATUS = {
    "state": "IDLE",
    "queue": 0,
    "current_file": ""
}

class StatusResponse(BaseModel):
    status: str
    model: str
    queue: int
    done: int
    current: str

@app.get("/status", response_model=StatusResponse)
def get_status():
    done_count = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')])
    return {
        "status": STATUS["state"],
        "model": MODEL_RESTORATION,
        "queue": STATUS["queue"],
        "done": done_count,
        "current": STATUS["current_file"]
    }

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    STATUS["state"] = "UPLOADING"
    for file in files:
        path = os.path.join(TEMP_DIR, file.filename)
        with open(path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    STATUS["queue"] += len(files)
    STATUS["state"] = "READY"
    return {"message": f"Uploaded {len(files)} files"}

@app.get("/cuts")
def get_cuts():
    files = sorted([f for f in os.listdir(TEMP_DIR) if f.lower().endswith(('.jpg', '.png'))])
    return [{"name": f, "url": f"http://localhost:8000/temp/{f}"} for f in files]

@app.get("/magic")
def get_results():
    files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.lower().endswith('.png')])
    return [{"name": f, "url": f"http://localhost:8000/output/{f}"} for f in files]

def process_image_pipeline(filename: str):
    STATUS["state"] = "PROCESSING"
    STATUS["current_file"] = filename
    
    input_path = os.path.join(TEMP_DIR, filename)
    img = cv2.imread(input_path)
    
    if img is None:
        STATUS["state"] = "ERROR_READ"
        return

    crops = perform_watershed(img, 0.2, 3, 2)
    if not crops: crops = perform_watershed(img, 0.4, 3, 1)
    if not crops: crops = perform_glue(img)
    
    if not crops:
        print(f"Failed to cut {filename}")
        STATUS["state"] = "CUT_FAIL"
        return

    for idx, pil_crop in enumerate(crops):
        base_name = os.path.splitext(filename)[0]
        save_name = f"{base_name}_restored_{idx+1:02d}.png"
        save_path = os.path.join(OUTPUT_DIR, save_name)
        if not os.path.exists(save_path):
            restore_image_generative(pil_crop, save_path)
    
    STATUS["state"] = "IDLE"
    STATUS["current_file"] = ""
    STATUS["queue"] = max(0, STATUS["queue"] - 1)

@app.post("/process/{filename}")
async def start_process(filename: str, bg_tasks: BackgroundTasks):
    bg_tasks.add_task(process_image_pipeline, filename)
    return {"status": "Started", "target": filename}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""
    write_file(f"{base_dir}/api_server.py", api_server_content)

    write_file("run_backend.bat", r"""
@echo off
cd Tissaia_Project
docker compose up --build
""")

    print(f"\n[+] SKRYPT GENESIS ZAKOŃCZONY (FINAL FIX).")
    print(f"[+] Uruchom 'run_backend.bat' aby postawić serwer.")

if __name__ == "__main__":
    main()