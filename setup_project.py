import os
import sys
from pathlib import Path

# ==========================================
# TISSAIA PROJECT - ONE-CLICK INSTALLER
# Fixed by: The Jester (because Architect crashed)
# ==========================================

def log(msg):
    print(f"[INSTALLER] {msg}")

PROJECT_FILES = {
    # ---------------------------------------------------------
    # CONFIGURATION & ENVIRONMENT
    # ---------------------------------------------------------
    "Tissaia_Project/.env": r"""
GOOGLE_API_KEY=YOUR_KEY_HERE_IF_NOT_IN_ENV_VARS
""",

    "Tissaia_Project/requirements.txt": r"""
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
pytest>=7.4.0
""",

    "Tissaia_Project/Dockerfile": r"""
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies for OpenCV (gl1, glib2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create necessary directories to ensure permissions
RUN mkdir -p temp_input odnowione_final

EXPOSE 8000

# Using reload for dev
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
""",

    "Tissaia_Project/docker-compose.yml": r"""
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
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/system/status"]
      interval: 30s
      timeout: 10s
      retries: 3
""",

    "Tissaia_Project/README.md": r"""
# Tissaia Project - Gemini Restoration API

A backend service for high-fidelity photo restoration using Google Gemini 3 Vision.

## Architecture

This project uses FastAPI as the web server and interacts with Google's Generative AI models.
It includes a job management system to handle image processing asynchronously.

### Key Components

* **API Server (`api_server.py`)**: FastAPI application exposing endpoints for upload and processing.
* **Pipeline (`src/pipeline.py`)**: Manages jobs (`JobManager`) and executes the restoration workflow.
* **AI Core (`src/ai_core.py`)**: Handles communication with Gemini API.
* **Graphics (`src/graphics.py`)**: OpenCV/PIL based image pre/post-processing.

## Setup & Running

### Prerequisites

* Docker & Docker Compose
* Google Cloud API Key (with Gemini Vision access)

### Deployment

1.  **Configure Environment**:
    Ensure `GOOGLE_API_KEY` is set in your system environment variables OR paste it into `.env`.

2.  **Run with Docker**:
    ```bash
    cd Tissaia_Project
    docker compose up --build -d
    ```

3.  **Access API**:
    * Swagger UI: `http://localhost:8000/docs`
    * System Status: `http://localhost:8000/system/status`
""",

    "Tissaia_Project/run_backend.bat": r"""
@echo off
cd Tissaia_Project
docker compose up --build
""",

    # ---------------------------------------------------------
    # SOURCE CODE: ROOT
    # NOTE: Switched to r''' (single quotes) to allow """ inside the file content
    # ---------------------------------------------------------
    "Tissaia_Project/api_server.py": r'''
import os
import shutil
import uvicorn
import uuid
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Local imports
from src.config import MODEL_RESTORATION, TEMP_DIR, OUTPUT_DIR
from src.pipeline import job_manager, JobStatus

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

# --- Pydantic Models ---

class JobResponse(BaseModel):
    job_id: str
    status: str
    filename: str
    message: str

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: int
    results: List[dict]
    error: Optional[str] = None
    created_at: float

class SystemStatusResponse(BaseModel):
    active_jobs: int
    model: str
    output_files_count: int

# --- Endpoints ---

@app.get("/system/status", response_model=SystemStatusResponse)
def get_system_status():
    """Returns general system health and stats."""
    active_count = sum(1 for j in job_manager.jobs.values() if j["status"] in [JobStatus.QUEUED, JobStatus.PROCESSING])
    done_count = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')])
    return {
        "active_jobs": active_count,
        "model": MODEL_RESTORATION,
        "output_files_count": done_count
    }

@app.post("/upload", tags=["Legacy"])
async def upload_files_legacy(files: List[UploadFile] = File(...)):
    """Legacy upload endpoint. Recommend using /process/upload for single atomic operation."""
    for file in files:
        path = os.path.join(TEMP_DIR, file.filename)
        with open(path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    return {"message": f"Uploaded {len(files)} files. Please call /process/{files[0].filename} to start."}

@app.post("/process/upload", response_model=JobResponse)
async def process_and_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Atomic endpoint: Uploads file and starts processing job."""
    try:
        # Create a unique filename to prevent overwrite race conditions
        file_ext = os.path.splitext(file.filename)[1]
        unique_id = str(uuid.uuid4())
        safe_filename = f"{unique_id}_{file.filename}"
        path = os.path.join(TEMP_DIR, safe_filename)

        with open(path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # We pass the safe_filename to create_job so the worker knows what to read
        job_id = job_manager.create_job(safe_filename)
        background_tasks.add_task(job_manager.process_job, job_id)

        return {
            "job_id": job_id,
            "status": JobStatus.QUEUED,
            "filename": safe_filename,
            "message": "File uploaded and processing started."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process/{filename}", response_model=JobResponse)
async def start_process_existing(filename: str, bg_tasks: BackgroundTasks):
    """Starts processing for an already uploaded file."""
    if not os.path.exists(os.path.join(TEMP_DIR, filename)):
        raise HTTPException(status_code=404, detail="File not found in temp storage.")

    job_id = job_manager.create_job(filename)
    bg_tasks.add_task(job_manager.process_job, job_id)
    
    return {
        "job_id": job_id,
        "status": JobStatus.QUEUED,
        "filename": filename,
        "message": "Processing started."
    }

@app.get("/job/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str):
    """Get status of a specific job."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@app.get("/magic")
def get_results_legacy():
    """Legacy endpoint to list all results."""
    files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.lower().endswith('.png')])
    return [{"name": f, "url": f"/output/{f}"} for f in files]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
''',

    # ---------------------------------------------------------
    # SOURCE CODE: MODULES (src)
    # ---------------------------------------------------------
    "Tissaia_Project/src/__init__.py": "",

    "Tissaia_Project/src/config.py": r"""
import os
from dotenv import load_dotenv

load_dotenv()

# Fallback: manually read .env if load_dotenv fails or ENV var not present
if not os.environ.get("GOOGLE_API_KEY"):
    try:
        if os.path.exists(".env"):
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
""",

    "Tissaia_Project/src/utils.py": r"""
import time
import random
import io
import base64
import requests
import logging
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            if r.status_code in [200, 404, 403, 400]:
                return r

            # Retryable errors: 429 (Rate Limit), 500+, 502, 503, 504
            if r.status_code == 429 or r.status_code >= 500:
                # Exponential Backoff with Jitter
                # Base 2 seconds, multiplier 2^i
                sleep_time = (2 ** i) + random.uniform(0, 1)
                logger.warning(f"Request failed with {r.status_code}. Retrying in {sleep_time:.2f}s (Attempt {i+1}/{max_retries})")
                time.sleep(sleep_time)
                continue

            return r
        except Exception as e:
            sleep_time = (2 ** i) + random.uniform(0, 1)
            logger.warning(f"Request exception: {str(e)}. Retrying in {sleep_time:.2f}s")
            time.sleep(sleep_time)

    logger.error(f"Failed to get valid response from {url} after {max_retries} retries.")
    return None
""",

    "Tissaia_Project/src/graphics.py": r"""
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
""",

    # Switched to r''' here as well to protect the triple quotes inside the prompt variable
    "Tissaia_Project/src/ai_core.py": r'''
import json
import re
import io
import base64
import os
import time
import logging
from PIL import Image
from src.config import API_KEY, MODEL_RESTORATION
from src.utils import make_request_with_retry
from src.graphics import apply_super_sharpen

logger = logging.getLogger(__name__)

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
    prompt = """
    ROLE: Forensic Photo Restoration Specialist using Gemini 3 Vision.
    INPUT: A vintage photo crop (center 90%).
    
    TASK: Reconstruct high-fidelity image.
    
    1. DIGITAL HYGIENE: Remove all dust, scratches, tears, and scan artifacts.
    2. FORENSIC FACE RECONSTRUCTION: Sharp, symmetric eyes. Realistic skin texture.
    3. HDR STUDIO REMASTERING: "Kodak Portra 400" aesthetic. Soft studio lighting.
    
    OUTPUT: Return ONLY the raw restored image.
    """

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
                        try:
                            img_data = base64.b64decode(part["inlineData"]["data"])
                            restored_img = Image.open(io.BytesIO(img_data))

                            final_img = apply_super_sharpen(restored_img)
                            final_img.save(save_path)
                            return True
                        except Exception as img_err:
                            logger.error(f"Failed to decode or save image: {str(img_err)}")
                            return False

            logger.warning(f"Response did not contain valid inlineData. Candidates: {len(data.get('candidates', []))}")

        elif response:
            logger.error(f"API ERROR {response.status_code}: {response.text[:200]}")
            
    except Exception as e:
        logger.error(f"CRITICAL EXCEPTION in restore_image_generative: {str(e)}")

    return False
''',

    "Tissaia_Project/src/pipeline.py": r"""
import os
import cv2
import uuid
import time
import logging
import traceback
from enum import Enum
from typing import Dict, Any, Optional

from src.config import TEMP_DIR, OUTPUT_DIR
from src.ai_core import restore_image_generative
from src.graphics import perform_watershed, perform_glue

logger = logging.getLogger(__name__)

class JobStatus(str, Enum):
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"

class JobManager:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def create_job(self, filename: str) -> str:
        job_id = str(uuid.uuid4())
        self.jobs[job_id] = {
            "id": job_id,
            "filename": filename,
            "status": JobStatus.QUEUED,
            "results": [],
            "error": None,
            "progress": 0,
            "created_at": time.time()
        }
        return job_id

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self.jobs.get(job_id)

    def update_job(self, job_id: str, **kwargs):
        if job_id in self.jobs:
            self.jobs[job_id].update(kwargs)

    def process_job(self, job_id: str):
        job = self.jobs.get(job_id)
        if not job:
            logger.error(f"Job {job_id} not found during processing start.")
            return

        filename = job["filename"]
        self.update_job(job_id, status=JobStatus.PROCESSING, progress=10)

        input_path = os.path.join(TEMP_DIR, filename)

        try:
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"File {filename} not found in {TEMP_DIR}")

            # 1. Read Image
            img = cv2.imread(input_path)
            if img is None:
                raise ValueError("Could not read image (cv2 returned None)")

            self.update_job(job_id, progress=30)

            # 2. Cut / Segment
            # Try watershed logic first
            crops = perform_watershed(img, 0.2, 3, 2)
            if not crops:
                crops = perform_watershed(img, 0.4, 3, 1)
            if not crops:
                crops = perform_glue(img)

            if not crops:
                logger.warning(f"No cuts found for {filename}")
                self.update_job(job_id, status=JobStatus.FAILED, error="No valid segments found to process.")
                return

            self.update_job(job_id, progress=50)

            # 3. Restore each crop
            results = []
            total_crops = len(crops)
            success_count = 0

            for idx, pil_crop in enumerate(crops):
                # Use job_id in output filename to ensure uniqueness
                save_name = f"{job_id}_{idx+1:02d}_restored.png"
                save_path = os.path.join(OUTPUT_DIR, save_name)

                if restore_image_generative(pil_crop, save_path):
                    results.append({"file": save_name, "status": "success"})
                    success_count += 1
                else:
                    results.append({"file": save_name, "status": "failed"})

                # Update progress
                current_progress = 50 + int((idx + 1) / total_crops * 45)
                self.update_job(job_id, progress=current_progress)

            # 4. Finalize
            if success_count == total_crops:
                final_status = JobStatus.COMPLETED
            elif success_count > 0:
                final_status = JobStatus.PARTIAL
            else:
                final_status = JobStatus.FAILED

            self.update_job(job_id, status=final_status, results=results, progress=100)

        except Exception as e:
            logger.error(f"Error processing job {job_id}: {str(e)}")
            traceback.print_exc()
            self.update_job(job_id, status=JobStatus.FAILED, error=str(e))

# Global instance for now
job_manager = JobManager()
""",

    # ---------------------------------------------------------
    # UNIT TESTS
    # ---------------------------------------------------------
    "Tissaia_Project/tests/__init__.py": "",
    "Tissaia_Project/tests/test_basic.py": r"""
def test_import_sanity():
    try:
        from src.pipeline import JobManager
        from src.config import MODEL_RESTORATION
        print("Imports successful")
    except ImportError as e:
        assert False, f"Import failed: {e}"

def test_job_manager_creation():
    from src.pipeline import JobManager
    jm = JobManager()
    job_id = jm.create_job("test_file.jpg")
    job = jm.get_job(job_id)
    assert job["filename"] == "test_file.jpg"
    assert job["status"] == "QUEUED"
""",
}

def main():
    log("Initializing deployment sequence...")
    base_path = Path.cwd()

    for file_path_str, content in PROJECT_FILES.items():
        # Handle correct path separation for OS
        file_path = base_path / Path(file_path_str)
        
        # Create directories if they don't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write file
        log(f"Writing {file_path_str}...")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content.strip() + "\n")

    log("=========================================")
    log("DEPLOYMENT COMPLETE")
    log("=========================================")
    log("Instructions:")
    log("1. cd Tissaia_Project")
    log("2. docker compose up --build -d")
    log("3. Go grab a coffee. The Architect is watching.")

if __name__ == "__main__":
    main()