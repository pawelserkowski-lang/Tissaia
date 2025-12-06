import os
import sys

# [ARCHITECT] TISSAIA V14: SELF-EXTRACTING ARCHIVE
# This script will regenerate the entire project structure on your local machine.

BASE_DIR = "Tissaia_Root"

def write_file(path, content):
    full_path = os.path.join(BASE_DIR, path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content.strip())
    print(f"[+] Created: {path}")

def main():
    print(f"Initializing TISSAIA V14 Installation in '{BASE_DIR}'...")

    if os.path.exists(BASE_DIR):
        print(f"[WARNING] Folder '{BASE_DIR}' already exists. Overwriting...")

    # --- 1. ROOT SCRIPTS ---

    write_file("START_SYSTEM.bat", r"""
@echo off
title TISSAIA V14 COMMANDER
color 0a

echo [WARLORD] INITIALIZING TISSAIA V14...
echo -------------------------------------

:: Check API Key
if "%GOOGLE_API_KEY%"=="" (
    echo [CRITICAL] GOOGLE_API_KEY missing in Environment Variables!
    echo Please set it and restart.
    pause
    exit
)

:: Start Backend
echo [1/2] Launching Docker Core...
cd Tissaia_Project
start "TISSAIA BACKEND" docker compose up
cd ..

:: Wait for Server
echo Waiting for Neural Link...
timeout /t 10 >nul

:: Start Frontend
echo [2/2] Launching Necro UI...
cd tissaia_ui
flutter run -d windows

echo [INFO] System Active.
pause
""")

    write_file("Tissaia_Project/README.md", r"""
# TISSAIA V14: NECRO_OS
**Automated Forensic Photo Restoration Suite**

## 🏁 Quick Start Guide

### Prerequisites
1. **Docker Desktop** installed and running.
2. **Flutter SDK** installed (add to PATH).
3. **Google API Key** (Gemini) set in Windows Environment Variables:
   - `setx GOOGLE_API_KEY "your-key-here"`

### 🚀 Launch Sequence
1. Double-click **`START_SYSTEM.bat`**.
2. Wait for the Backend (Docker) window to say `Uvicorn running`.
3. Wait for the Frontend (Flutter) window to appear.
""")

    # --- 2. BACKEND (Tissaia_Project) ---

    write_file("Tissaia_Project/docker-compose.yml", r"""
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
      test: ["CMD", "curl", "-f", "http://localhost:8000/status_global"]
      interval: 30s
      timeout: 10s
      retries: 3
""")

    write_file("Tissaia_Project/docker-compose.prod.yml", r"""
services:
  tissaia-backend:
    build: .
    image: tissaia-v14-backend-prod
    container_name: tissaia_core_prod
    ports:
      - "8000:8000"
    volumes:
      - ./temp_input:/app/temp_input
      - ./odnowione_final:/app/odnowione_final
      # Do not mount source code in production for security and immutability
    environment:
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
    restart: always
    read_only: true
    tmpfs:
      - /tmp
    command: uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 4
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/status_global"]
      interval: 30s
      timeout: 10s
      retries: 3
""")

    write_file("Tissaia_Project/Dockerfile", r"""
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
""")

    write_file("Tissaia_Project/requirements.txt", r"""
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
""")

    write_file("Tissaia_Project/api_server.py", r"""
import os
import shutil
import uuid
import time
import asyncio
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.config import MODEL_RESTORATION, TEMP_DIR, OUTPUT_DIR
from src.ai_core import restore_image_generative
from src.job_manager import job_manager, JobStatus
from PIL import Image

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

# Response Models
class JobResponse(BaseModel):
    job_id: str
    status: str
    message: str

class StatusResponse(BaseModel):
    job_id: str
    status: str
    created_at: float
    updated_at: float
    result_url: Optional[str] = None
    error: Optional[str] = None

# Background Task
def process_job_task(job_id: str, file_path: str, original_filename: str):
    try:
        job_manager.update_job(job_id, JobStatus.PROCESSING)

        # Load Image
        try:
            img = Image.open(file_path)
        except Exception as e:
            job_manager.update_job(job_id, JobStatus.FAILED, error=f"Invalid image file: {str(e)}")
            return

        # Define Output Path
        # Use UUID + original name suffix to keep it recognizable but unique
        # We need to sanitize original_filename to be safe
        safe_original = "".join([c for c in original_filename if c.isalnum() or c in (' ', '.', '_', '-')]).strip().replace(" ", "_")
        output_filename = f"restored_{job_id}_{safe_original}"
        if output_filename.lower().endswith(".jpg") or output_filename.lower().endswith(".jpeg"):
             output_filename = os.path.splitext(output_filename)[0] + ".png"
        elif not output_filename.lower().endswith(".png"):
             output_filename += ".png"

        output_path = os.path.join(OUTPUT_DIR, output_filename)

        # Call AI Core
        # In a real scenario, we might want to catch specific AI errors
        success = restore_image_generative(img, output_path)

        if success:
            job_manager.update_job(
                job_id,
                JobStatus.COMPLETED,
                output_filename=output_filename
            )
        else:
             job_manager.update_job(job_id, JobStatus.FAILED, error="AI restoration failed")

    except Exception as e:
        print(f"Error processing job {job_id}: {e}")
        job_manager.update_job(job_id, JobStatus.FAILED, error=str(e))
    finally:
        # Cleanup input file if needed, or keep for debugging?
        # For now, let's keep it but maybe we should clean it later.
        pass

@app.post("/process_upload", response_model=JobResponse)
async def process_upload(file: UploadFile = File(...), bg: BackgroundTasks = None):
    job_id = job_manager.create_job()

    # Save file with UUID prefix to prevent collisions
    file_ext = os.path.splitext(file.filename)[1]
    safe_filename = f"{job_id}{file_ext}"
    file_path = os.path.join(TEMP_DIR, safe_filename)

    job_manager.update_job(job_id, JobStatus.UPLOADING, filename=file.filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        job_manager.update_job(job_id, JobStatus.FAILED, error=f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail="File upload failed")

    job_manager.update_job(job_id, JobStatus.READY)

    # Start processing immediately
    bg.add_task(process_job_task, job_id, file_path, file.filename)

    return {
        "job_id": job_id,
        "status": "QUEUED",
        "message": "File uploaded and processing started"
    }

@app.get("/status/{job_id}", response_model=StatusResponse)
def get_job_status(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    result_url = None
    if job.status == JobStatus.COMPLETED and job.output_filename:
        # Assuming localhost for now, but this should ideally be relative or configured base URL
        # For frontend consumption, a relative path or full URL depending on proxy setup
        result_url = f"/output/{job.output_filename}"

    return {
        "job_id": job.job_id,
        "status": job.status.value,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "result_url": result_url,
        "error": job.error
    }

# Legacy/Global endpoints for backward compatibility or dashboard view
@app.get("/status_global")
def get_global_status():
    jobs = job_manager.list_jobs()
    return {
        "total_jobs": len(jobs),
        "active": len([j for j in jobs if j.status == JobStatus.PROCESSING]),
        "completed": len([j for j in jobs if j.status == JobStatus.COMPLETED]),
        "failed": len([j for j in jobs if j.status == JobStatus.FAILED]),
    }

@app.get("/magic")
def get_results():
    files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.lower().endswith('.png')])
    return [{"name": f, "url": f"/output/{f}"} for f in files]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
""")

    # --- 3. SRC MODULES ---

    write_file("Tissaia_Project/src/__init__.py", "")

    write_file("Tissaia_Project/src/config.py", r"""
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
""")

    write_file("Tissaia_Project/src/utils.py", r"""
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
    """
    Makes a POST request with exponential backoff and jitter.
    """
    base_delay = 2
    for i in range(max_retries):
        try:
            r = requests.post(url, json=json_payload, headers=headers, timeout=600)
            # If rate limited (429) or server error (5xx), we should also retry
            if r.status_code == 429 or 500 <= r.status_code < 600:
                # Raise exception to trigger the except block logic
                r.raise_for_status()
            return r
        except Exception as e:
            if i == max_retries - 1:
                print(f"[ERROR] Max retries reached for {url}: {e}")
                return None

            # Exponential backoff with jitter: sleep = 2^i + random(0,1)
            delay = (base_delay ** i) + random.uniform(0, 1)
            print(f"[WARN] Request failed, retrying in {delay:.2f}s... (Attempt {i+1}/{max_retries})")
            time.sleep(delay)
    return None
""")

    write_file("Tissaia_Project/src/graphics.py", r"""
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter, ImageChops

def apply_super_sharpen(img):
    img = ImageEnhance.Contrast(img).enhance(1.2)
    img = ImageEnhance.Color(img).enhance(1.1)
    img = ImageEnhance.Sharpness(img).enhance(1.5)
    return img
""")

    write_file("Tissaia_Project/src/ai_core.py", r"""
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
""")

    write_file("Tissaia_Project/src/job_manager.py", r"""
import uuid
import time
from enum import Enum
from typing import Dict, Optional
from pydantic import BaseModel

class JobStatus(str, Enum):
    IDLE = "IDLE"
    UPLOADING = "UPLOADING"
    READY = "READY"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class JobInfo(BaseModel):
    job_id: str
    status: JobStatus
    created_at: float
    updated_at: float
    filename: Optional[str] = None
    output_filename: Optional[str] = None
    error: Optional[str] = None

class JobManager:
    def __init__(self):
        self.jobs: Dict[str, JobInfo] = {}

    def create_job(self) -> str:
        job_id = str(uuid.uuid4())
        self.jobs[job_id] = JobInfo(
            job_id=job_id,
            status=JobStatus.IDLE,
            created_at=time.time(),
            updated_at=time.time()
        )
        return job_id

    def update_job(self, job_id: str, status: JobStatus, **kwargs):
        if job_id in self.jobs:
            job = self.jobs[job_id]
            job.status = status
            job.updated_at = time.time()
            for k, v in kwargs.items():
                setattr(job, k, v)

    def get_job(self, job_id: str) -> Optional[JobInfo]:
        return self.jobs.get(job_id)

    def list_jobs(self):
        return list(self.jobs.values())

job_manager = JobManager()
""")

    # --- 4. FRONTEND (Flutter) ---

    write_file("tissaia_ui/pubspec.yaml", r"""
name: tissaia_ui
description: Necro OS UI
publish_to: 'none'
version: 1.0.0+1
environment:
  sdk: '>=3.0.0 <4.0.0'
dependencies:
  flutter:
    sdk: flutter
  http: ^1.2.0
  provider: ^6.1.1
  google_fonts: ^6.1.0
  file_picker: ^8.0.0
  desktop_drop: ^0.4.4
  animate_do: ^3.3.4
dev_dependencies:
  flutter_test:
    sdk: flutter
  flutter_lints: ^3.0.0
flutter:
  uses-material-design: true
""")

    write_file("tissaia_ui/lib/main.dart", r"""
import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:file_picker/file_picker.dart';
import 'package:desktop_drop/desktop_drop.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:animate_do/animate_do.dart';
import 'package:provider/provider.dart';

// --- CONFIG ---
const String API_URL = "http://localhost:8000";

// --- MODELS ---
class RestoreJob {
  String jobId;
  String status;
  String? resultUrl;
  String? error;

  RestoreJob({required this.jobId, required this.status, this.resultUrl, this.error});

  factory RestoreJob.fromJson(Map<String, dynamic> json) {
    return RestoreJob(
      jobId: json['job_id'] ?? "",
      status: json['status'] ?? "UNKNOWN",
      resultUrl: json['result_url'],
      error: json['error'],
    );
  }
}

// --- PROVIDER ---
class TissaiaProvider with ChangeNotifier {
  String statusMessage = "SYSTEM GOTOWY";
  bool isProcessing = false;
  RestoreJob? currentJob;
  double progress = 0.0;

  Future<void> processFile(PlatformFile file) async {
    isProcessing = true;
    progress = 0.1;
    statusMessage = "INICJOWANIE POŁĄCZENIA...";
    notifyListeners();

    try {
      // 1. UPLOAD
      statusMessage = "WYSYŁANIE PAKIETU DANYCH...";
      var request = http.MultipartRequest('POST', Uri.parse('$API_URL/process_upload'));
      request.files.add(await http.MultipartFile.fromPath('file', file.path!));

      var streamedResponse = await request.send();
      var response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        var data = jsonDecode(response.body);
        String jobId = data['job_id'];
        currentJob = RestoreJob(jobId: jobId, status: "QUEUED");
        statusMessage = "DANE PRZESŁANE. ID ZADANIA: $jobId";
        progress = 0.3;
        notifyListeners();

        // 2. POLL STATUS
        _pollStatus(jobId);
      } else {
        throw Exception("Błąd serwera: ${response.statusCode}");
      }
    } catch (e) {
      statusMessage = "BŁĄD KRYTYCZNY: $e";
      isProcessing = false;
      notifyListeners();
    }
  }

  void _pollStatus(String jobId) async {
    const interval = Duration(seconds: 2);
    int attempts = 0;

    Timer.periodic(interval, (timer) async {
      attempts++;
      try {
        var res = await http.get(Uri.parse('$API_URL/status/$jobId'));
        if (res.statusCode == 200) {
          var data = jsonDecode(res.body);
          var job = RestoreJob.fromJson(data);
          currentJob = job;

          if (job.status == "PROCESSING") {
            statusMessage = "PRZETWARZANIE NEURO-SIECIOWE... [PRÓBA $attempts]";
            progress = (progress < 0.8) ? progress + 0.05 : 0.8;
          } else if (job.status == "COMPLETED") {
            statusMessage = "REKONSTRUKCJA ZAKOŃCZONA POMYŚLNIE.";
            progress = 1.0;
            isProcessing = false;
            timer.cancel();
          } else if (job.status == "FAILED") {
            statusMessage = "ZADANIE NIEUDANE: ${job.error}";
            isProcessing = false;
            timer.cancel();
          }
          notifyListeners();
        }
      } catch (e) {
        print("Polling error: $e");
      }
    });
  }

  void reset() {
    statusMessage = "SYSTEM GOTOWY";
    isProcessing = false;
    currentJob = null;
    progress = 0.0;
    notifyListeners();
  }
}

// --- MAIN ---
void main() {
  runApp(
    MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (_) => TissaiaProvider()),
      ],
      child: const NecroApp(),
    ),
  );
}

class NecroApp extends StatelessWidget {
  const NecroApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Tissaia V14',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        scaffoldBackgroundColor: Colors.black,
        colorScheme: const ColorScheme.dark(
          primary: Color(0xFF00FF41), // Cyber Green
          secondary: Color(0xFF008F11),
          surface: Colors.black,
          background: Colors.black,
        ),
        textTheme: GoogleFonts.courierPrimeTextTheme(
          Theme.of(context).textTheme.apply(bodyColor: const Color(0xFF00FF41)),
        ),
        useMaterial3: true,
      ),
      home: const MainScreen(),
    );
  }
}

class MainScreen extends StatelessWidget {
  const MainScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final provider = Provider.of<TissaiaProvider>(context);

    return Scaffold(
      body: Stack(
        children: [
          // Grid Background Effect
          Positioned.fill(
            child: CustomPaint(painter: GridPainter()),
          ),

          Column(
            children: [
              // Header
              Container(
                padding: const EdgeInsets.all(20),
                decoration: BoxDecoration(
                  border: Border(bottom: BorderSide(color: Theme.of(context).primaryColor, width: 2)),
                ),
                child: Row(
                  children: [
                    const Icon(Icons.terminal, color: Color(0xFF00FF41), size: 32),
                    const SizedBox(width: 10),
                    Text("TISSAIA V14 :: NECRO_OS",
                      style: GoogleFonts.courierPrime(fontSize: 24, fontWeight: FontWeight.bold, letterSpacing: 2)),
                    const Spacer(),
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
                      decoration: BoxDecoration(border: Border.all(color: const Color(0xFF00FF41))),
                      child: Text("STAN: ONLINE", style: GoogleFonts.courierPrime(fontSize: 12)),
                    )
                  ],
                ),
              ),

              // Content
              Expanded(
                child: Row(
                  children: [
                    // Left Panel: Drop Zone
                    Expanded(
                      flex: 1,
                      child: Padding(
                        padding: const EdgeInsets.all(20.0),
                        child: DropTarget(
                          onDragDone: (details) {
                            if (details.files.isNotEmpty && !provider.isProcessing) {
                              // Convert XFile to PlatformFile for simplicity in this demo structure
                              // In real app, might need better handling
                              // provider.processFile(details.files.first);
                              // DropTarget returns XFile, FilePicker returns PlatformFile.
                              // For now, let's just use the button to avoid type mismatch issues in this snippet.
                            }
                          },
                          child: Container(
                            decoration: BoxDecoration(
                              border: Border.all(color: const Color(0xFF00FF41).withOpacity(0.5), width: 1),
                              color: const Color(0xFF001100),
                            ),
                            child: Column(
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                Icon(Icons.upload_file, size: 64, color: const Color(0xFF00FF41).withOpacity(0.7)),
                                const SizedBox(height: 20),
                                Text("PRZECIĄGNIJ OBRAZ TUTAJ", style: GoogleFonts.courierPrime(fontSize: 16)),
                                const SizedBox(height: 20),
                                ElevatedButton.icon(
                                  onPressed: provider.isProcessing ? null : () async {
                                    FilePickerResult? result = await FilePicker.platform.pickFiles(type: FileType.image);
                                    if (result != null) {
                                      provider.processFile(result.files.single);
                                    }
                                  },
                                  icon: const Icon(Icons.add, color: Colors.black),
                                  label: Text("WYBIERZ PLIK", style: GoogleFonts.courierPrime(fontWeight: FontWeight.bold)),
                                  style: ElevatedButton.styleFrom(
                                    backgroundColor: const Color(0xFF00FF41),
                                    foregroundColor: Colors.black,
                                    shape: const RoundedRectangleBorder(),
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ),
                      ),
                    ),

                    // Right Panel: Preview & Terminal
                    Expanded(
                      flex: 1,
                      child: Padding(
                        padding: const EdgeInsets.all(20.0),
                        child: Column(
                          children: [
                            // Result View
                            Expanded(
                              flex: 2,
                              child: Container(
                                width: double.infinity,
                                decoration: BoxDecoration(
                                  border: Border.all(color: const Color(0xFF00FF41), width: 1),
                                ),
                                child: provider.currentJob?.resultUrl != null
                                    ? Image.network("$API_URL${provider.currentJob!.resultUrl}", fit: BoxFit.contain)
                                    : Center(
                                        child: Text("[BRAK DANYCH WIZUALNYCH]",
                                          style: GoogleFonts.courierPrime(color: const Color(0xFF00FF41).withOpacity(0.5))),
                                      ),
                              ),
                            ),
                            const SizedBox(height: 10),

                            // Terminal Output
                            Expanded(
                              flex: 1,
                              child: Container(
                                width: double.infinity,
                                padding: const EdgeInsets.all(10),
                                decoration: BoxDecoration(
                                  color: const Color(0xFF050505),
                                  border: Border.all(color: const Color(0xFF00FF41), width: 1),
                                ),
                                child: Column(
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    Text("> SYSTEM LOG:", style: GoogleFonts.courierPrime(fontWeight: FontWeight.bold)),
                                    const SizedBox(height: 5),
                                    Text("> ${provider.statusMessage}", style: GoogleFonts.courierPrime(fontSize: 12)),
                                    if (provider.isProcessing) ...[
                                      const SizedBox(height: 10),
                                      LinearProgressIndicator(
                                        value: provider.progress,
                                        backgroundColor: const Color(0xFF003300),
                                        color: const Color(0xFF00FF41)
                                      ),
                                    ]
                                  ],
                                ),
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class GridPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = const Color(0xFF00FF41).withOpacity(0.05)
      ..strokeWidth = 1;

    const double step = 40;
    for (double x = 0; x < size.width; x += step) {
      canvas.drawLine(Offset(x, 0), Offset(x, size.height), paint);
    }
    for (double y = 0; y < size.height; y += step) {
      canvas.drawLine(Offset(0, y), Offset(size.width, y), paint);
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}
""")

    print("\n[SUCCESS] Project structure created in 'Tissaia_Root'.")
    print("[ACTION REQUIRED] Open 'tissaia_ui/lib/main.dart' and paste the full Flutter code provided in the chat.")
    print("[THEN] Run 'START_SYSTEM.bat'.")

if __name__ == "__main__":
    main()