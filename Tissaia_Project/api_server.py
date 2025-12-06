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