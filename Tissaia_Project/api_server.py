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
