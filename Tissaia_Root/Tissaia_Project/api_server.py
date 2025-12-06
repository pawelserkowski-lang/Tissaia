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
        output_filename = f"restored_{os.path.basename(file_path)}"
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
