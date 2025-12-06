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
            "job_id": job_id,  # <--- FIXED: Was "id", causing Pydantic 500 error
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