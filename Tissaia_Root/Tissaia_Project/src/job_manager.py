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
