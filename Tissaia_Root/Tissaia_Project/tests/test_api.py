import uuid
import time
import requests
import pytest
from threading import Thread
import uvicorn
from api_server import app, job_manager, JobStatus

# Mock client for requests in api_server/utils/ai_core if needed
# But here we are testing the API server logic, mainly the job queue.

@pytest.fixture
def client_app():
    from fastapi.testclient import TestClient
    return TestClient(app)

def test_status_endpoint(client_app):
    response = client_app.get("/status_global")
    assert response.status_code == 200
    data = response.json()
    assert "total_jobs" in data

def test_upload_flow(client_app):
    # Create a dummy image file
    files = {'file': ('test.png', b'fakeimagecontent', 'image/png')}

    # We need to mock shutil.copyfileobj or just let it write to temp
    # Since we are using TestClient, it runs in same process, so FS is shared.

    response = client_app.post("/process_upload", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "QUEUED"

    job_id = data["job_id"]

    # Check status immediately
    status_resp = client_app.get(f"/status/{job_id}")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert status_data["job_id"] == job_id
    # It might be PROCESSING or FAILED (because fake content isn't a valid image)

    # Wait a bit for background task to pick up
    time.sleep(1)

    status_resp = client_app.get(f"/status/{job_id}")
    status_data = status_resp.json()

    # Expect FAILED because "fakeimagecontent" is not a valid image for PIL
    assert status_data["status"] == "FAILED"
    assert "Invalid image file" in status_data["error"]

def test_concurrent_jobs(client_app):
    # Create multiple jobs
    job_ids = []
    for i in range(3):
        files = {'file': (f'test_{i}.png', b'fake', 'image/png')}
        resp = client_app.post("/process_upload", files=files)
        job_ids.append(resp.json()["job_id"])

    # Check all exist
    for jid in job_ids:
        r = client_app.get(f"/status/{jid}")
        assert r.status_code == 200
        assert r.json()["job_id"] == jid
