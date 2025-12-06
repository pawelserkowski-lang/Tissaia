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
