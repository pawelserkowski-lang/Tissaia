import time
import random
import requests
import json
from src.utils import make_request_with_retry

class MockResponse:
    def __init__(self, status_code, json_data):
        self.status_code = status_code
        self._json_data = json_data

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if 400 <= self.status_code < 600:
             raise requests.exceptions.HTTPError(f"{self.status_code} Error")

def test_exponential_backoff(monkeypatch):
    """
    Test that make_request_with_retry retries on failure and uses backoff.
    We will mock requests.post to fail a few times then succeed.
    """
    call_times = []

    def mock_post(*args, **kwargs):
        call_times.append(time.time())
        # First 2 calls fail with 500, 3rd succeeds
        if len(call_times) < 3:
            return MockResponse(500, {})
        return MockResponse(200, {"success": True})

    monkeypatch.setattr(requests, "post", mock_post)

    # We also want to verify the sleep time, but it's hard to measure exactly with sleep.
    # We can mock time.sleep to record the delays.
    sleeps = []
    monkeypatch.setattr(time, "sleep", lambda x: sleeps.append(x))

    start_time = time.time()
    url = "http://fake.url"
    res = make_request_with_retry(url, {}, {})

    assert res is not None
    assert res.status_code == 200
    assert len(call_times) == 3 # Initial + 2 retries
    assert len(sleeps) == 2 # Sleep after 1st and 2nd fail

    # Verify exponential backoff: sleep times should be increasing
    # Attempt 1: 2^0 + jitter (1.0 - 2.0)
    # Attempt 2: 2^1 + jitter (2.0 - 3.0)
    assert sleeps[0] >= 1.0
    assert sleeps[1] >= 2.0
    assert sleeps[1] > sleeps[0] # Roughly increasing
