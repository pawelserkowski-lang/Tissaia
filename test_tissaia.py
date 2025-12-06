import requests
import time
import os
import sys

# CONFIG
API_URL = "http://localhost:8000"
TEST_IMAGE = "test.jpg"  

def main():
    print("--- TISSAIA CLIENT TESTER ---")
    
    # 0. Check if image exists
    if not os.path.exists(TEST_IMAGE):
        print(f"[ERROR] Please place a file named '{TEST_IMAGE}' in this folder to test.")
        return

    # 1. Upload
    print(f"[1/3] Uploading {TEST_IMAGE}...")
    try:
        with open(TEST_IMAGE, "rb") as f:
            files = {"file": f}
            response = requests.post(f"{API_URL}/process/upload", files=files)
            
        if response.status_code != 200:
            print(f"[FAIL] Upload Failed: {response.text}")
            return
            
        data = response.json()
        job_id = data["job_id"]
        print(f"[OK] Job Started! ID: {job_id}")
        
    except Exception as e:
        print(f"[FAIL] Connection Error: {e}")
        print("Is Docker running?")
        return

    # 2. Poll Status
    print(f"[2/3] Waiting for AI processing...")
    start_time = time.time()
    
    while True:
        try:
            r = requests.get(f"{API_URL}/job/{job_id}")
            if r.status_code != 200:
                print(f"Error checking status: {r.status_code}")
                break
                
            status_data = r.json()
            state = status_data["status"]
            progress = status_data.get("progress", 0)
            
            # Simple progress output
            sys.stdout.write(f"\rStatus: {state} | Progress: {progress}% | Time: {int(time.time() - start_time)}s")
            sys.stdout.flush()
            
            if state == "COMPLETED":
                print("\n[DONE] Processing Complete!")
                print(f"Result files: {status_data['results']}")
                break
            elif state == "FAILED":
                print(f"\n[FAIL] Job Failed: {status_data.get('error')}")
                break
                
            time.sleep(2)
        except KeyboardInterrupt:
            print("\nStopped by user.")
            break

    # 3. Locate Result
    print(f"\n[3/3] Check your output folder: Tissaia_Project/odnowione_final")

if __name__ == "__main__":
    main()