import os, zipfile, shutil, concurrent.futures
from src.config import INPUT_ZIP, OUTPUT_DIR, TEMP_DIR, MAX_WORKERS
from src.workflow import process

if __name__ == "__main__":
    print("=== TISSAIA V23: ANTI-STRIP ===")
    
    if os.path.exists(OUTPUT_DIR):
        print("[INIT] Cleaning old artifacts...")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    
    if os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR)
    
    if not os.path.exists(INPUT_ZIP): 
        print(f"CRITICAL: Nie znaleziono {INPUT_ZIP}!")
        exit(1)
        
    print("[INIT] Unzipping...")
    with zipfile.ZipFile(INPUT_ZIP,'r') as z: z.extractall(TEMP_DIR)
    
    fs = [os.path.join(r,f) for r,_,x in os.walk(TEMP_DIR) for f in x if f.lower().endswith(('jpg','png','jpeg'))]
    print(f"[START] Processing {len(fs)} scans with {MAX_WORKERS} threads.")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        ft = {ex.submit(process, f, os.path.basename(f)):f for f in fs}
        for f in concurrent.futures.as_completed(ft):
            for l in f.result(): print(l)
            
    try: shutil.rmtree(TEMP_DIR)
    except: pass
    print("\n[DONE] Mission Complete.")