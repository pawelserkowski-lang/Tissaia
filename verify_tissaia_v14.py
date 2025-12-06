import os
import re
import zipfile
import shutil
import cv2
import numpy as np
import concurrent.futures
import time

# --- CONFIG ---
INPUT_ZIP = "zdjecia.zip"
TEMP_DIR = "temp_input"
CHECKLIST_FILE = "Lista kontrolna.txt"
DEBUG_DIR = "debug_output"

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'

# --- 1. SETUP ENV ---
def setup_environment():
    if os.path.exists(DEBUG_DIR): shutil.rmtree(DEBUG_DIR)
    os.makedirs(DEBUG_DIR)
    if os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR)

    # Szukamy zdjęć w katalogu bieżącym, żeby spakować je do ZIPa (symulacja inputu)
    jpg_files = [f for f in os.listdir('.') if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Jeśli nie ma zipa, a są pliki, to pakujemy
    if not os.path.exists(INPUT_ZIP) and jpg_files:
        print(f"{Colors.OKBLUE}[INIT] Pakowanie {len(jpg_files)} plików do {INPUT_ZIP}...{Colors.ENDC}")
        with zipfile.ZipFile(INPUT_ZIP, 'w') as z:
            for f in jpg_files:
                z.write(f)
    
    if not os.path.exists(INPUT_ZIP):
        print(f"{Colors.FAIL}[CRITICAL] Brak {INPUT_ZIP} i brak zdjęć do spakowania!{Colors.ENDC}")
        return False
    return True

def parse_checklist(path):
    if not os.path.exists(path):
        print(f"{Colors.FAIL}[CRITICAL] Brak pliku {path}!{Colors.ENDC}")
        return {}
    
    truth = {}
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
        # Regex łapie "12.jpg" i liczbę po nim, ignorując polskie końcówki
        matches = re.findall(r"(\d+\.jpg).*?(\d+)", content, re.IGNORECASE | re.DOTALL)
        for filename, count in matches:
            truth[filename] = int(count)
    return truth

# --- CORE ALGORITHMS (SILNIK GRAFICZNY) ---

def run_watershed(img, dist_ratio=0.2, kernel_size=3, morph_iter=2):
    """
    Level 1 & 2: Watershed segmentation.
    """
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=morph_iter)
        
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, dist_ratio * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        markers = cv2.watershed(img, markers)
        
        cuts = 0
        map_img = img.copy()
        img_area = img.shape[0] * img.shape[1]
        unique_markers = np.unique(markers)
        
        for m in unique_markers:
            if m <= 1: continue 
            
            mask = np.zeros(gray.shape, dtype="uint8")
            mask[markers == m] = 255
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: continue
            
            c = contours[0]
            area = cv2.contourArea(c)
            if area < (img_area * 0.015): continue
            
            x, y, w, h = cv2.boundingRect(c)
            ratio = w / float(h)
            if ratio > 20 or ratio < 0.05: continue
            
            cuts += 1
            cv2.rectangle(map_img, (x, y), (x+w, y+h), (0, 255, 0), 5)
            cv2.putText(map_img, str(cuts), (x+50, y+100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
            
        return cuts, map_img
    except Exception:
        return 0, None

def run_simple_contours(img, threshold_method="OTSU", kernel_size=3):
    """
    Level 3: Simple Contours (Fallback).
    """
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if threshold_method == "OTSU":
            ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        elif threshold_method == "ADAPTIVE":
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cuts = 0
        map_img = img.copy()
        img_area = img.shape[0] * img.shape[1]
        
        for c in contours:
            area = cv2.contourArea(c)
            if area < (img_area * 0.015): continue
            
            x, y, w, h = cv2.boundingRect(c)
            ratio = w / float(h)
            if ratio > 20 or ratio < 0.05: continue
            
            cuts += 1
            cv2.rectangle(map_img, (x, y), (x+w, y+h), (255, 0, 0), 5)
            
        return cuts, map_img
    except: return 0, None

def run_glue_protocol(img, kernel_size=9, iterations=5):
    """
    Level 4: THE GLUE PROTOCOL (Heavy Repair).
    Używa agresywnej dylatacji i morfologicznego zamknięcia (CLOSE),
    żeby skleić pęknięte zdjęcia (np. 13.jpg) w jedną całość.
    """
    try:
        # 1. Blur, żeby zgubić detale pęknięć
        blurred = cv2.GaussianBlur(img, (9, 9), 0)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        
        # 2. Otsu
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 3. GLUE: Agresywne zamykanie (CLOSE) - łączy elementy blisko siebie
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        # Dilate łączy, Close wypełnia dziury
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cuts = 0
        map_img = img.copy()
        img_area = img.shape[0] * img.shape[1]
        
        for c in contours:
            area = cv2.contourArea(c)
            if area < (img_area * 0.02): continue # Większy filtr szumów
            
            x, y, w, h = cv2.boundingRect(c)
            ratio = w / float(h)
            if ratio > 20 or ratio < 0.05: continue
            
            cuts += 1
            # Rysuj debug (fioletowy - glue)
            cv2.rectangle(map_img, (x, y), (x+w, y+h), (255, 0, 255), 8)
            
        return cuts, map_img
    except: return 0, None

# --- THE BRUTE FORCE SOLVER ---

def solve_file(filepath, expected_count):
    filename = os.path.basename(filepath)
    img = cv2.imread(filepath)
    if img is None: return {"file": filename, "status": "ERROR", "msg": "Błąd odczytu"}

    # --- LEVEL 1: STANDARD CHECK ---
    cnt, dbg = run_watershed(img, dist_ratio=0.2, kernel_size=3)
    if cnt == expected_count:
        return {"file": filename, "status": "OK", "method": "Standard (dist=0.2)"}

    # Tracking best failures
    best_diff = abs(cnt - expected_count)
    best_map = dbg
    best_method = f"WS Default -> {cnt}"

    # --- LEVEL 2: TOTAL WAR (Brute Force Watershed) ---
    dist_ratios = [0.05, 0.1, 0.15, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]
    kernels = [3, 5, 7] 
    morph_iters = [1, 2, 3]
    
    for k in kernels:
        for i in morph_iters:
            for d in dist_ratios:
                c, m = run_watershed(img, dist_ratio=d, kernel_size=k, morph_iter=i)
                if c == expected_count:
                    return {"file": filename, "status": "FIXED", "method": f"Watershed(dist={d}, k={k}, iter={i})"}
                
                curr_diff = abs(c - expected_count)
                if curr_diff < best_diff:
                    best_diff = curr_diff
                    best_map = m
                    best_method = f"Best: WS(d={d},k={k},i={i})->{c}"

    # --- LEVEL 3: DESPERATION (Simple Contours) ---
    fallback_methods = ["OTSU", "ADAPTIVE"]
    for fm in fallback_methods:
        for k in [3, 5, 7]:
            c, m = run_simple_contours(img, threshold_method=fm, kernel_size=k)
            if c == expected_count:
                 return {"file": filename, "status": "FIXED", "method": f"SimpleContour({fm}, k={k})"}
            
            curr_diff = abs(c - expected_count)
            if curr_diff < best_diff:
                best_diff = curr_diff
                best_map = m
                best_method = f"Best: Simple({fm},k={k})->{c}"

    # --- LEVEL 4: THE GLUE PROTOCOL (Fix for 13.jpg / 10.jpg Over-segmentation) ---
    # Jeśli wykrywamy ZA DUŻO (over-segmentation), próbujemy sklejać.
    if best_diff > 0: # Jeśli nadal nie mamy sukcesu
        glue_kernels = [5, 7, 9, 13, 15] # Coraz większe ziarno
        glue_iters = [3, 5, 8, 12]       # Coraz mocniejsze zamykanie dziur
        
        for gk in glue_kernels:
            for gi in glue_iters:
                c, m = run_glue_protocol(img, kernel_size=gk, iterations=gi)
                if c == expected_count:
                    return {"file": filename, "status": "FIXED", "method": f"GLUE_PROTOCOL(k={gk}, iter={gi})"}
                
                curr_diff = abs(c - expected_count)
                if curr_diff < best_diff:
                    best_diff = curr_diff
                    best_map = m
                    best_method = f"Best: GLUE(k={gk},i={gi})->{c}"

    # --- FAILURE ---
    debug_path = os.path.join(DEBUG_DIR, f"FAIL_{filename}")
    if best_map is not None:
        cv2.imwrite(debug_path, best_map)
        
    return {
        "file": filename, 
        "status": "FAIL", 
        "method": best_method, 
        "detected": expected_count + best_diff if expected_count < cnt else expected_count - best_diff 
    }

def main():
    if not setup_environment(): exit()
    truth_db = parse_checklist(CHECKLIST_FILE)
    if not truth_db: exit()

    with zipfile.ZipFile(INPUT_ZIP, 'r') as z: z.extractall(TEMP_DIR)
    
    target_files = []
    for root, _, files in os.walk(TEMP_DIR):
        for f in files:
            if f in truth_db:
                target_files.append(os.path.join(root, f))
                
    target_files.sort(key=lambda x: int(re.search(r'(\d+)', os.path.basename(x)).group(1)))

    print(f"{Colors.HEADER}=== TISSAIA V14: TOTAL WAR PROTOCOL (Updated) ==={Colors.ENDC}")
    print(f"{Colors.CYAN}[INFO] Added Level 4: GLUE PROTOCOL for broken photos.{Colors.ENDC}")
    print("-" * 60)
    
    start_time = time.time()
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(solve_file, f, truth_db[os.path.basename(f)]): f for f in target_files}
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            
    results.sort(key=lambda x: int(re.search(r'(\d+)', x['file']).group(1)))

    print("-" * 105)
    print(f"{'PLIK':<10} | {'OCZEK.':<6} | {'STATUS':<10} | {'METODA SUKCESU / NAJLEPSZA PRÓBA'}")
    print("-" * 105)

    success_count = 0
    total = len(target_files)
    
    for r in results:
        expected = truth_db[r['file']]
        
        if r['status'] == "OK":
            color = Colors.OKGREEN
            method_str = r['method']
            success_count += 1
        elif r['status'] == "FIXED":
            color = Colors.CYAN
            method_str = f"{Colors.BOLD}{r['method']}{Colors.ENDC}"
            success_count += 1
        else:
            color = Colors.FAIL
            method_str = f"{r['method']} (Wykryto: {r.get('detected', '?')})"

        print(f"{color}{r['file']:<10} | {expected:<6} | {r['status']:<10} | {method_str}{Colors.ENDC}")

    acc = (success_count / total) * 100
    duration = time.time() - start_time
    
    print("-" * 105)
    print(f"{Colors.HEADER}RAPORT MISJI:{Colors.ENDC}")
    print(f"Skuteczność: {Colors.BOLD}{acc:.2f}%{Colors.ENDC} ({success_count}/{total})")
    print(f"Czas trwania: {duration:.2f}s")
    
    if success_count == total:
        print(f"\n{Colors.OKBLUE}[WARLORD] SYSTEM STABILNY. 100% SUKCESU. CHWAŁA ARCHITEKTOWI.{Colors.ENDC}")
    else:
        print(f"\n{Colors.FAIL}[WARLORD] NADAL WALCZYMY. SPRAWDŹ KATALOG {DEBUG_DIR}.{Colors.ENDC}")

if __name__ == "__main__":
    main()