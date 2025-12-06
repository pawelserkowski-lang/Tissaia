import os, random
from PIL import Image
from src.config import OUTPUT_DIR
from src.ai_core import detect_rotation_strict, restore_final
from src.graphics import find_cuts_robust

def process(fpath, fname):
    log = []
    temps = []
    try:
        # 1. Rotacja skanu
        ang = detect_rotation_strict(fpath)
        cur = fpath
        if ang != 0:
            rot_name = f"tmp_rot_{random.randint(111,999)}_{fname}"
            Image.open(fpath).rotate(-ang, expand=True).save(rot_name)
            cur = rot_name
            temps.append(rot_name)
        
        # 2. Wycinanie V23 (Anti-Strip)
        print(f">> Analyzing: {fname} ...")
        cuts, map_img = find_cuts_robust(cur)
        
        if map_img:
            map_path = os.path.join(OUTPUT_DIR, f"DEBUG_MAP_{fname}")
            map_img.save(map_path)
            log.append(f"MAP: {map_path}")

        if not cuts:
            log.append(f"WARN: No cuts for {fname}. Using full image.")
            cuts = [Image.open(cur)]
        
        # 3. Renowacja
        for i, cut in enumerate(cuts):
            suf = f"_{i+1}" if len(cuts)>1 else ""
            out = os.path.join(OUTPUT_DIR, f"restored_{os.path.splitext(fname)[0]}{suf}.png")
            
            print(f"   >> Restoring {fname} ({i+1}/{len(cuts)})...")
            if restore_final(cut, out): log.append(f"OK: {out}")
            else: log.append(f"ERR: {fname} part {i+1}")
            
    except Exception as e: log.append(f"CRASH {fname}: {e}")
    finally:
        for t in temps: 
            if os.path.exists(t): 
                try: os.remove(t)
                except: pass
    return log