import os, random
from PIL import Image
from src.config import OUTPUT_DIR
from src.ai_core import detect_rotation_strict, detect_corners, restore_final
from src.graphics import warp_perspective

def process(fpath, fname):
    log = []
    temps = []
    try:
        ang = detect_rotation_strict(fpath)
        cur = fpath
        if ang != 0:
            cur = f"tmp_rot_{random.randint(111,999)}_{fname}"
            Image.open(fpath).rotate(ang, expand=True).save(cur)
            temps.append(cur)
        
        data = detect_corners(cur)
        parts = []
        if data and "photos" in data:
            img = Image.open(cur).convert("RGB")
            w, h = img.size
            for i, item in enumerate(data["photos"]):
                try:
                    px = [[(p[0]/1000)*w, (p[1]/1000)*h] for p in item["corners"]]
                    s = warp_perspective(img, px)
                    if s:
                        n = f"tmp_cut_{i}_{fname}"
                        s.save(n, quality=95)
                        parts.append(n); temps.append(n)
                except: pass
        if not parts: parts.append(cur)
        
        for i, p in enumerate(parts):
            suf = f"_{i+1}" if len(parts)>1 else ""
            out = os.path.join(OUTPUT_DIR, f"restored_{os.path.splitext(fname)[0]}{suf}.png")
            if os.path.exists(out): log.append(f"SKIP: {out}"); continue
            
            print(f">> Processing {fname} part {i+1}")
            if restore_final(p, out): log.append(f"OK: {out}")
            else: log.append(f"ERR: {fname}")
            
    except Exception as e: log.append(f"CRASH {fname}: {e}")
    finally:
        for t in temps: 
            if os.path.exists(t): 
                try: os.remove(t)
                except: pass
    return log