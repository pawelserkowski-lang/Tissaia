import os, random
from PIL import Image, ImageDraw
from src.config import OUTPUT_DIR
from src.ai_core import detect_rotation_strict, restore_final
from src.graphics import find_cuts_watershed

def process(fpath, fname):
    log = []
    temps = []
    try:
        # 1. Rotacja
        ang = detect_rotation_strict(fpath)
        cur = fpath
        if ang != 0:
            rot_name = f"tmp_rot_{random.randint(111,999)}_{fname}"
            Image.open(fpath).rotate(ang, expand=True).save(rot_name)
            cur = rot_name
            temps.append(rot_name)
        
        # 2. Wycinanie V14
        cuts, map_img = find_cuts_watershed(cur)
        
        if map_img:
            map_path = os.path.join(OUTPUT_DIR, f"MAP_{fname}")
            map_img.save(map_path)
            log.append(f"MAP: {map_path}")
        else:
            try:
                img_debug = Image.open(cur)
                d = ImageDraw.Draw(img_debug)
                d.rectangle([0,0,img_debug.width-1, img_debug.height-1], outline="red", width=10)
                map_path = os.path.join(OUTPUT_DIR, f"MAP_FAIL_{fname}")
                img_debug.save(map_path)
            except: pass

        if not cuts:
            log.append(f"WARN: Brak wycinkow dla {fname}, przetwarzam calosc.")
            cuts = [Image.open(cur)]
        
        # 3. Renowacja
        for i, cut in enumerate(cuts):
            suf = f"_{i+1}" if len(cuts)>1 else ""
            out = os.path.join(OUTPUT_DIR, f"restored_{os.path.splitext(fname)[0]}{suf}.png")
            if restore_final(cut, out): log.append(f"OK: {out}")
            else: log.append(f"ERR: {fname} part {i+1}")
            
    except Exception as e: log.append(f"CRASH {fname}: {e}")
    finally:
        for t in temps: 
            if os.path.exists(t): 
                try: os.remove(t)
                except: pass
    return log