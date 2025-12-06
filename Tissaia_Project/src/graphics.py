import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageChops

def cv2_to_pil(img_cv):
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

def crop_min_area_rect(img, rect):
    box = cv2.boxPoints(rect)
    box = np.int32(box) 
    
    width = int(rect[1][0])
    height = int(rect[1][1])

    src_pts = box.astype("float32")
    s = src_pts.sum(axis=1)
    tl = src_pts[np.argmin(s)]
    br = src_pts[np.argmax(s)]
    diff = np.diff(src_pts, axis=1)
    tr = src_pts[np.argmin(diff)]
    bl = src_pts[np.argmax(diff)]
    
    ordered_src = np.array([tl, tr, br, bl], dtype="float32")

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst_pts = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(ordered_src, dst_pts)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    return warped

def is_valid_cut(w, h, total_area_img):
    area = w * h
    ratio = w / float(h) if h > 0 else 0
    
    # V23: Aggressive Strip Killer
    # Minimum 7% of total scan area to be considered a photo.
    # For a scan with 8 photos, each is ~12.5%, so 7% is safe.
    if area < (total_area_img * 0.07): return False 
    
    if ratio < 0.2 or ratio > 5.0: return False
    
    if min(w, h) < 50: return False
    
    return True

def remove_inner_rectangles(candidates):
    if not candidates: return []
    candidates = sorted(candidates, key=lambda x: x[1][2]*x[1][3], reverse=True)
    kept = []
    
    for i, current in enumerate(candidates):
        rect_data, (x1, y1, w1, h1) = current
        is_inner = False
        area1 = w1 * h1
        
        for _, (x2, y2, w2, h2) in kept:
            ix1 = max(x1, x2); iy1 = max(y1, y2)
            ix2 = min(x1+w1, x2+w2); iy2 = min(y1+h1, y2+h2)
            if ix1 < ix2 and iy1 < iy2:
                inter_area = (ix2 - ix1) * (iy2 - iy1)
                if inter_area / float(area1) > 0.85: 
                    is_inner = True
                    break
        if not is_inner:
            kept.append(current)
    return kept

def strategy_flood_fill_precise(img):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 21, 4)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open, iterations=2)
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def strategy_flood_fill_heavy(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 25, 6)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def find_cuts_robust(pil_image_path):
    try:
        img_orig = cv2.imread(pil_image_path)
        if img_orig is None: return [], None
        
        scale = 1.0
        h_orig, w_orig = img_orig.shape[:2]
        if max(h_orig, w_orig) > 2000:
            scale = 0.5
            img = cv2.resize(img_orig, (0,0), fx=scale, fy=scale)
        else:
            img = img_orig.copy()
            
        total_area = img.shape[0] * img.shape[1]
        
        strategies = [
            ("FLOOD_PRECISE", lambda i: strategy_flood_fill_precise(i)),
            ("FLOOD_HEAVY", lambda i: strategy_flood_fill_heavy(i))
        ]
        
        best_candidates = []
        debug_img = img.copy()
        
        for name, method in strategies:
            print(f"   [STRATEGY] Testing {name}...")
            contours = method(img)
            candidates = []
            
            for c in contours:
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)
                box = np.int32(box)
                x, y, w, h = cv2.boundingRect(box) 
                
                if is_valid_cut(w, h, total_area):
                    candidates.append((rect, (x, y, w, h)))
            
            clean = remove_inner_rectangles(candidates)
            
            if len(clean) >= 1:
                best_candidates = clean
                print(f"   [ACCEPTED] {name} found {len(best_candidates)} photos.")
                for (rect, _) in best_candidates:
                    box = cv2.boxPoints(rect)
                    box = np.int32(box)
                    cv2.drawContours(debug_img, [box], 0, (0, 255, 0), 4)
                    cx, cy = int(rect[0][0]), int(rect[0][1])
                    cv2.putText(debug_img, name, (cx-20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                break 
        
        final_cuts = []
        
        for (rect, _) in best_candidates:
            (cx, cy), (w, h), ang = rect
            cx /= scale; cy /= scale
            w /= scale; h /= scale
            scaled_rect = ((cx, cy), (w, h), ang)
            cut = crop_min_area_rect(img_orig, scaled_rect)
            final_cuts.append(cv2_to_pil(cut))
            
        return final_cuts, cv2_to_pil(debug_img)

    except Exception as e:
        print(f"[CV ERROR] {e}")
        return [], None

def aggressive_trim_borders(img):
    try:
        bg = Image.new(img.mode, img.size, img.getpixel((0,0)))
        diff = ImageChops.difference(img, bg)
        diff = ImageChops.add(diff, diff, 2.0, -100)
        bbox = diff.getbbox()
        if bbox: return img.crop(bbox)
    except: pass
    return img

def apply_super_sharpen(img):
    img = ImageEnhance.Contrast(img).enhance(1.1)
    img = ImageEnhance.Sharpness(img).enhance(1.3)
    return img