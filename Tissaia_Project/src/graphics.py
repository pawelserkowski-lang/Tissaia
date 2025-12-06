import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter, ImageChops

def cv2_to_pil(img_cv):
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

def find_cuts_watershed(pil_image_path):
    """
    V14 Strategy: Marker-Controlled Watershed Segmentation
    The only reliable way to separate touching objects (photos).
    """
    try:
        img = cv2.imread(pil_image_path)
        if img is None: return [], None
        original = img.copy()
        
        # 1. Binarization (Otsu is usually safest for scans)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Invert: Photos should be White, Background Black
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 2. Noise Removal (Opening)
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # 3. Sure Background area (Dilate)
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # 4. Sure Foreground area (Distance Transform)
        # This finds the "centers" of the photos, far away from edges
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        # Threshold: We take only the peaks (centers)
        ret, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        
        # 5. Unknown region (Edges)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # 6. Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)
        # Add 1 to all markers so that sure background is not 0, but 1
        markers = markers + 1
        # Mark the region of unknown with 0
        markers[unknown == 255] = 0
        
        # 7. Watershed
        markers = cv2.watershed(img, markers)
        
        # Extract cuts based on markers
        cuts = []
        map_img = original.copy()
        img_area = img.shape[0] * img.shape[1]
        
        # Unique markers (skip -1 for boundaries and 1 for background)
        unique_markers = np.unique(markers)
        
        i = 1
        for m in unique_markers:
            if m <= 1: continue # Skip background and edges
            
            # Create mask for this marker
            mask = np.zeros(gray.shape, dtype="uint8")
            mask[markers == m] = 255
            
            # Find contour of this segment
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: continue
            
            c = contours[0]
            area = cv2.contourArea(c)
            
            # Filter: Min 2% area
            if area < (img_area * 0.02): continue
            
            x, y, w, h = cv2.boundingRect(c)
            
            # Filter Strips
            ratio = w / float(h)
            if ratio > 15 or ratio < 0.05: continue
            
            # Padding
            pad = 10
            x = max(0, x - pad)
            y = max(0, y - pad)
            w = min(img.shape[1] - x, w + 2*pad)
            h = min(img.shape[0] - y, h + 2*pad)
            
            cut = img[y:y+h, x:x+w]
            cuts.append(cv2_to_pil(cut))
            
            # Debug Map: Draw bounding rect and Center
            cv2.rectangle(map_img, (x, y), (x+w, y+h), (0, 255, 0), 4)
            cv2.putText(map_img, f"#{i}", (x+int(w/2)-20, y+int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
            i += 1
            
        return cuts, cv2_to_pil(map_img)

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
    img = ImageEnhance.Contrast(img).enhance(1.2)
    img = ImageEnhance.Color(img).enhance(1.1)
    img = ImageEnhance.Sharpness(img).enhance(1.5)
    return img