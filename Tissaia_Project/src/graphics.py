import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter

def apply_super_sharpen(img):
    img = ImageEnhance.Contrast(img).enhance(1.1)
    img = ImageEnhance.Color(img).enhance(1.1)
    img = ImageEnhance.Sharpness(img).enhance(1.3)
    return img

def perform_watershed(img, dist_ratio, k, i):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((k, k), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=i)
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist, dist_ratio * dist.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        markers = cv2.watershed(img, markers)
        
        extracted = []
        unique = np.unique(markers)
        for m in unique:
            if m <= 1: continue
            mask = np.zeros(gray.shape, dtype="uint8")
            mask[markers == m] = 255
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts: continue
            c = cnts[0]
            if cv2.contourArea(c) < (img.shape[0]*img.shape[1]*0.015): continue
            x,y,w,h = cv2.boundingRect(c)
            crop = img[y:y+h, x:x+w]
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            extracted.append(Image.fromarray(crop_rgb))
        return extracted
    except: return []

def perform_glue(img):
    try:
        blurred = cv2.GaussianBlur(img, (9, 9), 0)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((9, 9), np.uint8)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=5)
        cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        extracted = []
        for c in cnts:
            if cv2.contourArea(c) < (img.shape[0]*img.shape[1]*0.015): continue
            x,y,w,h = cv2.boundingRect(c)
            crop = img[y:y+h, x:x+w]
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            extracted.append(Image.fromarray(crop_rgb))
        return extracted
    except: return []
