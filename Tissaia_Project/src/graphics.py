import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageChops

def aggressive_trim_borders(img):
    try:
        bg = Image.new(img.mode, img.size, img.getpixel((0,0)))
        diff = ImageChops.difference(img, bg)
        diff = ImageChops.add(diff, diff, 2.0, -100)
        bbox = diff.getbbox()
        if bbox and (bbox[2]-bbox[0]) > 50 and (bbox[3]-bbox[1]) > 50: return img.crop(bbox)
    except: pass
    return img

def apply_super_sharpen(img):
    img = ImageEnhance.Sharpness(img).enhance(1.5)
    return img.filter(ImageFilter.UnsharpMask(radius=3, percent=150, threshold=3))

def warp_perspective(img, corners):
    def find_coeffs(pa, pb):
        matrix = []
        for p1, p2 in zip(pa, pb):
            matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p1[0]*p2[0], -p1[0]*p2[1]])
            matrix.append([0, 0, 0, p1[0], p1[1], 1, -p1[1]*p2[0], -p1[1]*p2[1]])
        A = np.matrix(matrix, dtype=float)
        B = np.array(pb).reshape(8)
        try: return np.array(np.dot(np.linalg.inv(A.T * A) * A.T, B)).reshape(8)
        except: return None

    pts = np.array(corners, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    tl, br = pts[np.argmin(s)], pts[np.argmax(s)]
    tr, bl = pts[np.argmin(diff)], pts[np.argmax(diff)]

    wA = np.sqrt(((br[0]-bl[0])**2) + ((br[1]-bl[1])**2))
    wB = np.sqrt(((tr[0]-tl[0])**2) + ((tr[1]-tl[1])**2))
    mw = max(int(wA), int(wB))
    hA = np.sqrt(((tr[0]-br[0])**2) + ((tr[1]-br[1])**2))
    hB = np.sqrt(((tl[0]-bl[0])**2) + ((tl[1]-bl[1])**2))
    mh = max(int(hA), int(hB))

    coeffs = find_coeffs([(0,0), (mw,0), (mw,mh), (0,mh)], [(tl[0],tl[1]), (tr[0],tr[1]), (br[0],br[1]), (bl[0],bl[1])])
    if coeffs is not None: return img.transform((mw, mh), Image.PERSPECTIVE, coeffs, Image.BICUBIC)
    return img