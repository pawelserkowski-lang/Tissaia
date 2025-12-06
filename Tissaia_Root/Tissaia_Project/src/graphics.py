import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter, ImageChops

def apply_super_sharpen(img):
    img = ImageEnhance.Contrast(img).enhance(1.2)
    img = ImageEnhance.Color(img).enhance(1.1)
    img = ImageEnhance.Sharpness(img).enhance(1.5)
    return img