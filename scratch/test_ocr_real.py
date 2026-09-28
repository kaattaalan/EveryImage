import sys
from pathlib import Path
sys.path.insert(0, str(Path("E:/Workspace/EveryImage")))
from src.models import OcrEngine
from PIL import Image
import numpy as np
import time
import cv2
import sqlite3

engine = OcrEngine.get_instance()

conn = sqlite3.connect("E:/Workspace/EveryImage/data/everyimage.db")
cur = conn.cursor()
cur.execute("SELECT file_path FROM images WHERE ocr_text IS NULL OR ocr_text = '' LIMIT 1")
no_txt = cur.fetchone()[0]

cur.execute("SELECT file_path FROM images WHERE ocr_text IS NOT NULL AND ocr_text != '' LIMIT 1")
has_txt = cur.fetchone()[0]

def test(path, angle):
    img = cv2.imread(path)
    w, h = img.shape[1], img.shape[0]
    s = 1280 / max(w, h)
    if s < 1.0:
        img = cv2.resize(img, (int(w*s), int(h*s)))
    
    # warmup
    engine.engine(img, use_angle_cls=angle)
    
    t0 = time.time()
    for _ in range(5):
        engine.engine(img, use_angle_cls=angle)
    t1 = time.time()
    return (t1 - t0) / 5.0

print("No text, angle:", test(no_txt, True))
print("No text, no angle:", test(no_txt, False))
print("Text, angle:", test(has_txt, True))
print("Text, no angle:", test(has_txt, False))
