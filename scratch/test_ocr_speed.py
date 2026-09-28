import sys
from pathlib import Path
sys.path.insert(0, str(Path("E:/Workspace/EveryImage")))
from src.models import OcrEngine
from PIL import Image
import numpy as np
import time
import cv2

engine = OcrEngine.get_instance()
# Let's get a random image
img = cv2.imread("E:/Workspace/EveryImage/data/test_images/sample.jpg") # if exists
if img is None:
    # create a dummy image
    img = np.random.randint(0, 255, (1280, 1280, 3), dtype=np.uint8)

def test_speed(scale, angle):
    w, h = img.shape[1], img.shape[0]
    s = scale / max(w, h)
    if s < 1.0:
        im_s = cv2.resize(img, (int(w*s), int(h*s)))
    else:
        im_s = img.copy()
        
    t0 = time.time()
    for _ in range(5):
        engine.engine(im_s, use_angle_cls=angle)
    t1 = time.time()
    return (t1 - t0) / 5.0

print("1280 with angle:", test_speed(1280, True))
print("1280 no angle:", test_speed(1280, False))
print("960 no angle:", test_speed(960, False))
print("800 no angle:", test_speed(800, False))
