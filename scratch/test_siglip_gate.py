import sys
from pathlib import Path
sys.path.insert(0, str(Path("E:/Workspace/EveryImage")))
from src.models import SiglipEngine
from PIL import Image
import numpy as np
import sqlite3

siglip = SiglipEngine.get_instance()

p_text = siglip.encode_text("a photo containing text, document, sign, receipt, or writing")
p_none = siglip.encode_text("a normal photo without text, like landscape, portrait, animal, or object")

conn = sqlite3.connect("E:/Workspace/EveryImage/data/everyimage.db")
cur = conn.cursor()

cur.execute("SELECT file_path, ocr_text FROM images WHERE ocr_text IS NOT NULL AND ocr_text != '' LIMIT 10")
text_imgs = cur.fetchall()

cur.execute("SELECT file_path FROM images WHERE ocr_text IS NULL OR ocr_text = '' LIMIT 10")
no_text_imgs = cur.fetchall()

def eval_image(path):
    im = Image.open(path).convert("RGB")
    emb = siglip.encode_images_batch([im])[0]
    sim_text = float(np.dot(emb, p_text))
    sim_none = float(np.dot(emb, p_none))
    return sim_text, sim_none

print("Text images:")
for p, txt in text_imgs:
    try:
        st, sn = eval_image(p)
        print(f"[{st > sn}] text:{st:.3f} none:{sn:.3f}  - {txt[:30]}")
    except Exception as e:
        print(f"Error {p}: {e}")

print("\nNo text images:")
for (p,) in no_text_imgs:
    try:
        st, sn = eval_image(p)
        print(f"[{st > sn}] text:{st:.3f} none:{sn:.3f}")
    except Exception as e:
        pass
