import sys
import time
import queue
import threading
from pathlib import Path
from PIL import Image, ImageOps
import pillow_heif
pillow_heif.register_heif_opener()
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src.database as db
from src.models import OcrEngine

def run_backfill(include_empty: bool = True, batch_commit_size: int = 50):
    print("Starting EveryImage OCR Backfill Pipeline...")
    conn = db.get_connection()
    cur = conn.cursor()

    if include_empty:
        cur.execute("SELECT id, file_path FROM images WHERE ocr_text IS NULL OR ocr_text = '' ORDER BY id ASC")
    else:
        cur.execute("SELECT id, file_path FROM images WHERE ocr_text IS NULL ORDER BY id ASC")

    items = cur.fetchall()
    conn.close()

    total = len(items)
    print(f"Total images queued for OCR analysis: {total}")
    if total == 0:
        print("All images are already processed with OCR.")
        return

    ocr = OcrEngine.get_instance()

    processed = 0
    with_text = 0
    start_time = time.time()

    commit_batch = []

    def flush_commit(batch):
        if not batch:
            return
        c = db.get_connection()
        try:
            cu = c.cursor()
            cu.execute("BEGIN IMMEDIATE")
            for img_id, txt in batch:
                db.save_image_ocr(img_id, txt, cur=cu)
            c.commit()
        except Exception as e:
            c.rollback()
            print(f"Batch commit error: {e}")
        finally:
            c.close()

    for r in items:
        img_id = r["id"]
        p = Path(r["file_path"])
        txt = ""

        if p.exists():
            try:
                with Image.open(p) as im:
                    im_oriented = ImageOps.exif_transpose(im).convert("RGB")
                    w, h = im_oriented.size
                    scale = min(1.0, 1280.0 / max(w, h))
                    if scale < 1.0:
                        im_resized = im_oriented.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)
                    else:
                        im_resized = im_oriented
                    bgr = cv2.cvtColor(np.array(im_resized), cv2.COLOR_RGB2BGR)

                txt = ocr.extract_text(bgr)
            except Exception:
                txt = ""

        commit_batch.append((img_id, txt))
        processed += 1
        if txt:
            with_text += 1

        if len(commit_batch) >= batch_commit_size:
            flush_commit(commit_batch)
            commit_batch = []
            elapsed = max(0.1, time.time() - start_time)
            fps = processed / elapsed
            print(f"[{processed}/{total}] {fps:.1f} imgs/sec | Text found in {with_text} images")

    if commit_batch:
        flush_commit(commit_batch)
        commit_batch = []

    elapsed = max(0.1, time.time() - start_time)
    print(f"OCR Backfill complete in {elapsed:.1f}s. Processed {processed} images. Found text in {with_text} images.")

if __name__ == "__main__":
    include_all = "--all" in sys.argv
    run_backfill(include_empty=include_all)
