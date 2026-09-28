import sys
import os
import time
import io
import queue
import threading
import hashlib
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional, Tuple
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image, ImageOps
import pillow_heif
pillow_heif.register_heif_opener()
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src.config as cfg
import src.database as db
from src.models import SiglipEngine, FaceEngine

@dataclass
class RawItem:
    lib_id: int
    fpath: Path
    fsize: int
    mtime: float
    raw_bytes: bytes

@dataclass
class PreparedItem:
    lib_id: int
    fpath: Path
    fsize: int
    mtime: float
    hhash: str
    width: int
    height: int
    captured_at: Optional[str]
    siglip_tensor: np.ndarray
    bgr_1280: np.ndarray
    scale: float

def test_pipeline():
    folder = Path(r"F:\Backup\S21FE\DCIM\Camera")
    files = [p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".heic"}][:32]
    print(f"Testing 3-stage streaming pipeline with {len(files)} files...")

    raw_queue = queue.Queue(maxsize=32)
    ready_queue = queue.Queue(maxsize=32)
    io_done = threading.Event()
    cpu_done = threading.Event()

    # Stage 1: Single Sequential I/O Reader Thread
    def io_reader():
        t0 = time.perf_counter()
        total_bytes = 0
        for p in files:
            st = p.stat()
            with open(p, "rb") as f:
                data = f.read()
            total_bytes += len(data)
            raw_queue.put(RawItem(1, p, st.st_size, st.st_mtime, data))
        io_done.set()
        t_io = time.perf_counter() - t0
        mb = total_bytes / (1024*1024)
        print(f"[I/O Reader] Read {len(files)} files ({mb:.1f} MB) in {t_io:.2f}s ({mb/t_io:.1f} MB/s, {len(files)/t_io:.1f} files/s)")

    # Stage 2: Parallel CPU Decode & Preprocess
    def cpu_worker():
        while True:
            try:
                raw_item = raw_queue.get(timeout=0.1)
            except queue.Empty:
                if io_done.is_set():
                    break
                continue

            try:
                raw_bytes = raw_item.raw_bytes
                hasher = hashlib.blake2b()
                hasher.update(raw_bytes[:65536])
                if len(raw_bytes) > 65536:
                    hasher.update(raw_bytes[-65536:])
                hhash = hasher.hexdigest()[:16]

                with Image.open(io.BytesIO(raw_bytes)) as im:
                    im_oriented = ImageOps.exif_transpose(im)
                    im_rgb = im_oriented.convert("RGB")
                    w, h = im_rgb.size

                    captured_at = datetime.fromtimestamp(raw_item.mtime).strftime("%Y-%m-%d %H:%M:%S")
                    scale = min(1.0, 1280.0 / max(w, h))
                    if scale < 1.0:
                        face_w = int(w * scale)
                        face_h = int(h * scale)
                        f_im = im_rgb.resize((face_w, face_h), Image.Resampling.BILINEAR)
                    else:
                        f_im = im_rgb
                        scale = 1.0

                    s_im = f_im.resize((224, 224), Image.Resampling.BICUBIC)
                    arr = np.array(s_im, dtype=np.float32) / 255.0
                    arr = (arr - 0.5) / 0.5
                    siglip_tensor = np.transpose(arr, (2, 0, 1))
                    bgr_1280 = cv2.cvtColor(np.array(f_im), cv2.COLOR_RGB2BGR)

                ready_queue.put(PreparedItem(
                    lib_id=raw_item.lib_id,
                    fpath=raw_item.fpath,
                    fsize=raw_item.fsize,
                    mtime=raw_item.mtime,
                    hhash=hhash,
                    width=w,
                    height=h,
                    captured_at=captured_at,
                    siglip_tensor=siglip_tensor,
                    bgr_1280=bgr_1280,
                    scale=scale
                ))
            except Exception as e:
                print(f"CPU decode error: {e}")

    # Launch I/O thread
    t_start = time.perf_counter()
    io_thread = threading.Thread(target=io_reader, daemon=True)
    io_thread.start()

    # Launch CPU worker pool
    cpu_threads = [threading.Thread(target=cpu_worker, daemon=True) for _ in range(6)]
    for t in cpu_threads:
        t.start()

    # Stage 3: Consumer (GPU Inference)
    siglip = SiglipEngine.get_instance()
    fe = FaceEngine.get_instance()

    batch_size = 16
    current_batch = []
    processed_count = 0
    total_faces = 0

    while True:
        try:
            item = ready_queue.get(timeout=0.2)
            current_batch.append(item)
            if len(current_batch) >= batch_size:
                t0 = time.perf_counter()
                tensors = [it.siglip_tensor for it in current_batch]
                embs = siglip.encode_tensors_batch(tensors)
                for it in current_batch:
                    detected = fe.detect_and_embed_scaled(it.bgr_1280, it.scale)
                    total_faces += len(detected)
                t_gpu = time.perf_counter() - t0
                processed_count += len(current_batch)
                print(f"[Consumer] Processed TRUE batch {len(current_batch)} in {t_gpu*1000:.1f}ms ({len(current_batch)/t_gpu:.1f} img/s)")
                current_batch = []
        except queue.Empty:
            if io_done.is_set() and raw_queue.empty() and ready_queue.empty():
                if current_batch:
                    t0 = time.perf_counter()
                    tensors = [it.siglip_tensor for it in current_batch]
                    embs = siglip.encode_tensors_batch(tensors)
                    for it in current_batch:
                        detected = fe.detect_and_embed_scaled(it.bgr_1280, it.scale)
                        total_faces += len(detected)
                    t_gpu = time.perf_counter() - t0
                    processed_count += len(current_batch)
                    print(f"[Consumer] Processed final batch {len(current_batch)} in {t_gpu*1000:.1f}ms")
                    current_batch = []
                break

    t_total = time.perf_counter() - t_start
    print(f"Total: Processed {processed_count} images ({total_faces} faces) in {t_total:.2f}s ({processed_count/t_total:.1f} img/s)!")

if __name__ == "__main__":
    test_pipeline()
