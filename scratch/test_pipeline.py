import sys
import os
import time
import io
import hashlib
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional
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

def read_and_prepare(raw_item) -> Optional[PreparedItem]:
    lib_id, fpath, fsize, mtime = raw_item
    try:
        with open(fpath, "rb") as f:
            raw_bytes = f.read()

        hasher = hashlib.blake2b()
        hasher.update(raw_bytes[:65536])
        if len(raw_bytes) > 65536:
            hasher.update(raw_bytes[-65536:])
        hhash = hasher.hexdigest()[:16]

        with Image.open(io.BytesIO(raw_bytes)) as im:
            im_oriented = ImageOps.exif_transpose(im)
            im_rgb = im_oriented.convert("RGB")
            w, h = im_rgb.size

            captured_at = None
            try:
                exif = im.getexif()
                if exif:
                    date_str = exif.get(36867) or exif.get(36868) or exif.get(306)
                    if date_str:
                        dt = datetime.strptime(str(date_str).strip()[:19], "%Y:%m:%d %H:%M:%S")
                        captured_at = dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                pass

            if not captured_at:
                captured_at = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")

            scale = min(1.0, 1280.0 / max(w, h))
            if scale < 1.0:
                face_w = int(w * scale)
                face_h = int(h * scale)
                f_im = im_rgb.resize((face_w, face_h), Image.Resampling.BILINEAR)
            else:
                f_im = im_rgb
                scale = 1.0

            thumb_name = f"{hhash}.webp"
            thumb_path = cfg.CACHE_THUMBNAILS / thumb_name
            if not thumb_path.exists():
                t_im = f_im.copy()
                t_im.thumbnail((256, 256), Image.Resampling.BILINEAR)
                t_im.save(thumb_path, "WEBP", quality=80)

            s_im = f_im.resize((224, 224), Image.Resampling.BICUBIC)
            arr = np.array(s_im, dtype=np.float32) / 255.0
            arr = (arr - 0.5) / 0.5
            siglip_tensor = np.transpose(arr, (2, 0, 1))

            bgr_1280 = cv2.cvtColor(np.array(f_im), cv2.COLOR_RGB2BGR)

        return PreparedItem(
            lib_id=lib_id,
            fpath=fpath,
            fsize=fsize,
            mtime=mtime,
            hhash=hhash,
            width=w,
            height=h,
            captured_at=captured_at,
            siglip_tensor=siglip_tensor,
            bgr_1280=bgr_1280,
            scale=scale
        )
    except Exception as e:
        print(f"Error on {fpath.name}: {e}")
        return None

if __name__ == "__main__":
    folder = Path(r"F:\Backup\S21FE\DCIM\Camera")
    files = [p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg"}][:16]
    print(f"Testing pipeline with {len(files)} files...")
    
    items = []
    t0 = time.perf_counter()
    for p in files:
        st = p.stat()
        prep = read_and_prepare((1, p, st.st_size, st.st_mtime))
        if prep:
            items.append(prep)
    t_prep = time.perf_counter() - t0
    print(f"Prepared {len(items)} items in {t_prep:.2f}s ({len(items)/t_prep:.1f} img/s)")

    siglip = SiglipEngine.get_instance()
    t0 = time.perf_counter()
    tensors = [it.siglip_tensor for it in items]
    batch_tensors = np.stack(tensors, axis=0)
    out = siglip.vision_sess.run(None, {"pixel_values": batch_tensors})[0]
    norm = np.linalg.norm(out, axis=-1, keepdims=True)
    embeddings = out / np.maximum(norm, 1e-12)
    t_sig = time.perf_counter() - t0
    print(f"SigLIP on {len(items)} items: {t_sig:.3f}s ({len(items)/t_sig:.1f} img/s)")

    fe = FaceEngine.get_instance()
    t0 = time.perf_counter()
    total_faces = 0
    for it in items:
        faces = fe.detect_and_embed(it.bgr_1280, max_dim=1280)
        total_faces += len(faces)
    t_fe = time.perf_counter() - t0
    print(f"FaceEngine on {len(items)} items: {t_fe:.3f}s ({len(items)/t_fe:.1f} img/s), found {total_faces} faces")
