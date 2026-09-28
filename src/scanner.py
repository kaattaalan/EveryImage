import sys
import os
import time
import io
import queue
import threading
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import numpy as np
from PIL import Image, ImageOps
import pillow_heif
pillow_heif.register_heif_opener()
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src.config as cfg
import src.database as db
from src.models import SiglipEngine, FaceEngine, OcrEngine

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".heic", ".heif"}

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
    siglip_tensor: np.ndarray # Shape (3, 224, 224), float32 normalized
    bgr_1280: np.ndarray # Shape (H, W, 3), uint8 BGR (max_dim <= 1280)
    scale: float # 1280.0 / max(orig_w, orig_h)

class IndexingEngine:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.state = "idle" # idle, crawling, indexing, stopped, completed, error
        self.stop_requested = threading.Event()
        self.worker_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()

        # Stats
        self.total_files = 0
        self.scanned_files = 0
        self.tagged_files = 0
        self.session_tagged = 0
        self.faces_found = 0
        self.current_file = ""
        self.start_time = 0.0
        self.indexing_start_time = 0.0
        self.elapsed_sec = 0.0
        self.speed_fps = 0.0
        self.error_message: Optional[str] = None

    def get_status(self) -> Dict[str, Any]:
        with self.lock:
            pct = 0.0
            if self.total_files > 0:
                pct = round((self.tagged_files / self.total_files) * 100.0, 1)
            
            elapsed = self.elapsed_sec
            if self.state in ("crawling", "indexing") and self.start_time > 0:
                elapsed = time.time() - self.start_time

            speed = self.speed_fps
            if self.state == "indexing" and self.session_tagged > 0 and self.indexing_start_time > 0:
                elapsed_active = max(0.1, time.time() - self.indexing_start_time)
                speed = round(self.session_tagged / elapsed_active, 1)
            elif self.session_tagged == 0 and self.state in ("crawling", "completed"):
                speed = 0.0

            return {
                "state": self.state,
                "total_files": self.total_files,
                "scanned_files": self.scanned_files,
                "tagged_files": self.tagged_files,
                "faces_found": self.faces_found,
                "percentage": pct,
                "current_file": self.current_file,
                "speed_fps": round(speed, 1),
                "elapsed_sec": round(elapsed, 1),
                "error_message": self.error_message
            }

    def start(self, library_ids: Optional[List[int]] = None) -> bool:
        with self.lock:
            if self.state in ("crawling", "indexing"):
                return False # Already running
            self.state = "crawling"
            self.stop_requested.clear()
            self.total_files = 0
            self.scanned_files = 0
            self.tagged_files = 0
            self.session_tagged = 0
            self.faces_found = 0
            self.current_file = ""
            self.error_message = None
            self.start_time = time.time()
            self.indexing_start_time = 0.0

        self.worker_thread = threading.Thread(target=self._run_pipeline, args=(library_ids,), daemon=True)
        self.worker_thread.start()
        return True

    def stop(self) -> bool:
        with self.lock:
            if self.state not in ("crawling", "indexing"):
                return False
            self.stop_requested.set()
            self.state = "stopped"
            return True

    def _save_face_crop(self, bgr_img: np.ndarray, bbox_scaled: List[float], header_hash: str, face_idx: int) -> str:
        crop_name = f"faces/{header_hash}_{face_idx}.webp"
        crop_path = cfg.CACHE_THUMBNAILS / crop_name
        if not crop_path.exists():
            try:
                h, w = bgr_img.shape[:2]
                x1, y1, x2, y2 = [int(v) for v in bbox_scaled]
                bw, bh = x2 - x1, y2 - y1
                nx1 = max(0, int(x1 - bw * 0.2))
                ny1 = max(0, int(y1 - bh * 0.2))
                nx2 = min(w, int(x2 + bw * 0.2))
                ny2 = min(h, int(y2 + bh * 0.2))
                if nx2 <= nx1 or ny2 <= ny1:
                    return "placeholder.webp"
                crop = bgr_img[ny1:ny2, nx1:nx2]
                if crop is None or crop.size == 0:
                    return "placeholder.webp"
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                pil_crop = Image.fromarray(crop_rgb)
                pil_crop.thumbnail((128, 128), Image.Resampling.BILINEAR)
                crop_path.parent.mkdir(parents=True, exist_ok=True)
                pil_crop.save(crop_path, "WEBP", quality=75)
            except Exception:
                return "placeholder.webp"
        return crop_name

    def _run_pipeline(self, library_ids: Optional[List[int]] = None):
        try:
            # 1. Fetch libraries
            all_libs = db.get_libraries()
            if library_ids:
                libs_to_scan = [l for l in all_libs if l["id"] in library_ids]
            else:
                libs_to_scan = all_libs

            if not libs_to_scan:
                with self.lock:
                    self.state = "completed"
                    self.current_file = "No libraries configured to scan"
                return

            # Phase 1: Fast filesystem crawl with in-memory DB stamp comparison
            with self.lock:
                self.state = "crawling"
                self.current_file = "Scanning library folders..."

            unindexed_items: List[Tuple[int, Path, int, float]] = []
            already_indexed = 0

            def scan_dir_recursive(d_path: Path):
                try:
                    with os.scandir(d_path) as it:
                        for entry in it:
                            if self.stop_requested.is_set():
                                break
                            try:
                                if entry.is_file():
                                    ext = os.path.splitext(entry.name)[1].lower()
                                    if ext in SUPPORTED_EXTS:
                                        st = entry.stat()
                                        yield (Path(entry.path), st.st_size, st.st_mtime)
                                elif entry.is_dir(follow_symlinks=False):
                                    yield from scan_dir_recursive(Path(entry.path))
                            except Exception:
                                continue
                except Exception:
                    pass

            for lib in libs_to_scan:
                lib_path = Path(lib["path"])
                if not lib_path.exists():
                    continue
                lib_stamps = db.get_library_stamps(lib["id"])
                
                for p, fsize, mtime in scan_dir_recursive(lib_path):
                    if self.stop_requested.is_set():
                        break
                    p_str = str(p)
                    if p_str in lib_stamps:
                        prev_size, prev_mtime, has_emb = lib_stamps[p_str]
                        if prev_size == fsize and abs(prev_mtime - mtime) < 0.01 and has_emb == 1:
                            already_indexed += 1
                            continue
                    
                    unindexed_items.append((lib["id"], p, fsize, mtime))
                    if len(unindexed_items) % 50 == 0:
                        with self.lock:
                            self.total_files = already_indexed + len(unindexed_items)
                            self.scanned_files = already_indexed
                            self.tagged_files = already_indexed
                            self.current_file = f"Found {len(unindexed_items)} new images to index..."

            with self.lock:
                self.total_files = already_indexed + len(unindexed_items)
                self.scanned_files = already_indexed
                self.tagged_files = already_indexed
                self.state = "indexing"

            if not unindexed_items:
                with self.lock:
                    self.state = "completed"
                    self.current_file = f"All {already_indexed} images already up to date"
                    self.elapsed_sec = time.time() - self.start_time
                return

            # Preload and warm DirectML models into VRAM before starting active pipeline
            siglip = SiglipEngine.get_instance()
            face_engine = FaceEngine.get_instance()

            # Phase 2: 3-Stage Continuous Streaming Pipeline
            # Stage 1: Dedicated single-threaded sequential HDD stream reader (0 seek thrashing, max HDD health)
            # Stage 2: Parallel CPU decode pool working entirely in RAM (0 extra disk reads)
            # Stage 3: Consumer executing true batch inference on DirectML GPU and single SQLite transaction
            self.indexing_start_time = time.time()

            raw_queue: queue.Queue = queue.Queue(maxsize=32) # Reduced to prevent RAM bloat on large files
            ready_queue: queue.Queue = queue.Queue(maxsize=64) # ~220MB prepared structs
            io_done = threading.Event()
            cpu_done = threading.Event()

            # --- Stage 1: Single Sequential I/O Reader Thread ---
            def io_reader_worker():
                for item in unindexed_items:
                    if self.stop_requested.is_set():
                        break
                    lib_id, fpath, fsize, mtime = item
                    try:
                        # Clean single sequential read - monotonic head movement, zero thrashing
                        with open(fpath, "rb") as f:
                            raw_bytes = f.read()

                        while not self.stop_requested.is_set():
                            try:
                                raw_queue.put(RawItem(lib_id, fpath, fsize, mtime, raw_bytes), timeout=0.1)
                                break
                            except queue.Full:
                                continue
                    except Exception:
                        with self.lock:
                            self.scanned_files += 1

                io_done.set()

            io_thread = threading.Thread(target=io_reader_worker, daemon=True)
            io_thread.start()

            # --- Stage 2: Parallel CPU Decode & Normalization Pool (8 Threads) ---
            def cpu_worker():
                while not self.stop_requested.is_set():
                    try:
                        raw_item: RawItem = raw_queue.get(timeout=0.1)
                    except queue.Empty:
                        if io_done.is_set():
                            break
                        continue

                    try:
                        raw_bytes = raw_item.raw_bytes

                        # Blake2b hash computed in RAM
                        hasher = hashlib.blake2b()
                        hasher.update(raw_bytes[:65536])
                        if len(raw_bytes) > 65536:
                            hasher.update(raw_bytes[-65536:])
                        hhash = hasher.hexdigest()[:16]

                        with Image.open(io.BytesIO(raw_bytes)) as im:
                            exif = im.getexif()
                            
                            # Fast scale-on-load for JPEGs (thumbnail) before expensive rotation
                            im.thumbnail((2560, 2560), Image.Resampling.BILINEAR)
                            
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
                                captured_at = datetime.fromtimestamp(raw_item.mtime).strftime("%Y-%m-%d %H:%M:%S")

                            # Single downscale pass to max_dim 1280
                            scale = min(1.0, 1280.0 / max(w, h))
                            if scale < 1.0:
                                face_w = int(w * scale)
                                face_h = int(h * scale)
                                f_im = im_rgb.resize((face_w, face_h), Image.Resampling.BILINEAR)
                            else:
                                f_im = im_rgb
                                scale = 1.0

                            # Save 256x256 WebP thumbnail to NVMe (only if not already created)
                            thumb_name = f"{hhash}.webp"
                            thumb_path = cfg.CACHE_THUMBNAILS / thumb_name
                            if not thumb_path.exists():
                                t_im = f_im.copy()
                                t_im.thumbnail((256, 256), Image.Resampling.BILINEAR)
                                t_im.save(thumb_path, "WEBP", quality=75)

                            # SigLIP 224x224 float32 normalized tensor
                            s_im = f_im.resize((224, 224), Image.Resampling.BICUBIC)
                            arr = np.array(s_im, dtype=np.float32) / 255.0
                            arr = (arr - 0.5) / 0.5
                            siglip_tensor = np.transpose(arr, (2, 0, 1))

                            # BGR 1280px array for InsightFace
                            bgr_1280 = cv2.cvtColor(np.array(f_im), cv2.COLOR_RGB2BGR)

                        prep = PreparedItem(
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
                        )

                        while not self.stop_requested.is_set():
                            try:
                                ready_queue.put(prep, timeout=0.1)
                                with self.lock:
                                    self.scanned_files += 1
                                break
                            except queue.Full:
                                continue
                    except Exception:
                        with self.lock:
                            self.scanned_files += 1

            cpu_threads = [threading.Thread(target=cpu_worker, daemon=True) for _ in range(8)]
            for t in cpu_threads:
                t.start()

            def cpu_supervisor():
                for t in cpu_threads:
                    t.join()
                cpu_done.set()
                ready_queue.put(None)

            threading.Thread(target=cpu_supervisor, daemon=True).start()

            # --- Stage 3: Consumer (DirectML GPU Inference & Atomic SQLite Commit) ---
            siglip = SiglipEngine.get_instance()
            face_engine = FaceEngine.get_instance()

            batch_size = 16
            current_batch: List[PreparedItem] = []

            while not self.stop_requested.is_set():
                try:
                    item = ready_queue.get(timeout=0.2)
                except queue.Empty:
                    if cpu_done.is_set():
                        break
                    continue

                if item is None:
                    # Sentinel reached
                    break

                current_batch.append(item)
                # STRICT BATCHING: Only process when full batch of 16 is accumulated
                if len(current_batch) >= batch_size:
                    self._process_consumer_batch(current_batch, siglip, face_engine)
                    current_batch = []

            # Process remaining items at the very end of the scan
            if current_batch and not self.stop_requested.is_set():
                self._process_consumer_batch(current_batch, siglip, face_engine)
                current_batch = []

            io_thread.join(timeout=2.0)

            with self.lock:
                if self.stop_requested.is_set():
                    self.state = "stopped"
                else:
                    self.state = "completed"
                self.elapsed_sec = time.time() - self.start_time

        except Exception as e:
            with self.lock:
                self.state = "error"
                self.error_message = str(e)

    def _process_consumer_batch(self, batch: List[PreparedItem], siglip: SiglipEngine, face_engine: FaceEngine):
        if not batch:
            return

        # 1. GPU SigLIP Vision Batch Inference (DirectML)
        tensors = [it.siglip_tensor for it in batch]
        embeddings = siglip.encode_tensors_batch(tensors)

        # 2. GPU Accelerated OCR (RapidOCR on DirectML - scans all images without missing signs or bills)
        ocr_engine = OcrEngine.get_instance()
        batch_ocr_text = []
        for it in batch:
            txt = ocr_engine.extract_text(it.bgr_1280)
            batch_ocr_text.append(txt)

        # 3. GPU Face Detection (DirectML - Optimized detection + recognition only)
        batch_faces = []
        for it in batch:
            detected = face_engine.detect_and_embed_scaled(it.bgr_1280, it.scale)
            batch_faces.append(detected)

        # 4. Atomic SQLite Batch Commit
        conn = db.get_connection()
        try:
            cur = conn.cursor()
            cur.execute("BEGIN IMMEDIATE")

            batch_faces_count = 0
            for idx, it in enumerate(batch):
                # Upsert image
                img_id = db.upsert_image_record(
                    it.lib_id, str(it.fpath), it.fpath.name,
                    it.fsize, it.mtime, it.hhash, it.width, it.height, it.captured_at,
                    cur=cur
                )
                # Save embedding
                db.save_image_embedding(img_id, embeddings[idx], cur=cur)

                # Save OCR text (empty string marks image as already evaluated)
                db.save_image_ocr(img_id, batch_ocr_text[idx], cur=cur)

                # Save faces
                faces_for_img = batch_faces[idx]
                for f_idx, f_info in enumerate(faces_for_img):
                    crop_name = self._save_face_crop(it.bgr_1280, f_info["scaled_bbox"], it.hhash, f_idx)
                    face_emb = f_info["embedding"]

                    # Face clustering
                    matched_person_id = db.find_closest_person_for_face(face_emb, distance_threshold=0.45, cur=cur)
                    if matched_person_id is None:
                        matched_person_id = db.create_person(name=f"Person {img_id}_{f_idx+1}", cur=cur)

                    face_id = db.save_face(
                        image_id=img_id,
                        person_id=matched_person_id,
                        bbox=f_info["orig_bbox"],
                        det_score=f_info["det_score"],
                        crop_thumb=crop_name,
                        embedding=face_emb,
                        cur=cur
                    )

                    cur.execute("UPDATE persons SET cover_face_id = ? WHERE id = ? AND cover_face_id IS NULL", (face_id, matched_person_id))
                    batch_faces_count += 1

            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

        # 4. Update Stats
        with self.lock:
            self.tagged_files += len(batch)
            self.session_tagged += len(batch)
            self.faces_found += batch_faces_count
            self.current_file = batch[-1].fpath.name
            elapsed_active = max(0.1, time.time() - self.indexing_start_time)
            self.speed_fps = round(self.session_tagged / elapsed_active, 1)

class LibraryWatcher:
    _instance = None
    _lock = threading.RLock()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self):
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler

        self.observer = Observer()
        self.watched_paths = set()
        self.active_watches = {}
        self.is_running = False
        self.lock = threading.RLock()
        self.debounce_timer: Optional[threading.Timer] = None
        self.pending_files = set()

        outer = self
        class Handler(FileSystemEventHandler):
            def on_created(self, event):
                if not event.is_directory:
                    outer.handle_file_event(event.src_path)

            def on_moved(self, event):
                if not event.is_directory:
                    outer.handle_file_event(event.dest_path)

        self.handler = Handler()

    def handle_file_event(self, path_str: str):
        ext = os.path.splitext(path_str)[1].lower()
        if ext in SUPPORTED_EXTS:
            with self.lock:
                self.pending_files.add(path_str)
                if self.debounce_timer:
                    self.debounce_timer.cancel()
                self.debounce_timer = threading.Timer(2.0, self._trigger_scan)
                self.debounce_timer.daemon = True
                self.debounce_timer.start()

    def _trigger_scan(self):
        with self.lock:
            self.pending_files.clear()
        engine = IndexingEngine.get_instance()
        with engine.lock:
            if engine.state in ("idle", "completed", "stopped", "error"):
                engine.start()

    def start(self):
        with self.lock:
            if self.is_running:
                return
            from watchdog.observers import Observer
            self.observer = Observer()
            self.active_watches.clear()
            self.watched_paths.clear()
            self.sync_watched_paths()
            self.observer.start()
            self.is_running = True

    def stop(self):
        with self.lock:
            if not self.is_running:
                return
            if self.debounce_timer:
                self.debounce_timer.cancel()
            try:
                self.observer.stop()
                self.observer.join(timeout=1.5)
            except Exception:
                pass
            self.is_running = False

    def sync_watched_paths(self):
        with self.lock:
            current_libs = db.get_libraries()
            current_paths = {l["path"] for l in current_libs if Path(l["path"]).exists()}

            # Remove watches for deleted libraries
            for p in list(self.watched_paths):
                if p not in current_paths:
                    watch = self.active_watches.pop(p, None)
                    if watch:
                        try:
                            self.observer.unschedule(watch)
                        except Exception:
                            pass
                    self.watched_paths.remove(p)

            # Add watches for new libraries
            for p in current_paths:
                if p not in self.watched_paths:
                    try:
                        watch = self.observer.schedule(self.handler, p, recursive=True)
                        self.active_watches[p] = watch
                        self.watched_paths.add(p)
                    except Exception as e:
                        import logging
                        logging.getLogger("LibraryWatcher").warning(f"Could not watch directory {p}: {e}")

    def get_status(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "active": self.is_running,
                "watched_count": len(self.watched_paths),
                "watched_paths": list(self.watched_paths)
            }

