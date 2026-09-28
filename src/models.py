import sys
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image
import pillow_heif
pillow_heif.register_heif_opener()
import onnxruntime as ort
from transformers import AutoTokenizer
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src.config as cfg
from insightface.app import FaceAnalysis
import threading

class SiglipEngine:
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self):
        providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
        vision_path = str(cfg.MODELS_ONNX / "siglip" / "siglip_vision.onnx")
        text_path = str(cfg.MODELS_ONNX / "siglip" / "siglip_text.onnx")

        sess_opts = ort.SessionOptions()
        sess_opts.enable_mem_pattern = True
        sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self.vision_sess = ort.InferenceSession(vision_path, sess_opts, providers=providers)
        self.text_sess = ort.InferenceSession(text_path, sess_opts, providers=providers)
        self.tokenizer = AutoTokenizer.from_pretrained(str(cfg.MODELS_ONNX / "siglip"), local_files_only=True)
        self._text_gate_emb: Optional[np.ndarray] = None
        self.infer_lock = threading.Lock()

    def preprocess_image(self, pil_image: Image.Image) -> np.ndarray:
        im = pil_image.convert("RGB")
        im = im.resize((224, 224), Image.Resampling.BICUBIC)
        arr = np.array(im, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        arr = np.transpose(arr, (2, 0, 1))
        return arr # (3, 224, 224)

    def encode_images_batch(self, pil_images: List[Image.Image]) -> np.ndarray:
        if not pil_images:
            return np.empty((0, 768), dtype=np.float32)
        batch = np.stack([self.preprocess_image(im) for im in pil_images], axis=0) # (B, 3, 224, 224)
        orig_len = len(batch)
        if orig_len < 16:
            pad_len = 16 - orig_len
            batch = np.pad(batch, ((0, pad_len), (0, 0), (0, 0), (0, 0)), mode='constant')
        
        with self.infer_lock:
            out = self.vision_sess.run(["pooler_output"], {"pixel_values": batch})[0]
            
        out = out[:orig_len]
        norm = np.linalg.norm(out, axis=-1, keepdims=True)
        return out / np.maximum(norm, 1e-12)

    def encode_tensors_batch(self, tensors: List[np.ndarray]) -> np.ndarray:
        if not tensors:
            return np.empty((0, 768), dtype=np.float32)
        batch = np.stack(tensors, axis=0) # (B, 3, 224, 224)
        orig_len = len(batch)
        if orig_len < 16:
            pad_len = 16 - orig_len
            batch = np.pad(batch, ((0, pad_len), (0, 0), (0, 0), (0, 0)), mode='constant')
            
        with self.infer_lock:
            out = self.vision_sess.run(["pooler_output"], {"pixel_values": batch})[0]
            
        out = out[:orig_len]
        norm = np.linalg.norm(out, axis=-1, keepdims=True)
        return out / np.maximum(norm, 1e-12)

    def _encode_single_text(self, text: str) -> np.ndarray:
        tokens = self.tokenizer(
            text,
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="np"
        )
        input_ids = tokens["input_ids"].astype(np.int64)
        with self.infer_lock:
            out = self.text_sess.run(["pooler_output"], {"input_ids": input_ids})[0]
        norm = np.linalg.norm(out, axis=-1, keepdims=True)
        return (out / np.maximum(norm, 1e-12))[0]

    def encode_text(self, text: str) -> np.ndarray:
        cleaned = text.strip()
        if not cleaned:
            return np.zeros((768,), dtype=np.float32)
        words = cleaned.split()
        if len(words) <= 4 and not any(cleaned.lower().startswith(p) for p in ("a photo of", "a picture of", "photo of", "picture of")):
            return self._encode_single_text(f"a photo of {cleaned}")
        return self._encode_single_text(cleaned)

    def get_text_gate_embedding(self) -> np.ndarray:
        if self._text_gate_emb is None:
            prompt = "a photo of text, writing, document, sign, page, or screenshot"
            self._text_gate_emb = self.encode_text(prompt)
        return self._text_gate_emb

    def is_text_candidate(self, image_emb: np.ndarray, threshold: float = 0.075) -> bool:
        gate_emb = self.get_text_gate_embedding()
        sim = float(np.dot(image_emb, gate_emb))
        return sim >= threshold


class FaceEngine:
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self):
        providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
        self.app = FaceAnalysis(
            name="buffalo_l",
            root=str(cfg.MODELS_INSIGHTFACE),
            allowed_modules=['detection', 'recognition'],
            providers=providers
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.infer_lock = threading.Lock()
        
    def _pad_image(self, bgr: np.ndarray, target_dim: int = 1280) -> np.ndarray:
        h, w = bgr.shape[:2]
        if h == target_dim and w == target_dim:
            return bgr
        pad_h = max(0, target_dim - h)
        pad_w = max(0, target_dim - w)
        return np.pad(bgr, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')

    def detect_and_embed(self, img_bgr: np.ndarray, max_dim: int = 1280):
        """Returns list of detected faces with bounding boxes, landmarks, and ArcFace 512-d embeddings."""
        h, w = img_bgr.shape[:2]
        scale = min(1.0, float(max_dim) / max(h, w))
        if scale < 1.0:
            target_w = int(w * scale)
            target_h = int(h * scale)
            scaled_bgr = cv2.resize(img_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        else:
            scaled_bgr = img_bgr
            scale = 1.0

        padded_bgr = self._pad_image(scaled_bgr, max_dim)

        with self.infer_lock:
            faces = self.app.get(padded_bgr)
            
        results = []
        for f in faces:
            emb = f.embedding
            emb_norm = emb / np.maximum(np.linalg.norm(emb), 1e-12)
            orig_bbox = [coord / scale for coord in f.bbox.tolist()]
            results.append({
                "bbox": orig_bbox,
                "det_score": float(f.det_score),
                "embedding": emb_norm.astype(np.float32)
            })
        return results

    def detect_and_embed_scaled(self, scaled_bgr: np.ndarray, scale: float = 1.0):
        """Runs detection on pre-scaled BGR array and maps bounding boxes back to original coordinates."""
        padded_bgr = self._pad_image(scaled_bgr, 1280)
        
        with self.infer_lock:
            faces = self.app.get(padded_bgr)
            
        results = []
        inv_scale = 1.0 / max(scale, 1e-6)
        for f in faces:
            emb = f.embedding
            emb_norm = emb / np.maximum(np.linalg.norm(emb), 1e-12)
            orig_bbox = [coord * inv_scale for coord in f.bbox.tolist()]
            results.append({
                "orig_bbox": orig_bbox,
                "scaled_bbox": f.bbox.tolist(),
                "det_score": float(f.det_score),
                "embedding": emb_norm.astype(np.float32)
            })
        return results

from rapidocr_onnxruntime import RapidOCR

class OcrEngine:
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.engine = RapidOCR(det_use_dml=True, rec_use_dml=True, cls_use_dml=True)
        self.infer_lock = threading.Lock()

    def extract_text(self, img_bgr: np.ndarray, min_score: float = 0.5) -> str:
        """
        Runs GPU-accelerated text detection and recognition on the image.
        Returns extracted text formatted as a single cleaned string.
        """
        try:
            # Downscale slightly to drastically improve DBNet FPS (max 960px instead of 1280)
            h, w = img_bgr.shape[:2]
            scale = min(1.0, 960.0 / max(h, w))
            if scale < 1.0:
                ocr_img = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
            else:
                ocr_img = img_bgr

            with self.infer_lock:
                # Disable angle_cls to skip the 30ms-per-text-block overhead
                results, _ = self.engine(ocr_img, use_angle_cls=False)
            if not results:
                return ""
            valid_words = []
            for item in results:
                if len(item) >= 3 and item[2] >= min_score:
                    txt = str(item[1]).strip()
                    if txt:
                        valid_words.append(txt)
            return " ".join(valid_words)
        except Exception as e:
            import logging
            logging.getLogger("OcrEngine").warning("OCR extraction failed: %s", e)
            return ""

