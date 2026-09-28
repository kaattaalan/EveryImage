import os
import sys
from pathlib import Path

# Workspace Root: strictly self-contained
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

CACHE_DIR = WORKSPACE_ROOT / "cache"
CACHE_HF = CACHE_DIR / "huggingface"
CACHE_TORCH = CACHE_DIR / "torch"
CACHE_TEMP = CACHE_DIR / "temp"
CACHE_THUMBNAILS = CACHE_DIR / "thumbnails"
MODELS_DIR = WORKSPACE_ROOT / "models"
MODELS_INSIGHTFACE = MODELS_DIR / "insightface"
MODELS_ONNX = MODELS_DIR / "onnx"
DATA_DIR = WORKSPACE_ROOT / "data"

# Create all directories if they do not exist
for d in [
    CACHE_DIR,
    CACHE_HF,
    CACHE_TORCH,
    CACHE_TEMP,
    CACHE_THUMBNAILS,
    MODELS_DIR,
    MODELS_INSIGHTFACE,
    MODELS_ONNX,
    DATA_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)

# Strictly configure environment variables to prevent spillage
os.environ["HF_HOME"] = str(CACHE_HF)
os.environ["TORCH_HOME"] = str(CACHE_TORCH)
os.environ["TEMP"] = str(CACHE_TEMP)
os.environ["TMP"] = str(CACHE_TEMP)
os.environ["INSIGHTFACE_HOME"] = str(MODELS_INSIGHTFACE)

import tempfile
tempfile.tempdir = str(CACHE_TEMP)

# Feature flags: Set to False to completely bypass learning layer and use raw baseline
ENABLE_FACE_LEARNER = True

