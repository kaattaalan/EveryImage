import os
from huggingface_hub import hf_hub_download
import shutil
from pathlib import Path

target_dir = Path("E:/Workspace/EveryImage/models/onnx/siglip")
target_dir.mkdir(parents=True, exist_ok=True)

repo_id = "onnx-community/siglip2-base-patch16-224-ONNX"

print("Downloading vision_model.onnx...")
vision_path = hf_hub_download(repo_id=repo_id, filename="onnx/vision_model.onnx")
shutil.copy(vision_path, target_dir / "siglip_vision.onnx")

print("Downloading text_model.onnx...")
text_path = hf_hub_download(repo_id=repo_id, filename="onnx/text_model.onnx")
shutil.copy(text_path, target_dir / "siglip_text.onnx")

print("Downloading tokenizer files...")
files_to_download = [
    "config.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json"
]

for f in files_to_download:
    f_path = hf_hub_download(repo_id=repo_id, filename=f)
    shutil.copy(f_path, target_dir / f)

print("Done downloading SigLIP 2 ONNX models!")
