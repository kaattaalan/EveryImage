import sys
from pathlib import Path

# Insert workspace root to import config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src.config as cfg

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoModel

class SiglipVisionEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.vision_model = model.vision_model

    def forward(self, pixel_values):
        outputs = self.vision_model(pixel_values=pixel_values)
        # pooled output
        image_embeds = outputs.pooler_output
        # L2 normalize
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        return image_embeds

class SiglipTextEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.text_model = model.text_model

    def forward(self, input_ids):
        outputs = self.text_model(input_ids=input_ids)
        # pooled output
        text_embeds = outputs.pooler_output
        # L2 normalize
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        return text_embeds

def export():
    model_name = "google/siglip-base-patch16-224"
    print(f"Loading {model_name}...")
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    output_dir = cfg.MODELS_ONNX / "siglip"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save processor/tokenizer files for offline inference
    processor.save_pretrained(output_dir)

    vision_encoder = SiglipVisionEncoder(model)
    vision_encoder.eval()

    text_encoder = SiglipTextEncoder(model)
    text_encoder.eval()

    # Dummy inputs
    dummy_pixel = torch.randn(1, 3, 224, 224)
    dummy_input_ids = torch.randint(0, 1000, (1, 64), dtype=torch.long)

    vision_onnx_path = output_dir / "siglip_vision.onnx"
    text_onnx_path = output_dir / "siglip_text.onnx"

    print("Exporting vision encoder to ONNX...")
    torch.onnx.export(
        vision_encoder,
        dummy_pixel,
        str(vision_onnx_path),
        input_names=["pixel_values"],
        output_names=["image_embeds"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "image_embeds": {0: "batch_size"}
        },
        opset_version=18,
        dynamo=False
    )
    print(f"Vision encoder exported: {vision_onnx_path}")

    print("Exporting text encoder to ONNX...")
    torch.onnx.export(
        text_encoder,
        dummy_input_ids,
        str(text_onnx_path),
        input_names=["input_ids"],
        output_names=["text_embeds"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "text_embeds": {0: "batch_size"}
        },
        opset_version=18,
        dynamo=False
    )
    print(f"Text encoder exported: {text_onnx_path}")
    print("SigLIP export completed successfully!")

if __name__ == "__main__":
    export()
