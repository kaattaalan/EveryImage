import torch
from transformers import AutoModel, AutoProcessor

print("Loading model...")
model = AutoModel.from_pretrained('google/siglip2-base-patch16-224')
processor = AutoProcessor.from_pretrained('google/siglip2-base-patch16-224')

vision_model = model.vision_model
text_model = model.text_model

class VisionWrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, pixel_values):
        return self.m(pixel_values=pixel_values).pooler_output

class TextWrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, input_ids):
        return self.m(input_ids=input_ids).pooler_output

v_wrapper = VisionWrapper(vision_model).eval()
t_wrapper = TextWrapper(text_model).eval()

dummy_pixel = torch.randn(1, 3, 224, 224)
dummy_input_ids = torch.randint(0, 1000, (1, 64))

print("Exporting vision model...")
torch.onnx.export(
    v_wrapper, dummy_pixel, "scratch/siglip_vision.onnx", 
    input_names=["pixel_values"], output_names=["output"], 
    dynamic_axes={"pixel_values": {0: "batch_size"}, "output": {0: "batch_size"}}
)

print("Exporting text model...")
torch.onnx.export(
    t_wrapper, dummy_input_ids, "scratch/siglip_text.onnx", 
    input_names=["input_ids"], output_names=["output"], 
    dynamic_axes={"input_ids": {0: "batch_size"}, "output": {0: "batch_size"}}
)

print("Saving processor config...")
processor.save_pretrained("scratch/siglip2_tokenizer")
print("Done!")
