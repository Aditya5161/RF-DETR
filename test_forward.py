import torch
from models.rf_detr import RFDETR

device = "cuda" if torch.cuda.is_available() else "cpu"

model = RFDETR(num_classes=80)
model.to(device)
model.eval()

# Fake input image (batch_size=2, 3 channels, 640x640)
images = torch.randn(2, 3, 640, 640).to(device)

with torch.no_grad():
    outputs = model(images)

print("Pred logits shape:", outputs["pred_logits"].shape)
print("Pred boxes shape:", outputs["pred_boxes"].shape)
