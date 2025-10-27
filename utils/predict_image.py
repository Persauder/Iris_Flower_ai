from pathlib import Path
from typing import List, Tuple
from PIL import Image
import torch
from torchvision import transforms, models
from torch import nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_image_model(weights_path: Path):
    ckpt = torch.load(weights_path, map_location=DEVICE)
    classes: List[str] = ckpt["classes"]
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.last_channel, len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(DEVICE)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return model, transform, classes

@torch.no_grad()
def predict_image(path: Path, model, transform, classes: List[str]) -> Tuple[str, float]:
    img = Image.open(path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)
    logits = model(x)
    probs = logits.softmax(dim=1).squeeze(0).cpu().numpy()
    idx = probs.argmax()
    return classes[idx], float(probs[idx])