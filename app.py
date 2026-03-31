import os
import numpy as np
import torch
import torch.nn as nn                 
import torch.nn.functional as F
from PIL import Image
import pickle
from torchvision import transforms
from torchvision.models import resnet50
import io
import gdown
from fastapi.middleware.cors import CORSMiddleware


# ✅ ADD FASTAPI
from fastapi import FastAPI, File, UploadFile

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- CONFIG ----
DEVICE = torch.device("cpu")
MODEL_PATH = os.path.join(BASE_DIR, "best_resnet50.pth")

def load_model():
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))

    # ✅ Ensure model exists
    if not os.path.exists(MODEL_PATH):
        url = "https://drive.google.com/uc?id=16knNIyEL_biTfWMOWiKzWQgJzuAJ1y0L"
        gdown.download(url, MODEL_PATH, quiet=False)

    # ❌ remove weights_only=True
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE)
    )

    model.to(DEVICE)
    model.eval()
    return model


# ---- CLASSES ----
CLASSES = [
    "biotite","bornite","chrysocolla",
    "malachite","muscovite","pyrite","quartz"
]

# ---- TRANSFORM ----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

model = None
feature_extractor = None

def get_model():
    global model, feature_extractor

    if model is None:
        model = load_model()

        feature_extractor = nn.Sequential(*list(model.children())[:-1])
        feature_extractor.to(DEVICE)
        feature_extractor.eval()

    return model, feature_extractor
# ---- LOAD OOD ----
with open("ood_stats_resnet.pkl", "rb") as f:
    stats = pickle.load(f)

means = stats["means"]
inv_cov = stats["inv_cov"]
threshold = stats["threshold"]

# ============================================================
# ✅ FASTAPI ENDPOINT (ONLY ADDITION)
# ============================================================
@app.get("/")
def health():
    return {"status": "server awake"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except:
        return {"error": "Invalid image"}

    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    model, feature_extractor = get_model()

    with torch.no_grad():
        
        outputs = model(img_tensor)

        probs = F.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()

        feat = feature_extractor(img_tensor)
        feat = feat.view(feat.size(0), -1)
        feat = feat.cpu().numpy().flatten()

    feat = feat / (np.linalg.norm(feat) + 1e-8)

    distance = 1 - np.dot(feat, means[pred])

   # ---- OOD CHECk----
    if distance > 0.55:
        return {
            "label": "unknown",
            "confidence": 0.0,
            "distance": float(distance)
        }
    label = CLASSES[pred]
    confidence = probs[0][pred].item()

    return {
        "label": label,
        "confidence": float(confidence),
        "distance": float(distance)
    }