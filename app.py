import os
import numpy as np
import streamlit as st
import torch
import torch.nn as nn                 
import torch.nn.functional as F
from PIL import Image
import pickle
from torchvision import transforms
from torchvision.models import resnet50
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ---- CONFIG ----
MODEL_PATH = "best_resnet50.pth"
DEVICE = torch.device("cpu")

# ---- CLASSES (MUST MATCH TRAINING ORDER) ----
CLASSES = [
    "biotite",
    "bornite",
    "chrysocolla",
    "malachite",
    "muscovite",
    "pyrite",
    "quartz"
]

# ---- TRANSFORM (IDENTICAL TO TRAINING VAL TRANSFORM) ----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---- MODEL LOADING ----
@st.cache_resource
def load_model():
    model = resnet50(weights=None)

    # MUST match training
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE)
    )

    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# Feature extractor
feature_extractor = nn.Sequential(*list(model.children())[:-1])
feature_extractor.to(DEVICE)
feature_extractor.eval()

# Load OOD stats  ← 🔴 THIS SECTION
def mahalanobis(x, mean, inv_cov):
    diff = x - mean
    dist = np.sqrt(diff.T @ inv_cov @ diff)
    return dist
with open("ood_stats_resnet.pkl", "rb") as f:
    stats = pickle.load(f)

means = stats["means"]
inv_cov = stats["inv_cov"]
threshold = stats["threshold"]



# ---- LOGO + TITLE ----   ← PUT IT HERE
col1, col2, col3 = st.columns([1,6,1])

with col1:
    inst_logo = os.path.join(BASE_DIR, "nitn.logo.png")
    if os.path.exists(inst_logo):
        st.image("nitn.logo.png", width=200)

with col2:
    st.markdown(
        "<h1 style='text-align:center;'>Mineralogical Material Detection</h1>",
        unsafe_allow_html=True
    )

with col3:
    texmin_logo = os.path.join(BASE_DIR, "texmin_logo.png")
    if os.path.exists(texmin_logo):
       st.image("texmin_logo.png", width=200)

st.markdown("<hr>", unsafe_allow_html=True)



st.subheader("Mineral Image Input")

image = None

input_mode = st.radio(
    "Choose Input Method",
    ["Dropdown", "Camera", "Upload"],
    horizontal=True
)

SAMPLES_DIR = "samples"

# ---------------- DROPDOWN ----------------

if input_mode == "Dropdown":

    samples = {
        "biotite": "biotite.png",
        "bornite": "bornite.png",
        "chrysocolla": "chrysocolla.png",
        "malachite": "malachite.png",
        "muscovite": "muscovite.png",
        "pyrite": "pyrite.png",
        "quartz": "quartz.png"
    }

    option = st.selectbox("Select sample", ["None"] + list(samples.keys()))

    if option != "None":
        path = os.path.join(SAMPLES_DIR, samples[option])

        if os.path.exists(path):
            image = Image.open(path).convert("RGB")
            st.image(image, width=300)
        else:
            st.error("Sample image not found")

# ---------------- CAMERA ----------------

elif input_mode == "Camera":
    cam = st.camera_input("Capture image")

    if cam:
        image = Image.open(cam).convert("RGB")

# ---------------- UPLOAD ----------------

elif input_mode == "Upload":
    file = st.file_uploader("Upload image", type=["jpg","jpeg","png"])

    if file:
        image = Image.open(file).convert("RGB")


        st.set_page_config(
    page_title="Geological Material Detection",
    layout="wide"
)


# ============================================================
# PREDICTION
# ============================================================

if image is None:
    st.warning("Please select or upload an image first.")
    st.stop()

else:

    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Result")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(image, width=300)

    with col2:

        import numpy as np

       # ---- PREPROCESS ----
img_tensor = transform(image).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    
    outputs = model(img_tensor)

    # ---- CLASSIFICATION ----
    probs = F.softmax(outputs, dim=1)
    pred = torch.argmax(probs, dim=1).item()

    # ---- FEATURE EXTRACTION ----
    feat = feature_extractor(img_tensor)
    feat = feat.view(feat.size(0), -1)
    feat = feat.cpu().numpy().flatten()

# ---- NORMALIZATION (OUTSIDE BLOCK) ----
feat = feat / (np.linalg.norm(feat) + 1e-8)

# ---- COSINE DISTANCE ----
distance = 1 - np.dot(feat, means[pred])

# ---- OOD CHECK ----
distance = 1 - np.dot(feat, means[pred])
if distance > 0.55:
    
    st.error("Unknown Mineral (OOD detected)")
    st.write(f"Distance: {distance:.4f}")

else:
    # ensure probs is 1D tensor
    probs = F.softmax(outputs, dim=1).squeeze()

    label = CLASSES[pred]
    confidence = probs[pred].item()

    st.success(f"Prediction: {label}")
    st.write(f"Confidence: {confidence*100:.2f}%")
    st.write(f"Distance: {distance:.4f}")
  
  
   # ---- ABOUT PROJECT

st.markdown("", unsafe_allow_html=True)

st.subheader("About the Project")

st.write(""" To improve reliability, the system incorporates Out-of-Distribution (OOD)
detection using cosine similarity in the learned feature space. If an input
image does not resemble the training data distribution, it is labeled as
Unknown Mineral. Only validated predictions are displayed to the user.
""")

st.subheader("Acknowledgement")

st.write("""
This project was supported under the TEXMiN UG/PG Fellowship, a Technology
Innovation Hub at IIT (ISM) Dhanbad, and carried out under an institutional
MoU collaboration with the National Institute of Technology Nagaland.
""")

st.subheader("Supervisor")

st.write("""
I express my sincere gratitude to
Dr. Dushmanta Kumar Das, Associate Professor,
Department of Electrical and Electronics Engineering,
National Institute of Technology Nagaland,
for his continuous guidance, valuable insights and academic support
throughout the development of this Mineral Classification system.
""")

st.markdown(
"<h3 style='text-align:center;color:green;'>Thank You</h3>",
unsafe_allow_html=True
)