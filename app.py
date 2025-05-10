import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import gdown
import json

st.set_page_config(page_title="Improved 3D Object Dimension Estimator", layout="centered")

MODEL_FILE_ID = "1KuTHvjp0F5Fo4daiopwiu9ukGNfveVen"
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
MODEL_PATH = "trained_model.pth"

def download_model_from_drive():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading model from Google Drive..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        st.success("✅ Model downloaded.")

download_model_from_drive()

@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 3)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=False))
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict_dimensions(image, model):
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        log_preds = model(input_tensor).squeeze().numpy()
        dims = np.exp(log_preds)
    return dims

st.title("📏 Improved 3D Object Dimension Estimator")
st.write("Estimate real-world dimensions (L × B × H in meters) from an image using fine-tuned ResNet18.")

st.sidebar.header("🖼️ Choose Image")
mode = st.sidebar.radio("Input Type", ["Upload your own", "Use sample image"])

img = None
bbox = None
metadata = {}

if os.path.exists("pix3d-samples/metadata.json"):
    with open("pix3d-samples/metadata.json", "r") as f:
        metadata = json.load(f)

if mode == "Upload your own":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)
elif mode == "Use sample image":
    images = [f for f in os.listdir("pix3d-samples") if f.endswith((".png", ".jpg"))]
    selected = st.selectbox("Select sample image", images)
    if selected:
        path = os.path.join("pix3d-samples", selected)
        img = Image.open(path).convert("RGB")
        st.image(img, caption=selected, use_column_width=True)
        if selected in metadata:
            bbox = metadata[selected]["bbox"]

if img and bbox:
    img = img.crop((bbox[0], bbox[1], bbox[2], bbox[3]))

if img:
    model = load_model()
    dims = predict_dimensions(img, model)
    length, breadth, height = dims[0], dims[1], dims[2]
    volume = length * breadth * height

    st.subheader("📐 Predicted Dimensions")
    st.markdown(f"🔸 **Length**:  `{length:.3f} m`")
    st.markdown(f"🔸 **Breadth**: `{breadth:.3f} m`")
    st.markdown(f"🔸 **Height**:  `{height:.3f} m`")
    st.markdown(f"🔸 **Volume**:   `{volume:.4f} m³`")
