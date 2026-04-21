from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import torch
import numpy as np
from PIL import Image
import io

from model import DeepfakeDetector

app = FastAPI(title="MobileNet4DFR Deepfake Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model ──────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading model on {device}...")

model = DeepfakeDetector.load_from_checkpoint(
    "weights/cifake_MobileNetV3-Small_epoch=8-train_acc=1.00-val_acc=0.98.ckpt",
    map_location=device,
    # ── must match the hparams used during training ──
    num_classes=2,
    backbone="MobileNetV3-Small",
    freeze_backbone=False,
    add_magnitude_channel=False,
    add_fft_channel=True,
    add_lbp_channel=True,
    add_gabor_channel=False,
)
model.to(device)
model.eval()


# ── Preprocessing ────────────────────────────────────────────────────────────
# BUG 1 + 2 FIX: send plain 3-channel RGB.
# model.forward() calls add_new_channels() internally — do NOT duplicate it here.
def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    arr = np.array(image, dtype=np.float32) / 255.0          # (224, 224, 3)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()  # (1, 3, 224, 224)
    return tensor


# ── Predict endpoint ─────────────────────────────────────────────────────────
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        tensor = preprocess_image(image_bytes).to(device)

        with torch.no_grad():
            output = model(tensor)                         # returns {"logits": Tensor}

            # BUG 3 FIX: access the "logits" key, not the raw dict
            logit = output["logits"]                       # shape (1, 1)
            prob = torch.sigmoid(logit).item()             # probability of is_real == 1

        # BUG 4 FIX: training label is is_real (1 = REAL, 0 = FAKE)
        # → sigmoid > 0.5 means the model predicts REAL, not FAKE
        is_real = prob > 0.5
        confidence = prob if is_real else (1.0 - prob)

        return {
            "status": "success",
            "prediction": "REAL" if is_real else "FAKE",
            "confidence": f"{confidence * 100:.2f}%",
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


# ── Serve frontend ───────────────────────────────────────────────────────────
@app.get("/")
def main_page():
    with open("index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)