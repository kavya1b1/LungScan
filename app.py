from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from utils import preprocess_segmentation, preprocess_classification, segment_image, classify_pathology, generate_report
import torch
import io
from models import DummyUNet
from torchvision import models
import torchvision.transforms as transforms
from models import DummyUNet


# Initialize FastAPI app
app = FastAPI()

# --- DEVICE CONFIGURATION --- #
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --- MODEL INITIALIZATION --- #
# Load segmentation model correctly
segment_model = DummyUNet().to(device)
segment_model.load_state_dict(torch.load("model/unet_segmenter.pt", map_location=device))
segment_model.eval()

# Load classification model correctly
classify_model = models.resnet18(pretrained=True)
classify_model.fc = torch.nn.Linear(classify_model.fc.in_features, 2)  # Two-class classification
classify_model.to(device)
classify_model.load_state_dict(torch.load("model/resnet_classifier.pt", map_location=device))
classify_model.eval()

# --- ROUTES --- #
@app.post("/upload_image")
async def upload_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Use the right preprocess functions
        seg_tensor = preprocess_segmentation(image)
        cls_tensor = preprocess_classification(image)

        # Inference
        mask = segment_image(seg_tensor, segment_model)
        pathology, confidence = classify_pathology(cls_tensor, classify_model)
        report = generate_report(pathology, confidence, mask)

        return JSONResponse(content=report)

    except Exception as e:
        print("Upload failed with error:", str(e))
        return JSONResponse(content={"error": str(e)}, status_code=400)


@app.get("/")
def root():
    return {"message": "AI Medical Imaging API - use /upload_image to POST a medical image."}
