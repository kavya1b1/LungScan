from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import io
import json
from utils import preprocess_image, segment_image, classify_pathology, generate_report

app = FastAPI()

segment_model = torch.load("model/unet_segmenter.pt", map_location=torch.device('cpu'))
classify_model = torch.load("model/resnet_classifier.pt", map_location=torch.device('cpu'))
segment_model.eval()
classify_model.eval()

@app.post("/upload_image")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("L")

    image_tensor = preprocess_image(image)
    mask = segment_image(image_tensor, segment_model)
    pathology, confidence = classify_pathology(image_tensor, classify_model)
    report = generate_report(pathology, confidence, mask)

    return JSONResponse(content=report)

@app.get("/")
def root():
    return {"message": "AI Medical Imaging API - use /upload_image to POST a medical image."}
