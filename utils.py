import torchvision.transforms as transforms
from PIL import Image
import torch
import numpy as np

# 🔹 Preprocess for segmentation (grayscale)
def preprocess_segmentation(image):
    image = image.convert("L")  # Grayscale
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    return transform(image).unsqueeze(0).to(torch.device("mps"))  # Move to MPS

def preprocess_classification(image):
    image = image.convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    return transform(image).unsqueeze(0).to(torch.device("mps"))  # Move to MPS


# 🔹 Segmentation function
def segment_image(image_tensor, model):
    with torch.no_grad():
        output = model(image_tensor)
        mask = torch.sigmoid(output).squeeze().cpu().numpy()  # Move to CPU before NumPy conversion
        binary_mask = (mask > 0.5).astype(np.uint8) * 255
        return binary_mask.astype(np.uint8)

# 🔹 Classification function
def classify_pathology(image_tensor, model):
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1).cpu().numpy()  # Move to CPU before NumPy
        confidence_idx = np.argmax(probabilities, axis=1)
        confidence = probabilities[0][confidence_idx[0]]
        pathology_classes = ["Normal", "Pneumonia", "Tuberculosis"]  # Example classes
        return pathology_classes[confidence_idx[0]], confidence.item()

# 🔹 **Generate report function (missing piece)**
def generate_report(pathology, confidence, segmentation_mask):
    return {
        "PatientID": "Sample_001",
        "ScanType": "Chest-Xray",
        "Findings": [
            {
                "Pathology": pathology,
                "Confidence": round(confidence, 2),
                "Location": "Lung region",
                "BoundingBox": [50, 50, 180, 180],  # Placeholder values
                "Volume_cm3": 0.0  # 2D medical imaging lacks volume data
            }
        ],
        "ReportSummary": f"{pathology} detected with {round(confidence * 100)}% confidence.",
        "ComparativeAnalysis": "No prior scans available."
    }
