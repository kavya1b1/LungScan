# AI-Driven Medical Imaging Diagnostics System

This project is a web-based AI system designed to:
- Upload and analyze medical images (DICOM, PNG, JPEG)
- Detect anomalies (e.g., tumors, fractures)
- Segment regions of interest
- Predict pathologies using pretrained deep learning models
- Generate structured diagnostic reports

---

## 🔧 Features
- **Image formats**: Supports DICOM, PNG, JPEG
- **Segmentation**: U-Net model to highlight anomalies
- **Classification**: ResNet-50 for pathology prediction
- **Output**: Annotated image + JSON report
- **Interface**: Gradio UI for clinicians
- **API**: FastAPI backend for programmatic access
- **Deployment**: Docker-ready

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Gradio App
```bash
python app.py
```

### 3. Run the API Server (optional)
```bash
uvicorn api:app --reload
```

### 4. Docker Run
```bash
docker build -t med-ai .
docker run -p 7860:7860 -p 8000:8000 med-ai
```

---

## 📂 Project Structure
```
medical_ai_diagnostics/
├── app.py                # Gradio interface
├── api.py                # FastAPI API endpoints
├── model_training.ipynb  # Model training notebook
├── utils.py              # Preprocessing & helpers
├── model/                # Saved models (U-Net, ResNet)
├── sample_data/          # Sample test images
├── Dockerfile
├── requirements.txt
├── README.md
├── model_card.md
```

---

## 📊 Sample Input/Output
**Input**: Chest X-ray (sample_data/chest_xray_sample.png)

**Output JSON**:
```json
{
  "PatientID": "Sample_001",
  "ScanType": "Chest-Xray",
  "Findings": [
    {
      "Pathology": "Pneumonia",
      "Confidence": 0.92,
      "Location": "Lung region",
      "BoundingBox": [50, 50, 180, 180],
      "Volume_cm3": 0.0
    }
  ],
  "ReportSummary": "Pneumonia detected with 92% confidence.",
  "ComparativeAnalysis": "No prior scans available."
}
```

---

## 📚 Dataset Sources
- NIH ChestX-ray14: https://www.kaggle.com/nih-chest-xrays
- BraTS MRI: https://www.med.upenn.edu/cbica/brats2021/data.html

---

## 📌 Assumptions
- 2D slices only (no 3D volume processing)
- Non-contrast MRI/X-ray input
- Models trained on public sample datasets

---

## 🛡️ Security & Compliance
- DICOM anonymization (if enabled)
- HTTPS recommended for deployment

---

## 🤝 Contributing
Pull requests and feedback are welcome. Let’s improve AI in healthcare together!
