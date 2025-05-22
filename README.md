🩺 LungScan - AI-Powered Chest X-ray Analysis
LungScan is an advanced medical imaging API built with FastAPI and AI models to diagnose lung conditions like Pneumonia & Tuberculosis from chest X-rays.

🔗 Live GitHub Repo: LungScan

🚀 Features
✅ AI-powered pathology classification (Normal, Pneumonia, Tuberculosis) ✅ Segmentation masks for lung abnormalities ✅ Built with FastAPI, PyTorch, and React for a seamless experience ✅ Fully customizable API with real-time predictions

📂 Project Structure
LungScan/
│── sample_data/  # Contains dataset (excluded from GitHub)
│── backend/      # FastAPI-based AI model & inference logic
│── frontend/     # React UI for image uploads & results display
│── utils.py      # Preprocessing, segmentation, and classification functions
│── app.py        # Main FastAPI application
│── README.md     # You’re reading this!
📊 Dataset
Since the dataset contains thousands of images, it is hosted externally. 🔗 Download Dataset: [https://drive.google.com/file/d/1OmBzt3X_u_Sd3pZuuVvpzWHeb78E4Xgn/view?usp=drive_link]

To use it in your project, place the downloaded files inside:

LungScan/sample_data/chest_xray/
💻 Installation & Setup
🔹 Step 1: Clone Repo & Install Dependencies
bash
git clone https://github.com/kavya1b1/LungScan.git
cd LungScan
pip install -r requirements.txt
🔹 Step 2: Run FastAPI Backend
bash
uvicorn app:app --reload
🔗 Visit Swagger UI: http://127.0.0.1:8000/docs

npm install
npm start

🛠 API Endpoints
Endpoint	Method	Description
/upload_image	POST	Upload X-ray & get classification + segmentation mask
/predict	GET	Run AI pathology analysis
/docs	GET	Swagger UI for testing API
🤖 AI Model Details
Model: ResNet-based classifier

Trained on chest X-ray datasets

Uses PyTorch for inference

GPU/Apple MPS acceleration supported

📌 Contributors
👤 Kavya – AI Developer 📅 Project Started: May 2025
