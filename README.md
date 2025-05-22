# 🫁 LungScan: AI-Powered Chest X-Ray Analysis

LungScan is a medical imaging tool built using FastAPI that classifies chest X-ray images using a **ResNet-18** model and performs mock segmentation using a placeholder **U-Net** model.

> ⚠️ This project is for educational purposes only. It should not be used for real clinical diagnosis.

---

## 🧠 Features

- 🧪 Classifies chest X-rays as **Normal** or **Pneumonia**
- 🧩 Includes dummy segmentation (U-Net based placeholder)
- 📤 Upload image via API or frontend UI
- 📊 Visualization of prediction and segmented output
- 🚀 FastAPI backend with auto-generated Swagger docs

---

## 🗂 Project Structure

LungScan/
├── api/ # FastAPI endpoints
├── models/ # Model training & loading code
├── sample_data/ # Dataset (ignored in Git)
├── static/ # Web assets or visual outputs
├── templates/ # HTML templates for frontend (if any)
├── unet_dummy.py # Dummy segmentation logic
├── train_resnet.py # ResNet-18 training script
├── app.py # Main FastAPI app
├── requirements.txt # Dependencies
└── README.md # This file

---

## 🚀 Getting Started

### ✅ Prerequisites

- Python 3.10+
- pip

### 📦 Install Dependencies

```bash
pip install -r requirements.txt
▶️ Run the App
bash
Copy
Edit
uvicorn app:app --reload
Visit: http://localhost:8000/docs to use the interactive Swagger API.

🏋️‍♂️ Model Training
ResNet Classifier
bash
Copy
Edit
python train_resnet.py
This will train and save a ResNet-18 model for classification on your local dataset.

Dummy U-Net
A dummy U-Net model is implemented in unet_dummy.py as a placeholder for future segmentation tasks.

💾 Dataset
Chest X-ray dataset is stored locally in [sample_data/ folder.](https://drive.google.com/file/d/1OmBzt3X_u_Sd3pZuuVvpzWHeb78E4Xgn/view?usp=drive_link)

Due to size limits, this folder is excluded from the repository using .gitignore.

💡 You can download the dataset from:
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

🛡 Disclaimer
This project is not a certified diagnostic tool and should not be used for real-world clinical decision making.

🙋‍♀️ Author
Kavya Gupta
📧 kavya1b1@gmail.com
🔗 LinkedIn | GitHub

⭐️ Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

