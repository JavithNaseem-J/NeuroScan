# NeuroScan: 

## CNN-Based Brain Tumor Classification System

## 🧠 Project Overview

NeuroScan is an end-to-end machine learning system for automated brain tumor classification using Convolutional Neural Networks (CNNs). Built on the Kaggle Brain Tumor MRI dataset, it classifies MRI scans into four categories: Glioma, Meningioma, Pituitary Tumor, and No Tumor. The project emphasizes high model accuracy, modular architecture, and production-readiness.

## ✨ Key Features

* CNN-powered brain tumor classification
* **Test Accuracy**: 98.32%
* Modular ML pipeline (ingestion → cleaning → transformation → training → evaluation)
* DVC for reproducible data versioning
* MLflow-based training and evaluation logs
* Streamlit web app for live inference
* Dockerized deployment
* Poetry for dependency management

## 🔬 Model Performance

* **Test Accuracy**: 98.32%
* **F1 Score**: 98.32%
* **Precision**: 98.34%
* **Recall**: 98.32%
* **Test Loss**: 0.0567

## ⚒️ Technical Architecture

**1. Data Ingestion**

* Automatically downloads MRI dataset using Kaggle API
* Extracts and stores structured data for downstream stages

**2. Data Cleaning**

* Crops irrelevant brain scan areas
* Resizes images to 240x240
* Validates class directories

**3. Data Transformation**

* Applies augmentations (rotation, shift, flip)
* Creates and caches train/val/test generators
* Saves numpy datasets for reuse

**4. Model Training**

* Uses EfficientNetB1 pretrained on ImageNet
* Includes dropout, global pooling, and softmax head
* Tracks metrics and artifacts in MLflow
* Supports early stopping and LR reduction callbacks

**5. Model Evaluation**

* Uses TF dataset for batch-wise inference
* Generates classification report and confusion matrix
* Logs all metrics to MLflow

**6. Web Inference App**

* Built with Streamlit
* Accepts brain MRI images for real-time predictions
* Shows original, cropped, and predicted output with confidence

**7. Docker Support**

* Dockerfile provided for easy containerization and deployment

## 🚀 Quick Start

### Prerequisites

* Python 3.9
* Poetry
* Docker (optional)
* Kaggle API key

### Setup

```bash
# Clone the repo
git clone https://github.com/yourusername/NeuroScan.git
cd NeuroScan

# Install dependencies
poetry install
poetry shell
```

### Configure Credentials

Edit `.env` or set Kaggle API key directly:

```env
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
```

## 📂 Configuration

Edit:

* `config_file/config.yaml` for general pipeline configs
* `config_file/params.yaml` for model-specific parameters

## 📊 Run Pipelines

```bash
# Run full pipeline
python main.py

# Individual pipeline modules available via src/NeuroScan/pipelines
```

## 📈 Streamlit App

```bash
streamlit run app.py
```

Uploads an image and displays:

* Original scan
* Cropped input
* Predicted class and probabilities

## 🧰 Docker Deployment

```bash
# Build image
docker build -t neuroscan .

# Run container
docker run -p 8501:8501 neuroscan
```

## 📓 Project Structure

```
NeuroScan/
├── app.py                    # Streamlit web app
├── config_file/             # YAML config files
├── Dockerfile               # Docker build config
├── dvc.yaml                 # DVC pipeline stages
├── notebooks/               # Jupyter analysis notebooks
├── src/NeuroScan/           # All core modules
│   ├── components/          # data + model modules
│   ├── pipelines/           # feature/model pipelines
│   ├── config_entity/       # Pydantic config schemas
│   ├── constants/           # File paths/constants
│   ├── config/              # Config readers
│   └── utils/               # Logging, helpers
```

## 📄 License

This is Under MIT license.

---

### ⚠️ Important

**This tool is for educational and experimental purposes only.**
Predictions should never be used for actual medical decision-making without proper clinical validation.
