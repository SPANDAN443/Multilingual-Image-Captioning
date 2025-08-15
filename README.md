# 🌐 Multilingual Image Captioning

A deep learning-based application that automatically generates captions for images in multiple languages. This project leverages state-of-the-art image understanding models and translation pipelines to make image content accessible to a global audience.

## 🚀 Features

- 🖼️ Image caption generation using transformer-based models (e.g., BLIP, ViT-GPT2)
- 🌍 Supports multiple languages (English, Hindi, etc.)
- 🌐 Integrates translation using models like MarianMT or Google Translate API
- ⚡ Streamlit web interface for easy usage
- 📷 Upload your image and get captions instantly!

## 🛠️ Technologies Used

- Python
- Transformers (HuggingFace)
- Streamlit
- Torch / TensorFlow (depending on model)
- OpenCV / PIL for image preprocessing
- Google Translate / MarianMT for multilingual output

## 🧠 Model Pipeline

1. **Image Encoder**: Extract features from the input image.
2. **Caption Generator**: Generate a descriptive caption in English.
3. **Translator**: Translate the English caption to other languages like Hindi, Spanish, etc.

## 📦 Setup Instructions

```bash
git clone https://github.com/Hemanth170420/Multilingual-Image-Captioning.git
cd Multilingual-Image-Captioning
python -m venv venv
venv\Scripts\activate      # On Windows
pip install -r requirements.txt
python run app.py
