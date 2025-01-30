# utils.py

import os
import re
import cv2
import joblib

def load_model_and_vectorizer():
    knn = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return knn, vectorizer

def clean_caption(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

def extract_features(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at {image_path}")

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image from {image_path}")

    img = cv2.resize(img, (32, 32))  # Resize to 32x32 pixels to match training
    return img.flatten()

