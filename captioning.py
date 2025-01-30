import os
import pandas as pd
import re
import cv2
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# List top-level files and directories within the specified directory
def list_top_level(directory):
    for name in os.listdir(directory):
        print(name)

# Call the function to list top-level contents
directory = "C:/Users/B.Thabana/Downloads/archive (42)"
print("Listing top-level files and directories within 'archive (42)':")
list_top_level(directory)

# Load and preprocess captions
def load_captions(file_path):
    captions_dict = {}
    with open(file_path, 'r') as f:
        next(f)  # Skip the header line
        for line in f:
            parts = line.strip().split(",", 1)  # Split only on the first comma
            img_name = parts[0]  # Extract image filename
            caption = clean_caption(parts[1])
            if img_name not in captions_dict:
                captions_dict[img_name] = []
            captions_dict[img_name].append(caption)
    return captions_dict

# Clean captions (lowercase, remove special characters)
def clean_caption(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    return text

# Extract features from images
def extract_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))  # Resize image
    hog = cv2.HOGDescriptor()
    features = hog.compute(img)
    return features.flatten()

# Updated file paths
captions_dict = load_captions("C:/Users/B.Thabana/Downloads/archive (42)/captions.txt")
image_folder = "C:/Users/B.Thabana/Downloads/archive (42)/Images"
print("Loaded captions and image folder paths.")

# Prepare data
image_features = []
captions = []
for img_name in os.listdir(image_folder):
    img_path = os.path.join(image_folder, img_name)
    features = extract_features(img_path)
    for caption in captions_dict.get(img_name, []):
        image_features.append(features)
        captions.append(caption)
print("Prepared data and extracted features from images.")
print("Captions:", captions[:5])  # Print first 5 captions for inspection

# Convert captions to bag-of-words representation
vectorizer = CountVectorizer(max_features=1000, stop_words=None)
X_captions = vectorizer.fit_transform(captions).toarray()
print("Converted captions to bag-of-words representation.")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(image_features, X_captions, test_size=0.2, random_state=42)
print("Split data into training and testing sets.")

# Train model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
print("Trained k-NN model.")

# Predict and evaluate
y_pred = knn.predict(X_test)
accuracy = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
print(f"Accuracy: {accuracy}")

# Example prediction
test_img_path = "C:/Users/B.Thabana/Downloads/archive (42)/Images/1000268201_693b08cb0e.jpg"
test_features = extract_features(test_img_path).reshape(1, -1)
pred_caption = knn.predict(test_features)
pred_caption_words = vectorizer.inverse_transform(pred_caption)
print(f"Generated Caption: {' '.join(pred_caption_words[0])}")













