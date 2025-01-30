import pandas as pd
import re
import numpy as np
import os
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input

# Load and preprocess captions
def load_captions(file_path):
    captions_dict = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split("\t")
            img_name = parts[0].split("#")[0]  # Extract image filename
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

# Updated file path for captions
captions_dict = load_captions("C:/Users/B.Thabana/Downloads/archive (42)/Flickr8k_text/Flickr8k.token.txt")

# Example output
print("Example Captions for an Image:", captions_dict["1000268201_693b08cb0e.jpg"])

# Load InceptionV3 Model (removing the last layer)
base_model = InceptionV3(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

# Extract features from a single image
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(299, 299))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    features = model.predict(img)
    return features

# Updated folder path for images
image_folder = "C:/Users/B.Thabana/Downloads/archive (42)/Flickr8k_Dataset"

# Extract features for all images
features_dict = {}

for img_name in os.listdir(image_folder):
    img_path = os.path.join(image_folder, img_name)
    features_dict[img_name] = extract_features(img_path, model)

# Save extracted features to a file
np.save("image_features.npy", features_dict)

