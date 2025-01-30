# app.py

import os
import sys

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("Current working directory:", os.getcwd())  # Add this line to check the current directory

from flask import Flask, request, jsonify
from utils import load_model_and_vectorizer, extract_features

# Initialize the Flask app
app = Flask(__name__)

# Load model and vectorizer
knn, vectorizer = load_model_and_vectorizer()

# Define the API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    img_path = data['img_path']
    features = extract_features(img_path).reshape(1, -1)
    pred_caption = knn.predict(features)
    pred_caption_words = vectorizer.inverse_transform(pred_caption)
    return jsonify({'caption': ' '.join(pred_caption_words[0])})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)



