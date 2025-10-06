from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
from PIL import Image
import base64
import io
import torch

from model import load_model, predict_mask

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Load the model once when the application starts
predictor = load_model()

@app.route('/api/segment', methods=['POST'])
def segment_image():
    if 'image' not in request.json or 'prompt' not in request.json:
        return jsonify({'error': 'Missing image or prompt in the request'}), 400

    try:
        # Get the image and prompt from the request
        image_data = request.json['image']
        prompt = request.json['prompt']

        # Decode the base64 image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(image)

        # Generate the segmentation mask
        mask = predict_mask(predictor, image_np, prompt)

        # Convert the mask to a base64 string
        mask_pil = Image.fromarray(mask)
        buffered = io.BytesIO()
        mask_pil.save(buffered, format="PNG")
        mask_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return jsonify({'mask': mask_base64})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': 'An error occurred during segmentation'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)