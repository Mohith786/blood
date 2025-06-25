from flask import Flask, render_template, jsonify
import subprocess
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time
import base64
import os

app = Flask(__name__)
model = load_model(r"D:\blood_detection\fingerprint_blood_group_model_10.h5")
blood_groups = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-','O+', 'O-']

cpp_exe_path = r"D:\blood_detection\Secugen\SecugenConnect\x64\Debug\SecugenConnect.exe"
captured_image_path = r"D:\blood_detection\Secugen\SecugenConnect\fingerprint\fingerprint1.bmp"

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/predict')
def predict():
    try:
        subprocess.run([cpp_exe_path], timeout=10)
        time.sleep(1)

        if not os.path.exists(captured_image_path):
            return jsonify({'status': "Image not found", 'prediction': None, 'image_data': None})

        image = cv2.imread(captured_image_path, cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({'status': "Error reading image", 'prediction': None, 'image_data': None})

        resized_image = cv2.resize(image, (224, 224))
        processed_image = resized_image / 255.0
        processed_image = np.expand_dims(processed_image, axis=0)

        prediction = model.predict(processed_image)
        predicted_label = blood_groups[np.argmax(prediction)]

        _, buffer = cv2.imencode('.bmp', image)
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        image_data = f"data:image/bmp;base64,{encoded_image}"

        return jsonify({
            'status': "Fingerprint captured!",
            'prediction': predicted_label,
            'image_data': image_data
        })
    except Exception as e:
        return jsonify({'status': f"Error: {str(e)}", 'prediction': None, 'image_data': None})

if __name__ == '__main__':
    app.run(debug=True)
