from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from flask_cors import CORS
import base64

# Initialize app
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests (e.g. from your frontend)

# Load the face detector and emotion model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = load_model('emotion_model_49epochs.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Optional home route
@app.route('/')
def index():
    return render_template('index.html')


# Main API route for emotion detection
@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    try:
        data = request.json
        image_data = data.get('image')

        if not image_data:
            return jsonify({'error': 'No image provided'}), 400

        # Decode base64 image
        image_data = base64.b64decode(image_data.split(',')[1])
        np_arr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) == 0:
            return jsonify({'emotion': 'No Face Detected'})

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_resized = cv2.resize(roi_gray, (48, 48))
            roi_normalized = roi_resized / 255.0
            roi_input = np.expand_dims(roi_normalized, axis=0)
            roi_input = np.expand_dims(roi_input, axis=-1)

            prediction = model.predict(roi_input)
            label = emotion_labels[np.argmax(prediction)]

            return jsonify({'emotion': label})

        return jsonify({'emotion': 'Error processing face'}), 500

    except Exception as e:
        print('Error:', str(e))
        return jsonify({'error': 'Internal server error'}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
