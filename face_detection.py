from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from flask_cors import CORS
import base64
import requests
from dotenv import load_dotenv
import os
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

app = Flask(__name__)
CORS(app) 

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = load_model('emotion_model_49epochs.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    try:
        data = request.json
        image_data = data.get('image')

        if not image_data:
            return jsonify({'emotion': 'No Image Provided'}), 400

        image_data = base64.b64decode(image_data.split(',')[1])
        np_arr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

        if len(faces) == 0:
            return jsonify({'emotion': 'No Face Detected'})

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_resized = cv2.resize(roi_gray, (48, 48))
            roi_normalized = roi_resized / 255.0
            roi_input = np.expand_dims(roi_normalized, axis=0)
            roi_input = np.expand_dims(roi_input, axis=-1)

            prediction = model.predict(roi_input)
            predicted_emotion = emotion_labels[np.argmax(prediction)]

            return jsonify({'emotion': predicted_emotion})

        return jsonify({'emotion': 'Face Processing Error'}), 500

    except Exception as e:
        print('Error:', str(e))
        return jsonify({'emotion': 'Internal Error'}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json.get("message")
        if not user_message:
            return jsonify({'reply': "Message was empty."}), 400

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:5000", 
            "X-Title": "Emotion Assistant"
        }

        data = {
            "model": "mistralai/mistral-7b-instruct", 
            "messages": [
                {"role": "system", "content": "You are a friendly assistant that replies helpfully based on user emotions."},
                {"role": "user", "content": user_message}
            ]
        }


        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)

        if response.status_code != 200:
            return jsonify({'reply': f"⚠ Error from OpenRouter: {response.text}"}), 500

        reply = response.json()['choices'][0]['message']['content']
        return jsonify({'reply': reply})

    except Exception as e:
        return jsonify({'reply': f"❌ Server Error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)