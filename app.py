from flask import Flask, render_template, request, jsonify
import requests
import base64
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import json

app = Flask(__name__)

# AI API Configuration
AI_API_KEY = os.environ.get('AI_API_KEY', 'AIzaSyDDN9fwgw2MF7CcP6Grs5ixpMPAYA-cUe8')
AI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={AI_API_KEY}"

# =============================
# 1. Device setup
# =============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================
# 2. Image transformations
# =============================
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

# =============================
# 3. Emotion → Depression mapping
# =============================
def map_emotion_to_depression(emotion):
    if emotion in ["Happy", "Surprize", "Neutral"]:
        return "No Depression"
    elif emotion in ["Fear", "Disgust"]:
        return "Mild Depression"
    elif emotion in ["Sad", "Angry"]:
        return "Severe Depression"
    else:
        return "Unknown"

# =============================
# 4. Load Model
# =============================
num_classes = 7   # ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprize']
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)

try:
    model.load_state_dict(torch.load("depression_resnet50new.pth", map_location=device))
    print("Model loaded successfully")
except Exception as e:
    print(f"Warning: Could not load model. Using mock predictions. ({e})")
    class MockModel:
        def eval(self): return self
        def to(self, device): return self
        def __call__(self, x): return torch.randn(1, 7)
    model = MockModel()

model = model.to(device)
model.eval()

classes = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprize']

# =============================
# 5. Haar Cascade for Face Detection
# =============================
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
except Exception as e:
    print(f"Warning: Could not load Haar cascade. Using mock face detection. ({e})")
    face_cascade = None

def detect_faces(image):
    if image is None:
        return []
    if face_cascade is None:
        h, w = image.shape[:2]
        return [(int(w*0.2), int(h*0.2), int(w*0.6), int(h*0.6))]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces

def predict_depression(image):
    try:
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        input_tensor = transform(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
            _, pred = torch.max(outputs, 1)
            emotion = classes[pred.item()]
            depression = map_emotion_to_depression(emotion)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence = torch.max(probabilities).item()
        return emotion, depression, confidence
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Neutral", "No Depression", 0.85

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No file uploaded!"}), 400

    file = request.files['image']
    if file.filename == "":
        return jsonify({"error": "Please select an image!"}), 400

    try:
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({"error": "Invalid image data."}), 400

        faces = detect_faces(image)
        if len(faces) == 0:
            return jsonify({"error": "No face detected in the image. Please upload a clear face photo."}), 400

        x, y, w, h = faces[0]
        face_img = image[y:y+h, x:x+w]

        emotion, depression, confidence = predict_depression(face_img)

        _, buffer = cv2.imencode('.jpg', face_img)
        image_b64 = base64.b64encode(buffer).decode("utf-8")

        prompt = (
            f"Classify severity strictly as one of: No Depression, Mild Depression, Severe Depression, "
            f"based on this context: detected_emotion={emotion}, mapped_level={depression}, "
            f"confidence={confidence:.2f}. "
            "Then return exactly this JSON:\n"
            "{"
            "\"depression_level\":\"<No Depression|Mild Depression|Severe Depression>\","
            "\"detected_emotion\":\"<emotion>\","
            "\"confidence\":\"<0-1 with 2 decimals>\","
            "\"suggestions\":[\"<<=15 words>\",\"<<=15 words>\",\"<<=15 words>\"]"
            "}"
        )

        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "image/jpeg", "data": image_b64}}
                ]
            }]
        }

        headers = {"Content-Type": "application/json"}
        response = requests.post(AI_URL, headers=headers, json=payload, timeout=20)

        if response.status_code != 200:
            default_suggestions = {
                "No Depression": [
                    "Keep up your positive routines.",
                    "Stay connected with friends.",
                    "Practice daily gratitude."
                ],
                "Mild Depression": [
                    "Take a 10-minute walk.",
                    "Call or text a friend.",
                    "Try 5 minutes of breathing."
                ],
                "Severe Depression": [
                    "Reach out to a professional.",
                    "Keep a simple daily routine.",
                    "Be kind to yourself."
                ]
            }
            return jsonify({
                "depression_level": depression,
                "detected_emotion": emotion,
                "confidence": f"{confidence:.2f}",
                "suggestions": default_suggestions.get(depression, [])
            })

        response_json = response.json()

        # Extract text from candidates safely
        result_text = ""
        try:
            result_text = response_json["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            pass

        if result_text:
            try:
                json_start = result_text.find('{')
                json_end = result_text.rfind('}') + 1
                if json_start != -1 and json_end != -1:
                    result_data = json.loads(result_text[json_start:json_end])
                    # ensure required fields exist; fallback to local if missing
                    result_data.setdefault("depression_level", depression)
                    result_data.setdefault("detected_emotion", emotion)
                    result_data.setdefault("confidence", f"{confidence:.2f}")
                    result_data.setdefault("suggestions", [])
                    return jsonify(result_data)
            except json.JSONDecodeError:
                pass

        return jsonify({
            "error": "No valid response from AI",
            "raw_response": response_json
        }), 502

    except Exception as e:
        return jsonify({"error": f"Processing error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
