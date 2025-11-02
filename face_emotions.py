import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

MODEL_PATH = "models/emotion_model.h5"
HAAR_PATH = "haarcascades/haarcascade_frontalface_default.xml"

# If the model file is missing, we will raise when trying to import.
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Place a trained model there.")

model = load_model(MODEL_PATH)
face_detector = cv2.CascadeClassifier(HAAR_PATH)

LABELS_PATH = "models/labels.txt"
if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, "r") as f:
        labels = [l.strip() for l in f.readlines() if l.strip()]
else:
    labels = ['angry', 'happy', 'neutral', 'sad']

def predict_emotions_from_image(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
    results = []
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (48, 48))
        face_arr = face_resized.astype("float") / 255.0
        face_arr = img_to_array(face_arr)
        face_arr = np.expand_dims(face_arr, axis=0)
        preds = model.predict(face_arr)
        top_idx = int(np.argmax(preds))
        score = float(preds[0][top_idx])
        label = labels[top_idx] if top_idx < len(labels) else str(top_idx)
        results.append({
            "box": (int(x), int(y), int(w), int(h)),
            "emotion": label,
            "score": score
        })
    return results
