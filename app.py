import os
import cv2
import base64
import numpy as np
from flask import Flask, render_template, request, jsonify
from link_app import analyze_and_save

app = Flask(__name__)
UPLOAD_DIR = "datasets/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def read_bgr_from_file_storage(file_storage):
    file_bytes = file_storage.read()
    npimg = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    return img

def read_bgr_from_base64(data_url):
    header, encoded = data_url.split(",", 1)
    data = base64.b64decode(encoded)
    npimg = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    return img

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    save_collected = request.form.get("save_collected", "false").lower() == "true"
    img_bgr = None
    if 'image' in request.files:
        img_bgr = read_bgr_from_file_storage(request.files['image'])
    elif request.json and request.json.get("imageBase64"):
        img_bgr = read_bgr_from_base64(request.json.get("imageBase64"))
    elif request.form.get("imageBase64"):
        img_bgr = read_bgr_from_base64(request.form.get("imageBase64"))
    else:
        return jsonify({"error": "No image provided"}), 400

    file_name = f"upload_{int(cv2.getTickCount())}.jpg"
    cv2.imwrite(os.path.join(UPLOAD_DIR, file_name), img_bgr)

    results, saved_paths = analyze_and_save(img_bgr, save_collected=save_collected)
    out = []
    for r in results:
        out.append({
            "box": r["box"],
            "emotion": r["emotion"],
            "score": round(r["score"], 4)
        })
    return jsonify({"predictions": out, "saved": saved_paths})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
