import os
import cv2
from datetime import datetime
from face_emotions import predict_emotions_from_image

COLLECTED_DIR = "datasets/collected"
os.makedirs(COLLECTED_DIR, exist_ok=True)

def save_image_to_label(img_bgr, label):
    label_dir = os.path.join(COLLECTED_DIR, label)
    os.makedirs(label_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(label_dir, f"{label}_{ts}.jpg")
    cv2.imwrite(path, img_bgr)
    return path

def analyze_and_save(image_bgr, save_collected=False):
    results = predict_emotions_from_image(image_bgr)
    saved_paths = []
    for r in results:
        if save_collected:
            saved_paths.append(save_image_to_label(image_bgr, r['emotion']))
    return results, saved_paths
