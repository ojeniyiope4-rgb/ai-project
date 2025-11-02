Emotion Recognition - Flask Web App

This package includes a complete Flask web app plus a placeholder Keras model.
IMPORTANT: The included model is a small placeholder and is NOT high-accuracy.
You asked for a 'high-accuracy' model (~100MB). I cannot fetch large pretrained models
directly in this environment. To use a high-accuracy model, either:

1) Replace models/emotion_model.h5 with a pretrained model you obtain (recommended).
   Example sources:
     - public repositories hosting FER/AffectNet pretrained weights
     - train with transfer learning on a large dataset (FER2013, AffectNet)

2) Train the model locally using the provided model_training.py script:
     - Put a labeled dataset under datasets/train/<label>/
     - Run: python model_training.py
   This will save a trained model to models/emotion_model.h5

Notes:
 - Place OpenCV's haarcascade_frontalface_default.xml in the haarcascades/ folder.
 - To run the app:
     pip install -r requirements.txt
     python app.py
 - If you want, I can provide instructions or a script to download a known high-quality
   pretrained model from a URL you trust, or guide you through training with transfer learning.

Could not create Keras model due to error: No module named 'tensorflow'\nAn empty placeholder file was created instead.

