import os
import json
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from preprocess import load_and_preprocess
from gradcam import GradCAM

class ModelWrapper:
    def __init__(self, model_path, labels_path):
        from tensorflow.keras.models import load_model

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.model = load_model(model_path)

        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels file not found: {labels_path}")
        import json
        with open(labels_path, 'r', encoding='utf-8') as f:
            self.labels = json.load(f)

        self.gradcam = GradCAM(self.model)

    def predict_with_explanation(self, image_path, save_dir='static/uploads'):
        from preprocess import load_and_preprocess
        x = load_and_preprocess(image_path)
        preds = self.model.predict(x)[0]

        top_idx = preds.argsort()[-3:][::-1]
        top_probs = preds[top_idx]

        # Grad-CAM
        cam_path = None
        try:
            heatmap = self.gradcam.compute_heatmap(image_path, int(top_idx[0]))
            cam_filename = os.path.basename(image_path).rsplit('.', 1)[0] + '_gradcam.jpg'
            cam_path = os.path.join(save_dir, cam_filename)
            save_gradcam_on_image(image_path, heatmap, cam_path)
        except Exception as e:
            print("⚠ Grad-CAM failed:", e)
            cam_path = None

        return top_probs.tolist(), top_idx.tolist(), cam_path
