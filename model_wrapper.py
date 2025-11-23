import os
import json
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from preprocess import load_and_preprocess
from gradcam import GradCAM, save_gradcam_on_image

class ModelWrapper:
    def __init__(self, model_path, labels_path):
        import keras
        keras.config.enable_unsafe_deserialization()  # Required for Lambda layers in Keras 3

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.model = load_model(model_path, safe_mode=False)  # Load .h5 with Lambda layers

        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels file not found: {labels_path}")
        with open(labels_path, 'r', encoding='utf-8') as f:
            self.labels = json.load(f)

        # Identify last conv layer automatically for Grad-CAM
        last_conv_layer_name = None
        for layer in reversed(self.model.layers):
            if 'conv' in layer.name:
                last_conv_layer_name = layer.name
                break
        if last_conv_layer_name is None:
            raise ValueError("No convolutional layer found for Grad-CAM")
        self.gradcam = GradCAM(self.model, last_conv_layer_name)

    def predict_with_explanation(self, image_path, save_dir='static/uploads'):
        # Preprocess and batch the image
        x = load_and_preprocess(image_path)  # Should return shape (1, H, W, 3)
        if x.ndim == 3:
            x = np.expand_dims(x, axis=0)

        # Run prediction
        preds = self.model.predict(x)[0]
        top_idx = preds.argsort()[-3:][::-1]
        top_probs = preds[top_idx]

        # Generate Grad-CAM automatically
        cam_path = None
        try:
            heatmap = self.gradcam.compute_heatmap(image_path, int(top_idx[0]))
            cam_filename = os.path.basename(image_path).rsplit('.', 1)[0] + '_gradcam.jpg'
            cam_path = os.path.join(save_dir, cam_filename)
            save_gradcam_on_image(image_path, heatmap, cam_path, alpha=0.4)
        except Exception as e:
            print("⚠ Grad-CAM failed:", e)
            cam_path = None

        return top_probs.tolist(), top_idx.tolist(), cam_path
