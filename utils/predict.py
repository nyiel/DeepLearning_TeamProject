import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from ..preprocess import load_and_preprocess
from ..gradcam import GradCAM


class ModelWrapper:
    def __init__(self, model_dir, labels_path):
        # load model
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        # Try to load a SavedModel or .h5 / .keras
        try:
            self.model = load_model(model_dir)
        except Exception:
            # try loading specific file
            possible = [f for f in os.listdir(model_dir)
                        if f.endswith('.h5') or f.endswith('.keras')]
            if possible:
                self.model = load_model(os.path.join(model_dir, possible[0]))
            else:
                raise

        # labels
        with open(labels_path, 'r', encoding='utf-8') as f:
            self.labels = json.load(f)

        # create gradcam helper
        self.gradcam = GradCAM(self.model)

    def predict_with_explanation(self, image_path, save_dir='static/uploads'):
        x = load_and_preprocess(image_path)
        preds = self.model.predict(x)[0]

        # get top-3 indices
        top_idx = preds.argsort()[-3:][::-1]
        top_probs = preds[top_idx]

        # create gradcam overlay for top-1
        cam_path = None
        try:
            cam = self.gradcam.compute_heatmap(
                image_path, class_index=int(top_idx[0])
            )

            import cv2
            orig = cv2.imread(image_path)
            heatmap = cv2.resize(cam, (orig.shape[1], orig.shape[0]))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            overlay = cv2.addWeighted(orig, 0.6, heatmap, 0.4, 0)

            cam_filename = (
                os.path.basename(image_path).rsplit('.', 1)[0] + '_gradcam.jpg'
            )
            cam_path = os.path.join(save_dir, cam_filename)

            cv2.imwrite(cam_path, overlay)

        except Exception as e:
            cam_path = image_path  # fallback to original

        return top_probs.tolist(), top_idx.tolist(), cam_path

