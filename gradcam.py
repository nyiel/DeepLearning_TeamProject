import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import cv2
import os

class GradCAM:
    def __init__(self, model, last_conv_layer_name=None):
        self.model = model
        if last_conv_layer_name is None:
            self.last_conv_layer_name = self._find_last_conv_layer()
        else:
            self.last_conv_layer_name = last_conv_layer_name

    def _find_last_conv_layer(self):
        # Find last layer with 4D output
        for layer in reversed(self.model.layers):
            try:
                if len(layer.output.shape) == 4:
                    return layer.name
            except:
                continue
        raise ValueError("No Conv2D layer found. Model not suitable for Grad-CAM.")

    def compute_heatmap(self, img_path, class_index, target_size=(224, 224)):
        # Load and preprocess image
        img = image.load_img(img_path, target_size=target_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0) / 255.0

        # Grad-CAM model: outputs conv layer + predictions
        grad_model = tf.keras.models.Model(
            inputs=self.model.inputs,
            outputs=[self.model.get_layer(self.last_conv_layer_name).output, self.model.output]
        )

        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(x)
            loss = preds[:, class_index]

        # Compute gradients
        grads = tape.gradient(loss, conv_out)
        weights = tf.reduce_mean(grads, axis=(0, 1, 2))

        cam = tf.reduce_sum(tf.multiply(weights, conv_out[0]), axis=-1)
        cam = np.maximum(cam, 0)  # ReLU
        cam = cam / (np.max(cam) + 1e-8)  # Normalize

        return cam.numpy()

def save_gradcam_on_image(img_path, heatmap, save_path, alpha=0.5):
    # Read original image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize heatmap to match image
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)

    # Save overlay
    cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return save_path
