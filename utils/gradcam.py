import numpy as np
import cv2
from tensorflow import keras
from PIL import Image

def make_gradcam_heatmap(image_path, model, last_conv_layer_name, pred_index=None):
    """
    Produce a Grad-CAM heatmap for the top predicted class (or pred_index if provided).
    - image_path: path to image file
    - model: Keras model
    - last_conv_layer_name: name of last conv layer in model
    Returns: 2D heatmap (float32) resized to original image size range [0,1]
    """
    # Preprocess single image - replicate the same preprocessing used for training
    from utils.preprocess import preprocess_image_for_model
    img_array = preprocess_image_for_model(image_path)  # returns shape (1,h,w,3)
    img_tensor = img_array

    grad_model = keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])

    with keras.backend.gradient_tape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        if pred_index is None:
            pred_index = np.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    # Compute pooled gradients
    pooled_grads = keras.backend.mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]  # shape h x w x channels
    pooled_grads = pooled_grads.numpy()
    conv_outputs = conv_outputs.numpy()

    # Weight the channels by the corresponding gradients
    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    # Compute heatmap as mean of the weighted feature maps
    heatmap = np.mean(conv_outputs, axis=-1)

    # Relu and normalize
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) == 0:
        return np.zeros_like(heatmap)
    heatmap /= np.max(heatmap)

    # Resize heatmap to original image size
    img = Image.open(image_path).convert("RGB")
    heatmap = cv2.resize(heatmap, (img.width, img.height))
    return heatmap

def save_gradcam_overlay(original_image_path, heatmap, output_path, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Save the Grad-CAM overlay image to output_path.
    - heatmap: 2D array values [0,1] sized to original image
    """
    img = cv2.imread(original_image_path)
    if img is None:
        raise FileNotFoundError(original_image_path)

    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    # convert BGR to RGB if needed for blending correctness; OpenCV uses BGR
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)
    # Save overlay (BGR)
    cv2.imwrite(output_path, overlay)
    return output_path
