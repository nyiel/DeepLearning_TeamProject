from PIL import Image
import numpy as np
import tensorflow as tf

def preprocess_image_for_model(image_path, target_size=(224,224)):
    """
    Load image from file path and convert to model input tensor.
    - target_size: (width, height) used by the model (e.g. 224x224)
    Returns batch tensor shape (1, H, W, C) with pixel values scaled as needed.
    Adjust normalization depending on how you trained the model.
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size, Image.BICUBIC)
    arr = np.array(img).astype("float32") / 255.0   # assume model trained with values in [0,1]
    # Add batch dimension
    batch = np.expand_dims(arr, axis=0)
    return batch
