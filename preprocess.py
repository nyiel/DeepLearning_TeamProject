from PIL import Image
import numpy as np

TARGET_SIZE = (224, 224)

def load_and_preprocess(image_path, target_size=TARGET_SIZE):
    """
    Loads an image, resizes, normalizes, and expands dimensions for model prediction.
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    arr = np.array(img).astype('float32') / 255.0
    return arr[np.newaxis, ...]  # add batch dimension
