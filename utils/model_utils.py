import os
import numpy as np
import tensorflow as tf
import cv2
import pickle
import gdown
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

# --- SETTINGS ---
MODEL_FOLDER = 'model'
MODEL_PATH = os.path.join(MODEL_FOLDER, 'Best_Model.h5')
INDICES_PATH = os.path.join(MODEL_FOLDER, 'class_indices.pkl')
IMG_SIZE = (224, 224)
NUM_CLASSES = 4  # Mango, Neem, Guava, Lemon

# Google Drive file IDs
MODEL_DRIVE_ID = "1lxu5DPQWuDIHrHPk1fTgxbVNw6JLolD3"  # Replace with your link's ID if changed
INDICES_DRIVE_ID = "PUT_INDICES_FILE_ID_HERE"  # Optional if class_indices.pkl is also on Drive

# Ensure model folder exists
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Download model if missing
if not os.path.exists(MODEL_PATH):
    print("Model not found. Downloading from Google Drive...")
    url = f"https://drive.google.com/uc?id={MODEL_DRIVE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

# Download class indices if missing
if not os.path.exists(INDICES_PATH) and INDICES_DRIVE_ID != "":
    print("Class indices not found. Downloading from Google Drive...")
    url = f"https://drive.google.com/uc?id={INDICES_DRIVE_ID}"
    gdown.download(url, INDICES_PATH, quiet=False)

# --- MODEL LOADING ---
def build_model_architecture():
    print("Reconstructing ResNet50 architecture...")
    base_model = ResNet50(weights=None, include_top=False, input_shape=IMG_SIZE + (3,))
    base_model.trainable = False
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    model.build((None, 224, 224, 3))
    return model

try:
    model = tf.keras.models.load_model(MODEL_PATH)
except ValueError:
    print("Standard load failed. Rebuilding architecture...")
    model = build_model_architecture()
    model.load_weights(MODEL_PATH)
    print("Weights loaded successfully.")

# --- LOAD CLASS LABELS ---
try:
    with open(INDICES_PATH, 'rb') as f:
        indices = pickle.load(f)
        LABELS = {v: k for k, v in indices.items()}
except FileNotFoundError:
    print(f"Class indices file missing at {INDICES_PATH}, using default labels.")
    LABELS = {0: "Unknown_0", 1: "Unknown_1", 2: "Unknown_2", 3: "Unknown_3"}

# --- FUNCTIONS ---
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, IMG_SIZE)
    img_array = np.expand_dims(img_resized, axis=0)
    img_array = preprocess_input(img_array.astype(float))
    return img_array, img

def predict_species(img_path):
    processed_img, _ = preprocess_image(img_path)
    preds = model.predict(processed_img, verbose=0)[0]
    top_indices = preds.argsort()[-3:][::-1]
    top_3 = [(LABELS[i], float(preds[i]) * 100) for i in top_indices if i in LABELS]
    return top_3, processed_img

def generate_gradcam(img_array, save_path):
    try:
        base_model = model.layers[0]
        classifier_layers = model.layers[1:]
        with tf.GradientTape() as tape:
            base_output = base_model(img_array, training=False)
            tape.watch(base_output)
            x = base_output
            for layer in classifier_layers:
                x = layer(x)
            preds = x
            top_class_idx = tf.argmax(preds[0])
            top_class_channel = preds[:, top_class_idx]

        grads = tape.gradient(top_class_channel, base_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        base_output = base_output[0]
        heatmap = base_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.resize(heatmap, IMG_SIZE)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        cv2.imwrite(save_path, heatmap_colored)
        return True
    except Exception as e:
        print(f"Grad-CAM error: {e}")
        return False
