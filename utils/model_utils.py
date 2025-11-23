import os
import numpy as np
import tensorflow as tf
import cv2
import pickle
from tensorflow.keras import layers, models
# --- CHANGED TO RESNET50 ---
from tensorflow.keras.applications import ResNet50 
from tensorflow.keras.applications.resnet50 import preprocess_input 

# SETTINGS
MODEL_PATH = 'model/Best_Model.h5'
INDICES_PATH = 'model/class_indices.pkl'
IMG_SIZE = (224, 224)
NUM_CLASSES = 4 # Mango, Neem, Guava, Lemon (Ensure this matches your training data)

def build_model_architecture():
    """
    Reconstructs the ResNet50 architecture (the model identified in the error traceback).
    This bypasses Keras version conflicts when loading .h5 files by recreating the
    Transfer Learning model structure before attempting to load weights.
    """
    print("Reconstructing ResNet50 Architecture...")
    
    
    # 1. Base Model (Must match the architecture the weights were trained on)
    base_model = ResNet50(weights=None, include_top=False, input_shape=IMG_SIZE + (3,))
    base_model.trainable = False 
    
    # 2. Rebuild the Sequential container (Classifier Head)
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    # 3. Build with input shape to initialize tensor shapes
    model.build((None, 224, 224, 3))
    return model

# --- LOAD MODEL & WEIGHTS ---
# This robust block handles common deployment errors (like the weight count mismatch)
# by rebuilding the architecture if the standard load fails.
try:
    # Try standard load first (Fastest path if successful)
    model = tf.keras.models.load_model(MODEL_PATH)
except ValueError:
    print("Standard load failed (Version mismatch). Attempting to rebuild architecture...")
    model = build_model_architecture()
    model.load_weights(MODEL_PATH)
    print("Weights loaded successfully.")

# Load class mapping
try:
    with open(INDICES_PATH, 'rb') as f:
        indices = pickle.load(f)
        # Invert mapping: {'Mango': 0} -> {0: 'Mango'}
        LABELS = {v: k for k, v in indices.items()}
except FileNotFoundError:
    print(f"Error: Class indices file not found at {INDICES_PATH}. Predictions will fail.")
    LABELS = {0: "Unknown_0", 1: "Unknown_1", 2: "Unknown_2", 3: "Unknown_3"}


# --- FUNCTIONS ---

def preprocess_image(img_path):
    """Loads, resizes, and preprocesses the image using ResNet50's specific method."""
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {img_path}")
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, IMG_SIZE)
    img_array = np.expand_dims(img_resized, axis=0)
    
    # ResNet50 specific preprocessing (subtracting mean RGB values)
    img_array = preprocess_input(img_array.astype(float))
    return img_array, img

def predict_species(img_path):
    """Performs prediction and returns the top 3 species and confidence scores."""
    processed_img, _ = preprocess_image(img_path)
    # The verbose=0 suppresses Keras output during prediction
    preds = model.predict(processed_img, verbose=0)[0]
    
    # Get Top 3 predicted classes and their indices
    top_indices = preds.argsort()[-3:][::-1]
    top_3 = []
    for i in top_indices:
        if i in LABELS:
            # Convert prediction to percentage
            top_3.append((LABELS[i], float(preds[i]) * 100))
    
    return top_3, processed_img

def generate_gradcam(img_array, save_path):
    """
    Generates a Grad-CAM heatmap showing which regions of the image influenced the prediction.
    """
    
    try:
        # Access the internal base model (ResNet50)
        base_model = model.layers[0]
        classifier_layers = model.layers[1:] # Layers after the base model
        
        with tf.GradientTape() as tape:
            # 1. Get Base Model Output (last convolutional block)
            base_output = base_model(img_array, training=False)
            tape.watch(base_output)
            
            # 2. Pass through classifier head to get final prediction
            x = base_output
            for layer in classifier_layers:
                x = layer(x)
            preds = x
            
            # 3. Get Top Class Score (the neuron corresponding to the predicted class)
            top_class_idx = tf.argmax(preds[0])
            top_class_channel = preds[:, top_class_idx]

        # 4. Gradients of the top class score w.r.t. the last conv layer output
        grads = tape.gradient(top_class_channel, base_output)
        
        # Global average pooling of the gradients to get weight importance for each feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # 5. Create the heatmap (weighted activation map)
        base_output = base_output[0]
        heatmap = base_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Apply ReLU to only consider features that positively influence the prediction
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        
        # 6. Colorize and Resize the heatmap
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.resize(heatmap, IMG_SIZE)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Save the colored heatmap (it will be overlaid in the HTML template)
        cv2.imwrite(save_path, heatmap_colored)
        return True
        
    except Exception as e:
        print(f"Grad-CAM Error: {e}")
        return False