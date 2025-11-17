import os
import base64
from datetime import datetime
from flask import Flask, request, redirect, url_for, render_template, jsonify, session, abort
from werkzeug.utils import secure_filename

from model_wrapper import ModelWrapper  # your wrapper that handles preprocessing, prediction, Grad-CAM

# -----------------------------
# Configuration
# -----------------------------
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

MODEL_PATH = 'model/Best_Model.h5'
LABELS_PATH = 'model/labels.json'

# Medicinal-use database
SPECIES_INFO = {
    "Mango": {
        "english_name": "Mango",
        "scientific_name": "Mangifera indica",
        "benefit": (
            "Rich in vitamins A, C, and antioxidants; Supports immunity and digestion. "
            "Contains polyphenols that may reduce inflammation and Improve heart health. "
            "Leaves are traditionally used for treating diabetes and digestive disorders."
        )
    },
    "Neem": {
        "english_name": "Neem",
        "scientific_name": "Azadirachta indica",
        "benefit": (
            "Anti-bacterial, Anti-inflammatory, and Blood-purifying properties. "
            "Leaves are used for skin disorders, acne, and wound healing. "
            "Traditionally supports liver health, boosts immunity, and regulates blood sugar levels."
        )
    },
    "Lemon": {
        "english_name": "Lemon",
        "scientific_name": "Citrus limon",
        "benefit": (
            "Rich in Vitamin C, supports Immunity, Digestion, and Skin health. "
            "Helps detoxify the body and improve hydration. "
            "Lemon juice can aid weight management and has mild antimicrobial properties."
        )
    },
    "Guava": {
        "english_name": "Guava",
        "scientific_name": "Psidium guajava",
        "benefit": (
            "High in fiber and vitamin C; helps in Digestion and Boosts immunity. "
            "Leaves are used for anti-diarrheal and anti-inflammatory purposes. "
            "May help regulate blood sugar and improve heart health due to antioxidants."
        )
    }
}


# Flask app setup
app = Flask(__name__)
app.secret_key = 'supersecretkey123'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -----------------------------
# Load model at startup
# -----------------------------
model = ModelWrapper(MODEL_PATH, LABELS_PATH)

# -----------------------------
# Helper functions
# -----------------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# -----------------------------
# Routes
# -----------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check for file
    file = request.files.get('file')
    if not file or not allowed_file(file.filename):
        abort(400, "Invalid file type")

    # Save uploaded file
    filename = secure_filename(f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{file.filename}")
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(path)

    # Run prediction and Grad-CAM
    preds, top_indices, gradcam_path = model.predict_with_explanation(
        path, save_dir=app.config['UPLOAD_FOLDER']
    )

    # Prepare detailed results
    top_results = []
    for idx, prob in zip(top_indices, preds):
        label = model.labels.get(str(idx), str(idx))
        info = SPECIES_INFO.get(label, {})
        top_results.append({
            'label': label,
            'english_name': info.get('english_name', 'N/A'),
            'scientific_name': info.get('scientific_name', 'N/A'),
            'benefit': info.get('benefit', 'No info available'),
            'score': float(prob)
        })

    # Save to session for rendering result page
    session['result'] = {
        'image_url': f'uploads/{filename}',
        'gradcam_url': f'uploads/{os.path.basename(gradcam_path)}' if gradcam_path else None,
        'results': top_results
    }

    return redirect(url_for('result_page'))

@app.route('/result')
def result_page():
    if 'result' not in session:
        return redirect(url_for('index'))
    return render_template('result.html', **session['result'])

# Optional API endpoint for base64 images
@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json(force=True)
    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    img_b64 = data['image'].split(',')[1] if ',' in data['image'] else data['image']
    try:
        img_bytes = base64.b64decode(img_b64)
    except Exception:
        return jsonify({'error': 'Invalid base64 image'}), 400

    filename = secure_filename(f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_api.jpg")
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    with open(path, 'wb') as f:
        f.write(img_bytes)

    preds, top_indices, gradcam_path = model.predict_with_explanation(
        path, save_dir=app.config['UPLOAD_FOLDER']
    )

    response = {'predictions': []}
    for idx, prob in zip(top_indices, preds):
        label = model.labels.get(str(idx), str(idx))
        info = SPECIES_INFO.get(label, {})
        response['predictions'].append({
            'label': label,
            'english_name': info.get('english_name', 'N/A'),
            'scientific_name': info.get('scientific_name', 'N/A'),
            'benefit': info.get('benefit', 'No info available'),
            'score': float(prob)
        })

    response['explanation_url'] = url_for('static', filename=f'uploads/{os.path.basename(gradcam_path)}') if gradcam_path else None
    return jsonify(response)

# -----------------------------
# Run app
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
