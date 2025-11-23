# app.py
import os
import time
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
from utils.model_utils import predict_species, generate_gradcam
from utils.medicinal_data import get_plant_info

app = Flask(__name__)

# CONFIG
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['GRADCAM_FOLDER'] = 'static/explanations'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # Limit 10MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['GRADCAM_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error="No file uploaded")
    
    file = request.files['file']
    
    if file.filename == '':
        return render_template('index.html', error="No file selected")

    if file and allowed_file(file.filename):
        # 1. Save Image
        filename = secure_filename(f"{int(time.time())}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 2. Run Prediction
        top_3, processed_img = predict_species(filepath)
        best_species, confidence_score = top_3[0] # Score is already 0-100 from model_utils
        
        # 3. Generate Grad-CAM
        cam_filename = f"gradcam_{filename}"
        cam_path = os.path.join(app.config['GRADCAM_FOLDER'], cam_filename)
        generate_gradcam(processed_img, cam_path)
        
        # 4. Prepare Data for Result.html
        plant_info = get_plant_info(best_species)
        
        # Construct the 'results' list expected by your template
        results = [{
            'label': best_species,
            'score': confidence_score / 100.0, # Template expects 0.0-1.0 (it multiplies by 100)
            'english_name': plant_info['english_name'],
            'scientific_name': plant_info['scientific_name'],
            'benefit': plant_info['benefit']
        }]
        
        # Relative paths for HTML
        image_url = f"uploads/{filename}"
        gradcam_url = f"explanations/{cam_filename}"

        return render_template('result.html', 
                               results=results, 
                               image_url=image_url, 
                               gradcam_url=gradcam_url)
    
    return render_template('index.html', error="Invalid file type")

if __name__ == '__main__':
    app.run(debug=True, port=5000)