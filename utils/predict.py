@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        unique_name = f"{uuid.uuid4().hex}_{filename}"
        file_path = UPLOAD_FOLDER / unique_name
        file.save(file_path)

        # Preprocess and predict
        img_array = preprocess_image_for_model(file_path)
        preds = model.predict(img_array)
        pred_idx = np.argmax(preds[0])
        pred_label = class_names[pred_idx]
        confidence = float(np.max(preds[0]) * 100)

        # Grad-CAM
        heatmap = make_gradcam_heatmap(img_array, model)
        gradcam_path = RESULTS_FOLDER / f"gradcam_{unique_name}"
        save_gradcam(file_path, heatmap, cam_path=gradcam_path)

        # ✅ FIXED: Grad-CAM URL relative to static folder
        return jsonify({
            'filename': filename,
            'prediction': pred_label,
            'confidence': round(confidence, 2),
            'gradcam': f"/static/results/{gradcam_path.name}"
        })
    else:
        return jsonify({'error': 'Unsupported file type'}), 400
