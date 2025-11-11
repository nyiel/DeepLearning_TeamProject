from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result')
def result():
    return render_template('result.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file uploaded'})
    
    # ⚙️ Replace this section with your model prediction logic
    return jsonify({
        'prediction': 'Neem Leaf',
        'confidence': 97.3,
        'gradcam': '/static/images/sample_gradcam.jpg'
    })

if __name__ == '__main__':
    app.run(debug=True)
