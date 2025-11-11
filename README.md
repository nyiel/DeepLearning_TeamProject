# 🌿 Medicinal Leaf Classifier

![Python](https://img.shields.io/badge/python-3.11-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.3-orange?logo=flask&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14-orange?logo=tensorflow&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

A **deep learning web application** that classifies medicinal plant leaves (e.g., Neem, Mango, Lemon, Guava) using a trained CNN model.  
Built with **TensorFlow**, **Flask**, and **Bootstrap**, the app allows users to upload a leaf image and instantly get a prediction with visual feedback.

---

## 🚀 Features

- 🧠 **Deep Learning Model:** EfficientNet-based CNN trained on a curated medicinal leaf dataset  
- 📷 **Image Upload:** Upload or capture a photo of a leaf directly from your device  
- 🔍 **Prediction Results:** Displays the predicted species with confidence score  
- 🌡️ **Grad-CAM Visualization:** Highlights which regions of the image influenced the model’s decision  
- 💻 **User-Friendly UI:** Built with Bootstrap and responsive CSS for all screen sizes  

---

## 🧰 Tech Stack

- **Frontend:** HTML5, CSS3, Bootstrap, JavaScript  
- **Backend:** Flask (Python)  
- **Model:** TensorFlow / Keras EfficientNet  
- **Visualization:** Grad-CAM for explainability  

---

## 🗂️ Project Structure

Medicinal Plant Classifier/
│
├── app.py # Flask app
├── model/ # Trained model (.h5 or .keras)
├── static/
│ ├── style.css # Custom CSS
│ └── script.js # Client-side JS
├── templates/
│ ├── index.html # Upload page
│ └── result.html # Prediction page
├── utils/
│ ├── preprocess.py # Image preprocessing utilities
│ └── gradcam.py # Grad-CAM implementation
└── README.md


---

## ⚙️ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/nyiel/DeepLearning_TeamProject.git
cd DeepLearning_TeamProject
Create and activate a virtual environment (recommended):


python -m venv venv
venv\Scripts\activate       # On Windows
source venv/bin/activate    # On macOS/Linux
Install dependencies:


pip install -r requirements.txt
Run the Flask app:


python app.py
Open your browser and go to:
http://127.0.0.1:5000/
```
📖 Authors (MSc. AI – Team DeepLearning)

1. Kon James Ayuen

2. Malith Dut Malual

3. Deng Peter Nyuon

4. Biar Chagai Atem

5. Akot Deng Akot

🧩 License
This project is licensed under the MIT License — see the LICENSE file for details.

💚 Acknowledgments
Special thanks to:

1. Ustaz. Thon Malek Garang
   
2. University of Juba – MSc. AI Program

3. TensorFlow and Flask open-source communities



