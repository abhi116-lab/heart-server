# from flask import Flask, request, jsonify
# import joblib
# import numpy as np

# # Load model & label encoder
# model = joblib.load("heart_symptom_model.pkl")
# encoder = joblib.load("label_encoder.pkl")

# app = Flask(__name__)

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.json  # Expecting JSON input
#     # Example: {"Chest Pain":1, "Fever":0, "Hand/Leg Swelling":1, "Shortness of Breath":1, "Fatigue":0, "Irregular Heartbeat":1}
    
#     features = np.array(list(data.values())).reshape(1, -1)
#     prediction = model.predict(features)
#     disease = encoder.inverse_transform(prediction)[0]
    
#     return jsonify({"Predicted Disease": disease})

# if __name__ == '__main__':
#     app.run(debug=True)
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

# Load trained model and encoder
model = joblib.load("heart_symptom_model.pkl")
encoder = joblib.load("label_encoder.pkl")

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json  # Expecting JSON input
        # Convert values to numpy array
        features = np.array(list(data.values())).reshape(1, -1)

        prediction = model.predict(features)
        disease = encoder.inverse_transform(prediction)[0]

        return jsonify({"Predicted Disease": disease})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import joblib
# import numpy as np
# import cv2
# import tensorflow as tf
# from PIL import Image
# import io
# import base64
# import os
# from werkzeug.utils import secure_filename
# import logging

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# app = Flask(__name__)
# CORS(app)  # Enable CORS for React frontend

# # Configuration
# UPLOAD_FOLDER = 'uploads'
# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'dcm', 'dicom'}
# MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# # Create upload directory if it doesn't exist
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # Load trained models
# try:
#     heart_model = joblib.load("heart_symptom_model.pkl")
#     encoder = joblib.load("label_encoder.pkl")
#     logger.info("Heart symptom model and encoder loaded successfully")
# except Exception as e:
#     logger.error(f"Error loading heart model: {e}")
#     heart_model = None
#     encoder = None

# # Load image classification model (you'll need to train/load your MRI model)
# try:
#     # Replace this with your actual MRI classification model
#     # For now, using a placeholder - you'll need to train a model for MRI classification
#     mri_model = None  # tf.keras.models.load_model("mri_classification_model.h5")
#     logger.info("MRI model loaded successfully")
# except Exception as e:
#     logger.error(f"Error loading MRI model: {e}")
#     mri_model = None

# def allowed_file(filename):
#     return '.' in filename and \
#            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# def preprocess_image(image_file):
#     """Preprocess image for MRI analysis"""
#     try:
#         # Read image
#         image = Image.open(image_file)
        
#         # Convert to RGB if necessary
#         if image.mode != 'RGB':
#             image = image.convert('RGB')
        
#         # Resize to standard size (adjust based on your model requirements)
#         image = image.resize((224, 224))
        
#         # Convert to numpy array and normalize
#         image_array = np.array(image) / 255.0
#         image_array = np.expand_dims(image_array, axis=0)
        
#         return image_array
#     except Exception as e:
#         logger.error(f"Error preprocessing image: {e}")
#         return None

# def analyze_mri_image(image_array):
#     """Analyze MRI image using trained model"""
#     try:
#         if mri_model is None:
#             # Placeholder analysis - replace with actual model prediction
#             # For demonstration, returning mock results
#             mock_conditions = [
#                 "Normal cardiac structure",
#                 "Mild left ventricular hypertrophy", 
#                 "Possible myocardial infarction",
#                 "Dilated cardiomyopathy",
#                 "Valvular abnormality"
#             ]
            
#             # Random selection for demo (replace with actual model prediction)
#             import random
#             predicted_condition = random.choice(mock_conditions)
#             confidence = random.uniform(0.75, 0.95)
            
#             return {
#                 "condition": predicted_condition,
#                 "confidence": round(confidence * 100, 1),
#                 "findings": [
#                     "Cardiac chambers within normal limits",
#                     "No obvious wall motion abnormalities",
#                     "Normal ejection fraction estimated"
#                 ]
#             }
#         else:
#             # Actual model prediction
#             prediction = mri_model.predict(image_array)
#             # Process prediction based on your model's output format
#             # This is a placeholder - implement based on your model
#             return {"condition": "Analysis complete", "confidence": 85.0}
            
#     except Exception as e:
#         logger.error(f"Error analyzing MRI: {e}")
#         return None

# def extract_symptoms_from_text(text):
#     """Extract symptoms from text input and convert to binary format"""
#     symptoms_mapping = {
#         "Chest Pain": ["chest pain", "chest discomfort", "angina", "chest pressure"],
#         "Fever": ["fever", "high temperature", "pyrexia", "elevated temperature"],
#         "Hand/Leg Swelling": ["swelling", "edema", "fluid retention", "puffy"],
#         "Shortness of Breath": ["shortness of breath", "dyspnea", "breathing difficulty", "breathless"],
#         "Fatigue": ["fatigue", "tiredness", "weakness", "exhaustion", "tired"],
#         "Irregular Heartbeat": ["irregular heartbeat", "arrhythmia", "palpitations", "heart racing"]
#     }
    
#     text_lower = text.lower()
#     symptoms_binary = {}
    
#     for symptom, keywords in symptoms_mapping.items():
#         symptoms_binary[symptom] = 1 if any(keyword in text_lower for keyword in keywords) else 0
    
#     return symptoms_binary

# @app.route('/health', methods=['GET'])
# def health_check():
#     return jsonify({"status": "healthy", "message": "HeartPulse AI Backend is running"})

# @app.route('/predict-symptoms', methods=['POST'])
# def predict_symptoms():
#     """Predict heart condition based on symptoms"""
#     try:
#         data = request.json
        
#         if not data:
#             return jsonify({"error": "No data provided"}), 400
        
#         # Check if symptoms are provided as text or binary
#         if "symptoms_text" in data:
#             # Extract symptoms from text
#             symptoms_binary = extract_symptoms_from_text(data["symptoms_text"])
#         else:
#             # Use provided binary symptoms
#             symptoms_binary = {
#                 "Chest Pain": data.get("Chest Pain", 0),
#                 "Fever": data.get("Fever", 0),
#                 "Hand/Leg Swelling": data.get("Hand/Leg Swelling", 0),
#                 "Shortness of Breath": data.get("Shortness of Breath", 0),
#                 "Fatigue": data.get("Fatigue", 0),
#                 "Irregular Heartbeat": data.get("Irregular Heartbeat", 0)
#             }
        
#         if heart_model is None or encoder is None:
#             return jsonify({"error": "Heart symptom model not available"}), 500
        
#         # Convert to numpy array for prediction
#         features = np.array(list(symptoms_binary.values())).reshape(1, -1)
#         prediction = heart_model.predict(features)
#         disease = encoder.inverse_transform(prediction)[0]
        
#         # Get prediction probability
#         prediction_proba = heart_model.predict_proba(features)
#         confidence = float(np.max(prediction_proba) * 100)
        
#         return jsonify({
#             "predicted_disease": disease,
#             "confidence": round(confidence, 1),
#             "symptoms_detected": symptoms_binary,
#             "analysis_type": "symptom_based"
#         })
        
#     except Exception as e:
#         logger.error(f"Error in symptom prediction: {e}")
#         return jsonify({"error": str(e)}), 500

# @app.route('/analyze-images', methods=['POST'])
# def analyze_images():
#     """Analyze uploaded medical images"""
#     try:
#         if 'images' not in request.files:
#             return jsonify({"error": "No images provided"}), 400
        
#         files = request.files.getlist('images')
#         results = []
        
#         for file in files:
#             if file and allowed_file(file.filename):
#                 filename = secure_filename(file.filename)
                
#                 # Preprocess image
#                 image_array = preprocess_image(file)
#                 if image_array is None:
#                     continue
                
#                 # Analyze image
#                 analysis = analyze_mri_image(image_array)
#                 if analysis:
#                     results.append({
#                         "filename": filename,
#                         "analysis": analysis
#                     })
        
#         if not results:
#             return jsonify({"error": "No valid images could be analyzed"}), 400
        
#         return jsonify({
#             "image_analysis": results,
#             "total_images": len(results),
#             "analysis_type": "image_based"
#         })
        
#     except Exception as e:
#         logger.error(f"Error in image analysis: {e}")
#         return jsonify({"error": str(e)}), 500

# @app.route('/comprehensive-analysis', methods=['POST'])
# def comprehensive_analysis():
#     """Perform comprehensive analysis using both symptoms and images"""
#     try:
#         # Get symptoms from form data or JSON
#         symptoms_text = request.form.get('symptoms', '')
        
#         if not symptoms_text:
#             return jsonify({"error": "Symptoms text is required"}), 400
        
#         # Analyze symptoms
#         symptoms_binary = extract_symptoms_from_text(symptoms_text)
        
#         symptom_result = None
#         if heart_model and encoder:
#             features = np.array(list(symptoms_binary.values())).reshape(1, -1)
#             prediction = heart_model.predict(features)
#             disease = encoder.inverse_transform(prediction)[0]
#             prediction_proba = heart_model.predict_proba(features)
#             confidence = float(np.max(prediction_proba) * 100)
            
#             symptom_result = {
#                 "predicted_disease": disease,
#                 "confidence": round(confidence, 1),
#                 "symptoms_detected": symptoms_binary
#             }
        
#         # Analyze images if provided
#         image_results = []
#         if 'images' in request.files:
#             files = request.files.getlist('images')
#             for file in files:
#                 if file and allowed_file(file.filename):
#                     filename = secure_filename(file.filename)
#                     image_array = preprocess_image(file)
#                     if image_array is not None:
#                         analysis = analyze_mri_image(image_array)
#                         if analysis:
#                             image_results.append({
#                                 "filename": filename,
#                                 "analysis": analysis
#                             })
        
#         # Combine results
#         final_diagnosis = "Comprehensive Analysis Complete"
#         overall_confidence = 85.0  # This should be calculated based on both analyses
        
#         if symptom_result:
#             final_diagnosis = symptom_result["predicted_disease"]
#             overall_confidence = symptom_result["confidence"]
        
#         # If images provide conflicting or additional information, adjust accordingly
#         if image_results:
#             # This is where you'd implement logic to combine symptom and image analysis
#             # For now, we'll use the symptom-based result as primary
#             pass
        
#         return jsonify({
#             "comprehensive_analysis": {
#                 "final_diagnosis": final_diagnosis,
#                 "overall_confidence": round(overall_confidence, 1),
#                 "symptom_analysis": symptom_result,
#                 "image_analysis": image_results,
#                 "recommendations": [
#                     "Immediate cardiac evaluation recommended",
#                     "Consider ECG and cardiac enzymes",
#                     "Monitor vital signs closely",
#                     "Consult cardiology specialist if symptoms persist"
#                 ],
#                 "urgency_level": "High" if overall_confidence > 80 else "Medium"
#             }
#         })
        
#     except Exception as e:
#         logger.error(f"Error in comprehensive analysis: {e}")
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5000)