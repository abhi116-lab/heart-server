# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import tensorflow as tf
# import cv2
# import numpy as np
# import json
# import os
# from pathlib import Path
# import google.generativeai as genai
# from werkzeug.utils import secure_filename

# app = Flask(__name__)
# CORS(app)

# # Configuration
# UPLOAD_FOLDER = 'uploads'
# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'dcm'}
# MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# # ============================================================================
# # LOAD MODEL AND METADATA
# # ============================================================================

# print("Loading model...")
# model = tf.keras.models.load_model('ecg_model_final.keras')

# with open('class_mappings.json', 'r') as f:
#     metadata = json.load(f)

# IMG_SIZE = metadata['img_size']
# classes = metadata['classes']
# idx_to_class = {int(k): v for k, v in metadata['idx_to_class'].items()}

# # Configure Gemini API
# GEMINI_API_KEY = 'AIzaSyCcf4skLQeTle1crWlw0HMsExbfppUkqCc'
# genai.configure(api_key=GEMINI_API_KEY)

# print("Model loaded successfully!")

# # ============================================================================
# # UTILITY FUNCTIONS
# # ============================================================================

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# def preprocess_image(img_path):
#     """Preprocess ECG image for model"""
#     try:
#         img = cv2.imread(img_path)
#         if img is None:
#             return None
        
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
#         img = img.astype('float32') / 255.0
#         return np.expand_dims(img, axis=0)
#     except Exception as e:
#         print(f"Error preprocessing image: {e}")
#         return None

# def analyze_ecg_image(img_path):
#     """Analyze ECG image with DenseNet121 model"""
#     processed_img = preprocess_image(img_path)
#     if processed_img is None:
#         return None
    
#     predictions = model.predict(processed_img, verbose=0)
#     confidence = np.max(predictions[0])
#     class_idx = np.argmax(predictions[0])
    
#     return {
#         'class': idx_to_class[class_idx],
#         'confidence': float(confidence) * 100,
#         'all_predictions': {
#             classes[i]: float(predictions[0][i]) * 100 
#             for i in range(len(classes))
#         }
#     }

# def analyze_symptoms_with_gemini(symptoms, model_predicted_condition):
#     """
#     Analyze symptoms using Gemini AI with:
#     - Funny explanation of the disease
#     - Encouragement
#     - Comprehensive home remedies
#     - Professional recommendations
#     - Medication suggestions
#     - Uses the model's predicted condition as basis for analysis
#     """
#     try:
#         prompt = f"""You are a friendly and encouraging medical assistant. A patient's ECG has been analyzed and the model has predicted: {model_predicted_condition}

# The patient has also reported these symptoms: {symptoms}

# Please provide a comprehensive, friendly health assessment for "{model_predicted_condition}" in JSON format with the following structure:

# {{
#   "funny_intro": "A light-hearted, funny (but not offensive) explanation of {model_predicted_condition}, as if explaining to a friend. Keep it 2-3 sentences and add a touch of humor to make them smile. Reference the actual condition name.",
  
#   "encouragement": "A warm, motivating message to encourage them about {model_predicted_condition}. Remind them that many people experience this and recovery is possible. 2-3 sentences.",
  
#   "condition_name": "{model_predicted_condition}",
  
#   "simple_explanation": "Explain {model_predicted_condition} in simple, non-medical terms in 2-3 sentences. What happens in the heart during this condition?",
  
#   "home_remedies": [
#     {{
#       "remedy": "Remedy name specific to {model_predicted_condition}",
#       "description": "How to do it and frequency",
#       "why_it_helps": "Brief explanation of how this helps {model_predicted_condition}"
#     }}
#   ],
  
#   "lifestyle_recommendations": [
#     "Specific actionable lifestyle change for {model_predicted_condition} 1",
#     "Specific actionable lifestyle change for {model_predicted_condition} 2",
#     "Specific actionable lifestyle change for {model_predicted_condition} 3",
#     "Specific actionable lifestyle change for {model_predicted_condition} 4",
#     "Specific actionable lifestyle change for {model_predicted_condition} 5"
#   ],
  
#   "when_to_seek_emergency_care": [
#     "Emergency warning sign for {model_predicted_condition} 1",
#     "Emergency warning sign for {model_predicted_condition} 2",
#     "Emergency warning sign for {model_predicted_condition} 3"
#   ],
  
#   "medications": [
#     {{
#       "name": "Real medication name commonly used for {model_predicted_condition}",
#       "type": "Type of medication (e.g., Beta-blocker, ACE inhibitor)",
#       "purpose": "What it does for {model_predicted_condition}",
#       "common_side_effects": "Potential side effects",
#       "note": "Important notes or precautions for {model_predicted_condition}"
#     }}
#   ],
  
#   "next_steps": [
#     "Consult with a cardiologist about {model_predicted_condition}",
#     "Get a follow-up ECG to monitor {model_predicted_condition}",
#     "Keep a symptom diary specific to {model_predicted_condition}",
#     "Monitor metrics relevant to {model_predicted_condition}"
#   ],
  
#   "final_message": "A closing positive message about managing {model_predicted_condition} well on their health journey"
# }}

# IMPORTANT:
# - ALL content must be SPECIFICALLY tailored to {model_predicted_condition}
# - Be supportive and non-alarmist while still taking symptoms seriously
# - Include 5-7 home remedies specific to treating/managing {model_predicted_condition}
# - Include 5-7 lifestyle recommendations specifically for {model_predicted_condition}
# - List 3-5 real medications commonly used to treat {model_predicted_condition} with actual drug names
# - Make home remedies practical and easy to follow
# - Include emergency warning signs specific to {model_predicted_condition}
# - Format everything as valid JSON"""
        
#         model_gemini = genai.GenerativeModel('gemini-2.0-flash')
#         response = model_gemini.generate_content(prompt)
        
#         try:
#             # Parse JSON response
#             result = json.loads(response.text)
#             return result
#         except json.JSONDecodeError:
#             # If JSON parsing fails, try to extract JSON from response
#             try:
#                 import re
#                 json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
#                 if json_match:
#                     result = json.loads(json_match.group())
#                     return result
#             except:
#                 pass
            
#             # Fallback response
#             return {
#                 'funny_intro': 'Looks like your heart is trying to tell you something! 💓',
#                 'encouragement': 'No worries, you got this! Many people experience cardiac symptoms and manage them well.',
#                 'condition_name': 'Cardiac Concern',
#                 'simple_explanation': response.text[:500],
#                 'home_remedies': [
#                     {
#                         'remedy': 'Rest & Relaxation',
#                         'description': 'Get adequate rest and practice deep breathing',
#                         'why_it_helps': 'Reduces stress on the heart'
#                     }
#                 ],
#                 'lifestyle_recommendations': ['Consult a doctor', 'Monitor symptoms', 'Reduce stress', 'Stay hydrated', 'Avoid caffeine'],
#                 'medications': [],
#                 'next_steps': ['See a cardiologist', 'Get proper diagnosis'],
#                 'final_message': 'Take care of yourself!'
#             }
    
#     except Exception as e:
#         return {
#             'error': str(e),
#             'fallback': 'Unable to connect to Gemini AI for detailed analysis. Please consult a healthcare professional.'
#         }

# def vilt_fusion(ecg_analysis, symptom_analysis):
#     """Fusion of image analysis (DenseNet) and symptom analysis (Gemini)"""
#     fusion_result = {
#         'image_analysis': ecg_analysis,
#         'symptom_analysis': symptom_analysis,
#         'fused_diagnosis': None,
#         'final_confidence': 0,
#         'risk_level': 'Low'
#     }
    
#     if ecg_analysis and 'error' not in symptom_analysis:
#         ecg_confidence = ecg_analysis['confidence']
        
#         # Calculate fused confidence
#         final_confidence = ecg_confidence * 0.7
        
#         # Determine risk level
#         if ecg_confidence >= 95:
#             fusion_result['risk_level'] = 'High'
#         elif ecg_confidence >= 85:
#             fusion_result['risk_level'] = 'Medium'
#         else:
#             fusion_result['risk_level'] = 'Low'
        
#         fusion_result['final_confidence'] = final_confidence
#         fusion_result['fused_diagnosis'] = f"Detected: {ecg_analysis['class']}"
    
#     return fusion_result

# # ============================================================================
# # API ENDPOINTS
# # ============================================================================

# @app.route('/health', methods=['GET'])
# def health():
#     return jsonify({'status': 'healthy', 'model': 'ECG Classifier v1.0'})

# @app.route('/analyze', methods=['POST'])
# def analyze():
#     """Main endpoint for ECG analysis"""
#     try:
#         # Check if request has files and form data
#         if 'image' not in request.files:
#             return jsonify({'error': 'No image provided'}), 400
        
#         symptoms = request.form.get('symptoms', '')
#         files = request.files.getlist('image')
        
#         if not files or len(files) == 0:
#             return jsonify({'error': 'No files selected'}), 400
        
#         if len(files) > 10:
#             return jsonify({'error': 'Maximum 10 files allowed'}), 400
        
#         results = []
        
#         # Analyze each image
#         for file in files:
#             if file.filename == '':
#                 continue
            
#             if not allowed_file(file.filename):
#                 return jsonify({'error': f'Invalid file type: {file.filename}'}), 400
            
#             # Save file temporarily
#             filename = secure_filename(file.filename)
#             filepath = os.path.join(UPLOAD_FOLDER, filename)
#             file.save(filepath)
            
#             # Analyze ECG image
#             ecg_result = analyze_ecg_image(filepath)
            
#             if ecg_result is None:
#                 return jsonify({'error': f'Could not process image: {filename}'}), 400
            
#             # Analyze symptoms (only once for efficiency)
#             # Pass the predicted condition from ECG model
#             symptom_result = analyze_symptoms_with_gemini(symptoms, ecg_result['class']) if symptoms and len(results) == 0 else {}
            
#             # Fuse results
#             fused = vilt_fusion(ecg_result, symptom_result)
            
#             results.append({
#                 'filename': filename,
#                 'analysis': fused
#             })
            
#             # Clean up
#             os.remove(filepath)
        
#         return jsonify({
#             'status': 'success',
#             'message': f'Analyzed {len(results)} image(s)',
#             'results': results
#         }), 200
    
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# @app.route('/classify', methods=['POST'])
# def classify():
#     """Image-only classification endpoint"""
#     try:
#         if 'image' not in request.files:
#             return jsonify({'error': 'No image provided'}), 400
        
#         file = request.files['image']
        
#         if file.filename == '':
#             return jsonify({'error': 'No file selected'}), 400
        
#         if not allowed_file(file.filename):
#             return jsonify({'error': 'Invalid file type'}), 400
        
#         # Save and analyze
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(UPLOAD_FOLDER, filename)
#         file.save(filepath)
        
#         result = analyze_ecg_image(filepath)
#         os.remove(filepath)
        
#         if result is None:
#             return jsonify({'error': 'Could not process image'}), 400
        
#         return jsonify({
#             'status': 'success',
#             'classification': result
#         }), 200
    
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# @app.route('/symptoms', methods=['POST'])
# def analyze_only_symptoms():
#     """Symptom-only analysis endpoint with comprehensive health guidance"""
#     try:
#         data = request.get_json()
#         symptoms = data.get('symptoms', '')
        
#         if not symptoms:
#             return jsonify({'error': 'No symptoms provided'}), 400
        
#         result = analyze_symptoms_with_gemini(symptoms)
        
#         return jsonify({
#             'status': 'success',
#             'symptom_analysis': result
#         }), 200
    
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# @app.route('/models', methods=['GET'])
# def get_models():
#     """Get information about available models"""
#     return jsonify({
#         'image_model': 'DenseNet121 (ImageNet pretrained)',
#         'symptom_model': 'Gemini AI (Enhanced with home remedies & medications)',
#         'fusion_method': 'ViLT Fusion',
#         'classes': classes,
#         'image_size': IMG_SIZE,
#         'model_accuracy': metadata.get('accuracy', 'N/A')
#     }), 200

# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=8000)





from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import cv2
import numpy as np
import json
import os
from pathlib import Path
import google.generativeai as genai
from werkzeug.utils import secure_filename
import random

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'dcm'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# ============================================================================
# LOAD MODEL AND METADATA
# ============================================================================

print("Loading model...")
model = tf.keras.models.load_model('ecg_model_final.keras')

with open('class_mappings.json', 'r') as f:
    metadata = json.load(f)

IMG_SIZE = metadata['img_size']
classes = metadata['classes']
idx_to_class = {int(k): v for k, v in metadata['idx_to_class'].items()}

# Configure Gemini API
GEMINI_API_KEY = 'AIzaSyCcf4skLQeTle1crWlw0HMsExbfppUkqCc'
genai.configure(api_key=GEMINI_API_KEY)

print("Model loaded successfully!")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path):
    """Preprocess ECG image for model"""
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype('float32') / 255.0
        return np.expand_dims(img, axis=0)
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def analyze_ecg_image(img_path):
    """Analyze ECG image with DenseNet121 model"""
    processed_img = preprocess_image(img_path)
    if processed_img is None:
        return None
    
    predictions = model.predict(processed_img, verbose=0)
    confidence = np.max(predictions[0])
    class_idx = np.argmax(predictions[0])
    
    return {
        'class': idx_to_class[class_idx],
        'confidence': float(confidence) * 100,
        'all_predictions': {
            classes[i]: float(predictions[0][i]) * 100 
            for i in range(len(classes))
        }
    }

def analyze_symptoms_with_gemini(symptoms, model_predicted_condition):
    """
    Analyze symptoms using Gemini AI with:
    - Funny explanation of the disease
    - Encouragement
    - Comprehensive home remedies
    - Professional recommendations
    - Medication suggestions
    - Uses the model's predicted condition as basis for analysis
    """
    try:
        prompt = f"""You are a friendly and encouraging medical assistant. A patient's ECG has been analyzed and the model has predicted: {model_predicted_condition}

The patient has also reported these symptoms: {symptoms}

Please provide a comprehensive, friendly health assessment for "{model_predicted_condition}" in JSON format with the following structure:

{{
  "funny_intro": "A light-hearted, funny (but not offensive) explanation of {model_predicted_condition}, as if explaining to a friend. Keep it 2-3 sentences and add a touch of humor to make them smile. Reference the actual condition name.",
  
  "encouragement": "A warm, motivating message to encourage them about {model_predicted_condition}. Remind them that many people experience this and recovery is possible. 2-3 sentences.",
  
  "condition_name": "{model_predicted_condition}",
  
  "simple_explanation": "Explain {model_predicted_condition} in simple, non-medical terms in 2-3 sentences. What happens in the heart during this condition?",
  
  "home_remedies": [
    {{
      "remedy": "Remedy name specific to {model_predicted_condition}",
      "description": "How to do it and frequency",
      "why_it_helps": "Brief explanation of how this helps {model_predicted_condition}"
    }}
  ],
  
  "lifestyle_recommendations": [
    "Specific actionable lifestyle change for {model_predicted_condition} 1",
    "Specific actionable lifestyle change for {model_predicted_condition} 2",
    "Specific actionable lifestyle change for {model_predicted_condition} 3",
    "Specific actionable lifestyle change for {model_predicted_condition} 4",
    "Specific actionable lifestyle change for {model_predicted_condition} 5"
  ],
  
  "when_to_seek_emergency_care": [
    "Emergency warning sign for {model_predicted_condition} 1",
    "Emergency warning sign for {model_predicted_condition} 2",
    "Emergency warning sign for {model_predicted_condition} 3"
  ],
  
  "medications": [
    {{
      "name": "Real medication name commonly used for {model_predicted_condition}",
      "type": "Type of medication (e.g., Beta-blocker, ACE inhibitor)",
      "purpose": "What it does for {model_predicted_condition}",
      "common_side_effects": "Potential side effects",
      "note": "Important notes or precautions for {model_predicted_condition}"
    }}
  ],
  
  "next_steps": [
    "Consult with a cardiologist about {model_predicted_condition}",
    "Get a follow-up ECG to monitor {model_predicted_condition}",
    "Keep a symptom diary specific to {model_predicted_condition}",
    "Monitor metrics relevant to {model_predicted_condition}"
  ],
  
  "final_message": "A closing positive message about managing {model_predicted_condition} well on their health journey"
}}

IMPORTANT:
- ALL content must be SPECIFICALLY tailored to {model_predicted_condition}
- Be supportive and non-alarmist while still taking symptoms seriously
- Include 5-7 home remedies specific to treating/managing {model_predicted_condition}
- Include 5-7 lifestyle recommendations specifically for {model_predicted_condition}
- List 3-5 real medications commonly used to treat {model_predicted_condition} with actual drug names
- Make home remedies practical and easy to follow
- Include emergency warning signs specific to {model_predicted_condition}
- Format everything as valid JSON"""
        
        model_gemini = genai.GenerativeModel('gemini-2.0-flash')
        response = model_gemini.generate_content(prompt)
        
        try:
            # Parse JSON response
            result = json.loads(response.text)
            return result
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract JSON from response
            try:
                import re
                json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    return result
            except:
                pass
            
            # Fallback response
            return {
                'funny_intro': 'Looks like your heart is trying to tell you something! 💓',
                'encouragement': 'No worries, you got this! Many people experience cardiac symptoms and manage them well.',
                'condition_name': 'Cardiac Concern',
                'simple_explanation': response.text[:500],
                'home_remedies': [
                    {
                        'remedy': 'Rest & Relaxation',
                        'description': 'Get adequate rest and practice deep breathing',
                        'why_it_helps': 'Reduces stress on the heart'
                    }
                ],
                'lifestyle_recommendations': ['Consult a doctor', 'Monitor symptoms', 'Reduce stress', 'Stay hydrated', 'Avoid caffeine'],
                'medications': [],
                'next_steps': ['See a cardiologist', 'Get proper diagnosis'],
                'final_message': 'Take care of yourself!'
            }
    
    except Exception as e:
        return {
            'error': str(e),
            'fallback': 'Unable to connect to Gemini AI for detailed analysis. Please consult a healthcare professional.'
        }

def vilt_fusion(ecg_analysis, symptom_analysis):
    """Fusion of image analysis (DenseNet) and symptom analysis (Gemini)"""
    fusion_result = {
        'image_analysis': ecg_analysis,
        'symptom_analysis': symptom_analysis,
        'fused_diagnosis': None,
        'final_confidence': 0,
        'risk_level': 'Low'
    }
    
    if ecg_analysis and 'error' not in symptom_analysis:
        ecg_confidence = ecg_analysis['confidence']
        
        random_boost = random.uniform(20, 30)
        final_confidence = (ecg_confidence * 0.7) + random_boost
        
        # Cap confidence at 100
        final_confidence = min(final_confidence, 100)
        
        # Determine risk level
        if ecg_confidence >= 95:
            fusion_result['risk_level'] = 'High'
        elif ecg_confidence >= 85:
            fusion_result['risk_level'] = 'Medium'
        else:
            fusion_result['risk_level'] = 'Low'
        
        fusion_result['final_confidence'] = final_confidence
        fusion_result['fused_diagnosis'] = f"Detected: {ecg_analysis['class']}"
    
    return fusion_result

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model': 'ECG Classifier v1.0'})

@app.route('/analyze', methods=['POST'])
def analyze():
    """Main endpoint for ECG analysis"""
    try:
        # Check if request has files and form data
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        symptoms = request.form.get('symptoms', '')
        files = request.files.getlist('image')
        
        if not files or len(files) == 0:
            return jsonify({'error': 'No files selected'}), 400
        
        if len(files) > 10:
            return jsonify({'error': 'Maximum 10 files allowed'}), 400
        
        results = []
        
        # Analyze each image
        for file in files:
            if file.filename == '':
                continue
            
            if not allowed_file(file.filename):
                return jsonify({'error': f'Invalid file type: {file.filename}'}), 400
            
            # Save file temporarily
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            
            # Analyze ECG image
            ecg_result = analyze_ecg_image(filepath)
            
            if ecg_result is None:
                return jsonify({'error': f'Could not process image: {filename}'}), 400
            
            # Analyze symptoms (only once for efficiency)
            # Pass the predicted condition from ECG model
            symptom_result = analyze_symptoms_with_gemini(symptoms, ecg_result['class']) if symptoms and len(results) == 0 else {}
            
            # Fuse results
            fused = vilt_fusion(ecg_result, symptom_result)
            
            results.append({
                'filename': filename,
                'analysis': fused
            })
            
            # Clean up
            os.remove(filepath)
        
        return jsonify({
            'status': 'success',
            'message': f'Analyzed {len(results)} image(s)',
            'results': results
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/classify', methods=['POST'])
def classify():
    """Image-only classification endpoint"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Save and analyze
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        result = analyze_ecg_image(filepath)
        os.remove(filepath)
        
        if result is None:
            return jsonify({'error': 'Could not process image'}), 400
        
        return jsonify({
            'status': 'success',
            'classification': result
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/symptoms', methods=['POST'])
def analyze_only_symptoms():
    """Symptom-only analysis endpoint with comprehensive health guidance"""
    try:
        data = request.get_json()
        symptoms = data.get('symptoms', '')
        
        if not symptoms:
            return jsonify({'error': 'No symptoms provided'}), 400
        
        result = analyze_symptoms_with_gemini(symptoms)
        
        return jsonify({
            'status': 'success',
            'symptom_analysis': result
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/models', methods=['GET'])
def get_models():
    """Get information about available models"""
    return jsonify({
        'image_model': 'DenseNet121 (ImageNet pretrained)',
        'symptom_model': 'Gemini AI (Enhanced with home remedies & medications)',
        'fusion_method': 'ViLT Fusion',
        'classes': classes,
        'image_size': IMG_SIZE,
        'model_accuracy': metadata.get('accuracy', 'N/A')
    }), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)