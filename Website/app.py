import os
import numpy as np
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import io

app = Flask(__name__)
app.secret_key = os.environ.get('SESSION_SECRET', 'dev-secret-key')

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MODEL_FOLDER = 'models'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

image_model = None
clinical_model = None
tumor_model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_models():
    global image_model, clinical_model, tumor_model
    
    for ext in ['.keras', '.h5']:
        image_model_path = os.path.join(MODEL_FOLDER, f'image_model{ext}')
        if os.path.exists(image_model_path):
            try:
                from tensorflow import keras
                image_model = keras.models.load_model(image_model_path)
                print(f"Image model loaded successfully from {image_model_path}")
                break
            except Exception as e:
                print(f"Error loading image model: {e}")
    
    clinical_model_path = os.path.join(MODEL_FOLDER, 'clinical_model.pkl')
    if os.path.exists(clinical_model_path):
        try:
            import joblib
            clinical_model = joblib.load(clinical_model_path)
            print("Clinical model loaded successfully")
        except Exception as e:
            print(f"Error loading clinical model: {e}")
    
    tumor_model_path = os.path.join(MODEL_FOLDER, 'tumor_model.pkl')
    if os.path.exists(tumor_model_path):
        try:
            import joblib
            tumor_model = joblib.load(tumor_model_path)
            print("Tumor model loaded successfully")
        except Exception as e:
            print(f"Error loading tumor model: {e}")

@app.route('/')
def index():
    models_status = {
        'image': image_model is not None,
        'clinical': clinical_model is not None,
        'tumor': tumor_model is not None
    }
    return render_template('index.html', models_status=models_status)

@app.route('/histopathology')
def histopathology():
    model_loaded = image_model is not None
    return render_template('histopathology.html', model_loaded=model_loaded)

@app.route('/clinical')
def clinical():
    model_loaded = clinical_model is not None
    return render_template('clinical.html', model_loaded=model_loaded)

@app.route('/tumor')
def tumor():
    model_loaded = tumor_model is not None
    return render_template('tumor.html', model_loaded=model_loaded)

@app.route('/upload-models')
def upload_models_page():
    models_status = {
        'image': image_model is not None,
        'clinical': clinical_model is not None,
        'tumor': tumor_model is not None
    }
    return render_template('upload_models.html', models_status=models_status)

@app.route('/api/upload-model', methods=['POST'])
def upload_model():
    global image_model, clinical_model, tumor_model
    
    if 'model' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'})
    
    file = request.files['model']
    model_type = request.form.get('model_type')
    
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
    
    try:
        if model_type == 'image':
            ext = os.path.splitext(file.filename)[1] or '.keras'
            filename = f'image_model{ext}'
            for old_ext in ['.h5', '.keras']:
                old_file = os.path.join(MODEL_FOLDER, f'image_model{old_ext}')
                if os.path.exists(old_file):
                    os.remove(old_file)
            filepath = os.path.join(MODEL_FOLDER, filename)
            file.save(filepath)
            from tensorflow import keras
            image_model = keras.models.load_model(filepath)
            return jsonify({'success': True, 'message': 'Image model uploaded successfully'})
        
        elif model_type == 'clinical':
            filename = 'clinical_model.pkl'
            filepath = os.path.join(MODEL_FOLDER, filename)
            file.save(filepath)
            import joblib
            clinical_model = joblib.load(filepath)
            return jsonify({'success': True, 'message': 'Clinical model uploaded successfully'})
        
        elif model_type == 'tumor':
            filename = 'tumor_model.pkl'
            filepath = os.path.join(MODEL_FOLDER, filename)
            file.save(filepath)
            import joblib
            tumor_model = joblib.load(filepath)
            return jsonify({'success': True, 'message': 'Tumor model uploaded successfully'})
        
        else:
            return jsonify({'success': False, 'error': 'Invalid model type'})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/predict-image', methods=['POST'])
def predict_image():
    if image_model is None:
        return jsonify({'success': False, 'error': 'Image model not loaded. Please upload the model first.'})
    
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image provided'})
    
    file = request.files['image']
    
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Invalid file'})
    
    try:
        img = Image.open(file.stream)
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = image_model.predict(img_array)
        
        if len(prediction.shape) > 1 and prediction.shape[1] > 1:
            benign_prob = float(prediction[0][0])
            malignant_prob = float(prediction[0][1])
        else:
            malignant_prob = float(prediction[0][0])
            benign_prob = 1 - malignant_prob
        
        result = 'Malignant' if malignant_prob > 0.5 else 'Benign'
        confidence = max(benign_prob, malignant_prob) * 100
        
        return jsonify({
            'success': True,
            'result': result,
            'confidence': round(confidence, 2),
            'benign_probability': round(benign_prob * 100, 2),
            'malignant_probability': round(malignant_prob * 100, 2)
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/predict-clinical', methods=['POST'])
def predict_clinical():
    if clinical_model is None:
        return jsonify({'success': False, 'error': 'Clinical model not loaded. Please upload the model first.'})
    
    try:
        data = request.json
        # Original features
        age = float(data.get('age', 0))
        bmi = float(data.get('bmi', 0))
        glucose = float(data.get('glucose', 0))
        insulin = float(data.get('insulin', 0))
        homa = float(data.get('homa', 0))
        leptin = float(data.get('leptin', 0))
        adiponectin = float(data.get('adiponectin', 0))
        resistin = float(data.get('resistin', 0))
        mcp1 = float(data.get('mcp1', 0))
        
        # Apply the same feature engineering as in data preparation
        # These are the engineered features from your colleague's code
        bmi_glucose = bmi * glucose / 100
        homa_leptin = homa * leptin / 100
        insulin_resistin = insulin * resistin / 100
        adiponectin_leptin_ratio = adiponectin / (leptin + 1e-10)  # Add small constant to avoid division by zero
        
        # Create feature array with original + engineered features
        # Order should match the training data order
        features = [
            age, bmi, glucose, insulin, homa, leptin, 
            adiponectin, resistin, mcp1,
            bmi_glucose, homa_leptin, insulin_resistin, adiponectin_leptin_ratio
        ]
        
        features_array = np.array([features])
        
        # Apply the same scaler used during training
        # You'll need to load the scaler along with the model
        if hasattr(clinical_model, 'scaler'):
            features_array = clinical_model.scaler.transform(features_array)
        
        prediction = clinical_model.predict(features_array)
        
        # Get probabilities
        if hasattr(clinical_model, 'predict_proba'):
            proba = clinical_model.predict_proba(features_array)
            if len(proba[0]) > 1:
                benign_prob = float(proba[0][0]) * 100
                malignant_prob = float(proba[0][1]) * 100
            else:
                malignant_prob = float(proba[0][0]) * 100
                benign_prob = 100 - malignant_prob
        else:
            # For models without probability estimates
            malignant_prob = 75 if prediction[0] == 1 else 25
            benign_prob = 100 - malignant_prob
        
        result = 'High Risk' if prediction[0] == 1 else 'Low Risk'
        confidence = max(benign_prob, malignant_prob)
        
        # Add risk factors explanation based on feature importance
        risk_factors = identify_risk_factors(features, clinical_model)
        
        return jsonify({
            'success': True,
            'result': result,
            'confidence': round(confidence, 2),
            'benign_probability': round(benign_prob, 2),
            'malignant_probability': round(malignant_prob, 2),
            'risk_factors': risk_factors
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def identify_risk_factors(features, model):
    """Identify which features contribute most to the risk prediction"""
    risk_factors = []
    
    # Feature names including engineered features
    feature_names = [
        'Age', 'BMI', 'Glucose', 'Insulin', 'HOMA', 'Leptin', 
        'Adiponectin', 'Resistin', 'MCP.1',
        'BMI-Glucose Index', 'HOMA-Leptin Index', 'Insulin-Resistin Index', 'Adiponectin/Leptin Ratio'
    ]
    
    # Get feature importances if available (for tree-based models)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        for i, importance in enumerate(importances):
            if importance > 0.1:  # Threshold for significant features
                risk_factors.append({
                    'name': feature_names[i],
                    'value': round(features[i], 2),
                    'importance': round(importance * 100, 1)
                })
    
    return risk_factors

@app.route('/api/predict-tumor', methods=['POST'])
def predict_tumor():
    if tumor_model is None:
        return jsonify({'success': False, 'error': 'Tumor model not loaded. Please upload the model first.'})
    
    try:
        data = request.json
        feature_names = [
            'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
            'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
            'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
            'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
            'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
            'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
        ]
        
        features = [float(data.get(name, 0)) for name in feature_names]
        features_array = np.array([features])
        prediction = tumor_model.predict(features_array)
        
        if hasattr(tumor_model, 'predict_proba'):
            proba = tumor_model.predict_proba(features_array)
            if len(proba[0]) > 1:
                benign_prob = float(proba[0][0]) * 100
                malignant_prob = float(proba[0][1]) * 100
            else:
                malignant_prob = float(proba[0][0]) * 100
                benign_prob = 100 - malignant_prob
        else:
            malignant_prob = 50
            benign_prob = 50
        
        result = 'Malignant' if prediction[0] == 1 else 'Benign'
        confidence = max(benign_prob, malignant_prob)
        
        return jsonify({
            'success': True,
            'result': result,
            'confidence': round(confidence, 2),
            'benign_probability': round(benign_prob, 2),
            'malignant_probability': round(malignant_prob, 2)
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

load_models()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
