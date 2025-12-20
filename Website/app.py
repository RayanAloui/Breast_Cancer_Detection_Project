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
class SimpleStandardScaler:
    """Simple implementation of StandardScaler"""
    
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std
    
    def fit(self, X):
        """Calculate mean and std from data"""
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        # Avoid division by zero
        self.std[self.std == 0] = 1.0
        return self
    
    def transform(self, X):
        """Apply scaling to data"""
        if self.mean is None or self.std is None:
            raise ValueError("Scaler not fitted yet")
        return (X - self.mean) / self.std
    
    def fit_transform(self, X):
        """Fit and transform in one step"""
        return self.fit(X).transform(X)
class TensorFlowModelWrapper:
    """Wrapper for TensorFlow softmax regression model saved with joblib"""
    
    def __init__(self, W, b):
        """
        Initialize with weights and bias
        W: weight matrix (input_dim x output_dim)
        b: bias vector (output_dim)
        """
        self.W = W
        self.b = b
        
    def softmax(self, x):
        """Softmax function"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def predict(self, X):
        """Predict class labels"""
        # Convert to numpy array if needed
        if not isinstance(X, np.ndarray):
            X = np.array(X, dtype=np.float32)
        
        # Compute logits
        logits = np.dot(X, self.W) + self.b
        
        # Apply softmax
        probabilities = self.softmax(logits)
        
        # Return class with highest probability
        return np.argmax(probabilities, axis=1)
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        # Convert to numpy array if needed
        if not isinstance(X, np.ndarray):
            X = np.array(X, dtype=np.float32)
        
        # Compute logits
        logits = np.dot(X, self.W) + self.b
        
        # Apply softmax
        return self.softmax(logits)
    
    def __repr__(self):
        return f"TensorFlowModelWrapper(W shape: {self.W.shape}, b shape: {self.b.shape})"

# Add this global variable near the top
clinical_scaler = None

def create_clinical_scaler():
    """Create and fit a scaler with estimated values for clinical data"""
    global clinical_scaler
    
    # Generate synthetic data with 14 features
    np.random.seed(42)
    n_samples = 1000
    
    synthetic_data = np.column_stack([
        np.random.normal(52, 15, n_samples),      # Age
        np.random.normal(28, 6, n_samples),       # BMI
        np.random.normal(100, 20, n_samples),     # Glucose
        np.random.normal(15, 8, n_samples),       # Insulin
        np.random.normal(2.5, 1.5, n_samples),    # HOMA
        np.random.normal(40, 20, n_samples),      # Leptin
        np.random.normal(10, 5, n_samples),       # Adiponectin
        np.random.normal(30, 15, n_samples),      # Resistin
        np.random.normal(300, 150, n_samples),    # MCP.1
        np.random.binomial(1, 0.3, n_samples),    # BMI_Category (30% obese)
        np.random.binomial(1, 0.5, n_samples),    # High_Glucose (50% > 100)
        np.random.binomial(1, 0.2, n_samples),    # High_Insulin (20% > 25)
        np.random.exponential(5, n_samples),      # Leptin_Adiponectin_Ratio
        np.random.normal(9000, 5000, n_samples)   # Inflammatory_Index
    ])
    
    clinical_scaler = SimpleStandardScaler()
    clinical_scaler.fit(synthetic_data)
    
    print(f"\nCreated clinical scaler with {synthetic_data.shape[1]} features")
    print(f"Mean shape: {clinical_scaler.mean.shape}")
    
    return clinical_scaler


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
                print(f"✓ Image model loaded successfully from {image_model_path}")
                break
            except Exception as e:
                print(f"✗ Error loading image model: {e}")
    
    clinical_model_path = os.path.join(MODEL_FOLDER, 'clinical_model.pkl')
    if os.path.exists(clinical_model_path):
        try:
            import joblib
            clinical_model = joblib.load(clinical_model_path)
            print("✓ Clinical model loaded successfully")
            print(f"  Model type: {type(clinical_model)}")
        except Exception as e:
            print(f"✗ Error loading clinical model: {e}")
    if clinical_model:
        print(f"\nCreating clinical data scaler...")
        clinical_scaler = create_clinical_scaler()
        
        # Test the model with scaling
        print(f"\nTesting clinical model with scaling...")
        try:
            # Test with a sample case
            test_features = np.array([[54, 35.20738923, 103, 5.642, 1.37866043, 
                                      65.6699, 9.738408, 31.17499, 197.66,
                                      54*35.20738923/100, 1.37866043*65.6699/100, 
                                      5.642*31.17499/100, 9.738408/(65.6699+1e-10)]], dtype=np.float32)
            
            if clinical_scaler and test_features.shape[1] == clinical_scaler.mean.shape[0]:
                test_scaled = clinical_scaler.transform(test_features)
                pred = clinical_model.predict(test_scaled)
                print(f"Test prediction with scaling: {pred[0]}")
            else:
                pred = clinical_model.predict(test_features)
                print(f"Test prediction without scaling: {pred[0]}")
        except Exception as e:
            print(f"Test failed: {str(e)}")
    tumor_model_path = os.path.join(MODEL_FOLDER, 'tumor_model.pkl')
    if os.path.exists(tumor_model_path):
        try:
            import joblib
            loaded_obj = joblib.load(tumor_model_path)
            print(f"✓ Tumor model object loaded, type: {type(loaded_obj)}")
            
            # Check what we loaded
            if isinstance(loaded_obj, dict):
                print("  Loaded object is a dictionary")
                # If it's a dictionary with W and b keys
                if 'W' in loaded_obj and 'b' in loaded_obj:
                    print("  Creating wrapper from dictionary")
                    tumor_model = TensorFlowModelWrapper(loaded_obj['W'], loaded_obj['b'])
                elif 'weights' in loaded_obj and 'bias' in loaded_obj:
                    print("  Creating wrapper from weights/bias")
                    tumor_model = TensorFlowModelWrapper(loaded_obj['weights'], loaded_obj['bias'])
                else:
                    print(f"  Dictionary keys: {loaded_obj.keys()}")
                    # Try to find W and b in the dictionary
                    for key, value in loaded_obj.items():
                        if isinstance(value, np.ndarray):
                            print(f"    {key}: shape={value.shape}, dtype={value.dtype}")
            elif hasattr(loaded_obj, 'W') and hasattr(loaded_obj, 'b'):
                # If it's already an object with W and b attributes
                print("  Loaded object has W and b attributes")
                tumor_model = TensorFlowModelWrapper(loaded_obj.W, loaded_obj.b)
            elif hasattr(loaded_obj, 'predict'):
                # If it's already a scikit-learn style model
                print("  Loaded object has predict method")
                tumor_model = loaded_obj
            elif isinstance(loaded_obj, tuple) and len(loaded_obj) == 2:
                # If it's a tuple (W, b)
                print("  Loaded object is a tuple (W, b)")
                tumor_model = TensorFlowModelWrapper(loaded_obj[0], loaded_obj[1])
            else:
                print(f"  Unknown object type, trying to wrap anyway")
                # Try to use it as-is
                tumor_model = loaded_obj
            
            print(f"  Final tumor_model type: {type(tumor_model)}")
            print(f"  Has predict: {hasattr(tumor_model, 'predict')}")
            print(f"  Has predict_proba: {hasattr(tumor_model, 'predict_proba')}")
                
        except Exception as e:
            print(f"✗ Error loading tumor model: {e}")
            import traceback
            traceback.print_exc()

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
    
    temp_path = None
    try:
        # Save the file temporarily
        import tempfile
        import tensorflow as tf
        
        # Create temporary file
        temp_fd, temp_path = tempfile.mkstemp(suffix='.png')
        os.close(temp_fd)  # Close the file descriptor
        
        # Save the uploaded file
        file.save(temp_path)
        
        # Use EXACTLY the same preprocessing as in training
        def preprocess_image(path):
            # Replicate your training preprocessing exactly
            img = tf.io.read_file(path)
            img = tf.io.decode_png(img, channels=3)
            img = tf.image.resize(img, (50, 50))  # Your IMG_SIZE was 50
            img = tf.image.convert_image_dtype(img, tf.float32)  # This scales to [0,1]
            return img
        
        # Preprocess the image
        img_tensor = preprocess_image(temp_path)
        
        # Add batch dimension
        img_array = tf.expand_dims(img_tensor, axis=0)
        
        print(f"\n=== DEBUG: Image Tensor Info ===")
        print(f"Image tensor shape: {img_array.shape}")
        print(f"Image tensor dtype: {img_array.dtype}")
        print(f"Image tensor range: [{tf.reduce_min(img_array):.3f}, {tf.reduce_max(img_array):.3f}]")
        
        # Make prediction
        prediction = image_model.predict(img_array, verbose=0)
        
        print(f"Raw prediction: {prediction}")
        print(f"Prediction shape: {prediction.shape}")
        
        # Your model has sigmoid output
        malignant_prob = float(prediction[0][0])  # Single value from sigmoid
        benign_prob = 1 - malignant_prob
        
        print(f"Malignant probability: {malignant_prob:.6f}")
        print(f"Benign probability: {benign_prob:.6f}")
        
        result = 'Malignant' if malignant_prob > 0.5 else 'Benign'
        confidence = max(benign_prob, malignant_prob) * 100
        
        print(f"Result: {result}, Confidence: {confidence:.2f}%")
        
        # Clean up
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
        
        return jsonify({
            'success': True,
            'result': result,
            'confidence': round(confidence, 2),
            'benign_probability': round(benign_prob * 100, 2),
            'malignant_probability': round(malignant_prob * 100, 2)
        })
    
    except Exception as e:
        # Clean up on error
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass
        
        print(f"Error in predict_image: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/predict-clinical', methods=['POST'])
def predict_clinical():
    if clinical_model is None:
        return jsonify({'success': False, 'error': 'Clinical model not loaded. Please upload the model first.'})
    
    try:
        data = request.json
        
        # Extract and cast all values to float
        age = float(data.get('age', 0))
        bmi = float(data.get('bmi', 0))
        glucose = float(data.get('glucose', 0))
        insulin = float(data.get('insulin', 0))
        homa = float(data.get('homa', 0))
        leptin = float(data.get('leptin', 0))
        adiponectin = float(data.get('adiponectin', 0))
        resistin = float(data.get('resistin', 0))
        mcp1 = float(data.get('mcp1', 0))
        
        print(f"\n=== CLINICAL PREDICTION DEBUG ===")
        print(f"Raw input: age={age}, bmi={bmi}, glucose={glucose}, insulin={insulin}")
        
        # Based on typical Coimbra breast cancer dataset ranges
        # Create features - use the engineered features from your training code
        bmi_category = 1 if bmi > 30 else 0
        high_glucose = 1 if glucose > 100 else 0
        high_insulin = 1 if insulin > 25 else 0
        leptin_adiponectin_ratio = leptin / (adiponectin + 1e-6)
        inflammatory_index = resistin * mcp1
        
        # Create the feature array in the SAME ORDER as during training
        # The model expects 14 features
        all_features = [
            age,
            bmi,
            glucose,
            insulin,
            homa,
            leptin,
            adiponectin,
            resistin,
            mcp1,
            bmi_category,                  # BMI_Category (0 or 1)
            high_glucose,                  # High_Glucose (0 or 1)
            high_insulin,                  # High_Insulin (0 or 1)
            leptin_adiponectin_ratio,      # Leptin_Adiponectin_Ratio
            inflammatory_index             # Inflammatory_Index
        ]
        
        features_array = np.array([all_features], dtype=np.float32)
        print(f"Features array shape: {features_array.shape}")
        print(f"Features: {all_features}")
        
        # Apply MANUAL scaling based on typical ranges from Coimbra dataset
        # These are educated guesses - adjust based on your actual data distribution
        print("\nApplying manual scaling...")
        
        # Typical means for Coimbra dataset (approximate)
        typical_means = np.array([
            52,      # Age
            28,      # BMI
            100,     # Glucose
            15,      # Insulin
            2.5,     # HOMA
            40,      # Leptin
            10,      # Adiponectin
            30,      # Resistin
            300,     # MCP.1
            0.3,     # BMI_Category (30% obese)
            0.5,     # High_Glucose (50% > 100)
            0.2,     # High_Insulin (20% > 25)
            5.0,     # Leptin_Adiponectin_Ratio
            9000     # Inflammatory_Index
        ])
        
        # Typical standard deviations
        typical_stds = np.array([
            15,      # Age
            6,       # BMI
            20,      # Glucose
            8,       # Insulin
            1.5,     # HOMA
            20,      # Leptin
            5,       # Adiponectin
            15,      # Resistin
            150,     # MCP.1
            0.46,    # BMI_Category std for binary
            0.5,     # High_Glucose std for binary
            0.4,     # High_Insulin std for binary
            3.0,     # Leptin_Adiponectin_Ratio
            5000     # Inflammatory_Index
        ])
        
        # Apply manual scaling
        features_scaled = (features_array - typical_means) / typical_stds
        print(f"After scaling (first 5): {features_scaled[0][:5]}")
        
        # Try to detect what the model expects
        print("\nChecking model expectations...")
        if hasattr(clinical_model, 'n_features_in_'):
            expected_features = clinical_model.n_features_in_
            print(f"Model expects {expected_features} features")
            
            if expected_features != len(all_features):
                print(f"WARNING: Expected {expected_features} features, but have {len(all_features)}")
                # Try to match by truncating or padding
                if expected_features < len(all_features):
                    features_scaled = features_scaled[:, :expected_features]
                    print(f"Truncated to {expected_features} features")
        
        # Make prediction
        print("\nMaking prediction...")
        prediction = clinical_model.predict(features_scaled)
        print(f"Raw prediction: {prediction}")
        
        # Get probabilities
        if hasattr(clinical_model, 'predict_proba'):
            try:
                proba = clinical_model.predict_proba(features_scaled)
                print(f"Probabilities: {proba[0]}")
                
                if len(proba[0]) > 1:
                    # Assuming class 0 = Healthy (Low Risk), class 1 = Patients (High Risk)
                    benign_prob = float(proba[0][0]) * 100
                    malignant_prob = float(proba[0][1]) * 100
                else:
                    malignant_prob = float(proba[0][0]) * 100
                    benign_prob = 100 - malignant_prob
            except:
                print("predict_proba failed, using fallback")
                benign_prob = 30 if prediction[0] == 1 else 70
                malignant_prob = 100 - benign_prob
        else:
            print("No predict_proba available, estimating probabilities")
            if prediction[0] == 1:
                malignant_prob = 70  # High Risk
                benign_prob = 30
            else:
                malignant_prob = 30  # Low Risk
                benign_prob = 70
        
        # Map prediction to result
        # In your training: 0=Healthy controls, 1=Patients
        result = 'High Risk' if prediction[0] == 1 else 'Low Risk'
        confidence = max(benign_prob, malignant_prob)
        
        print(f"\n=== FINAL RESULT ===")
        print(f"Result: {result}")
        print(f"Confidence: {confidence:.2f}%")
        print(f"Low Risk probability: {benign_prob:.2f}%")
        print(f"High Risk probability: {malignant_prob:.2f}%")
        
        # If confidence is very low, indicate uncertainty
        if confidence < 55:
            result = f"Uncertain ({result})"
            print("WARNING: Model confidence is low (<55%)")
        
        return jsonify({
            'success': True,
            'result': result,
            'confidence': round(confidence, 2),
            'benign_probability': round(benign_prob, 2),
            'malignant_probability': round(malignant_prob, 2),
            'debug': {
                'feature_count': features_scaled.shape[1],
                'prediction_value': int(prediction[0]),
                'scaling_applied': True
            }
        })
    
    except Exception as e:
        print(f"\n=== ERROR ===")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/predict-tumor', methods=['POST'])
def predict_tumor():
    if tumor_model is None:
        return jsonify({'success': False, 'error': 'Tumor model not loaded.'})
    
    try:
        data = request.json
        
        # Extract features in correct order
        feature_names = [
            'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
            'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
            'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
            'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
            'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
            'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
        ]
        
        features = [float(data.get(name, 0)) for name in feature_names]
        features_array = np.array([features], dtype=np.float32)
        
        # Apply manual scaling with typical values
        print("Applying manual scaling...")
        
        # Typical means from breast cancer dataset
        typical_means = np.array([14.13, 19.29, 91.97, 654.89, 0.09636,
                                  0.10434, 0.0888, 0.0489, 0.1812, 0.0628,
                                  0.4052, 1.2169, 2.866, 40.34, 0.00704,
                                  0.02548, 0.0319, 0.0118, 0.0205, 0.00379,
                                  16.27, 25.68, 107.26, 880.58, 0.1324,
                                  0.2548, 0.2722, 0.1146, 0.2901, 0.08395])
        
        # Typical standard deviations
        typical_stds = np.array([3.524, 4.301, 24.53, 351.9, 0.01406,
                                 0.05281, 0.07972, 0.0388, 0.0274, 0.00706,
                                 0.2773, 0.5516, 2.022, 45.49, 0.003,
                                 0.01791, 0.03043, 0.00617, 0.0088, 0.00265,
                                 4.833, 6.146, 33.6, 569.4, 0.0228,
                                 0.1573, 0.2086, 0.0659, 0.0619, 0.01806])
        
        # Apply scaling
        features_scaled = (features_array - typical_means) / typical_stds
        
        # Now use the scaled features for prediction
        if hasattr(tumor_model, 'predict'):
            prediction = tumor_model.predict(features_scaled)  # Use scaled features
            print(f"Raw prediction: {prediction}")  # DEBUG
        else:
            return jsonify({'success': False, 'error': 'Model does not have predict method'})
        
        # Get probabilities - MUST USE SCALED FEATURES HERE TOO!
        if hasattr(tumor_model, 'predict_proba'):
            proba = tumor_model.predict_proba(features_scaled)  # FIXED: Use scaled features
            print(f"Raw probabilities: {proba}")  # DEBUG
            
            if len(proba[0]) > 1:
                # Assuming index 0 = Benign, index 1 = Malignant
                benign_prob = float(proba[0][0]) * 100
                malignant_prob = float(proba[0][1]) * 100
                print(f"Benign probability from model: {benign_prob:.2f}%")
                print(f"Malignant probability from model: {malignant_prob:.2f}%")
            else:
                # For binary classification with single probability output
                malignant_prob = float(proba[0][0]) * 100
                benign_prob = 100 - malignant_prob
                print(f"Single probability output: {malignant_prob:.2f}%")
        else:
            # For models without probability estimates
            print("Model doesn't have predict_proba, using fallback")
            malignant_prob = 75 if prediction[0] == 1 else 25
            benign_prob = 100 - malignant_prob
        
        # Determine result
        if prediction[0] == 1:
            result = 'Malignant'
            confidence = malignant_prob
        else:
            result = 'Benign'
            confidence = benign_prob
        
        print(f"Final result: {result}")
        print(f"Benign probability: {benign_prob:.2f}%")
        print(f"Malignant probability: {malignant_prob:.2f}%")
        print(f"Confidence: {confidence:.2f}%")
        
        # Sanity check: probabilities should add to ~100%
        total_prob = benign_prob + malignant_prob
        if abs(total_prob - 100) > 1:  # Allow small floating point errors
            print(f"WARNING: Probabilities don't add to 100%: {total_prob:.2f}%")
            # Normalize them
            benign_prob = (benign_prob / total_prob) * 100
            malignant_prob = (malignant_prob / total_prob) * 100
            confidence = max(benign_prob, malignant_prob)
        
        return jsonify({
            'success': True,
            'result': result,
            'confidence': round(confidence, 2),
            'benign_probability': round(benign_prob, 2),
            'malignant_probability': round(malignant_prob, 2)
        })
    
    except Exception as e:
        print(f"Error in predict_tumor: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

load_models()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
