import os
import requests
import numpy as np
import hashlib
import time
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from flask_cors import CORS
import uuid
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

# Configure CORS for production
CORS(app, origins=[
    "https://brain-tumor-frontend.onrender.com",  # Your actual frontend URL
    "https://brain-rumor-api.onrender.com",       # Allow backend to call itself
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:5000"
])

# Model configuration
GOOGLE_DRIVE_FILE_ID = "1AFQMBxhUsok-Z6lBJ0zHCFdFLWgNbgmW"
MODEL_URL = f"https://drive.google.com/uc?export=download&id={GOOGLE_DRIVE_FILE_ID}"
MODEL_PATH = "model.h5"
MODEL_INFO_PATH = "model_info.txt"  # Store model metadata
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Default class labels
class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

def get_model_info():
    """Get expected model information for validation"""
    return {
        "file_id": GOOGLE_DRIVE_FILE_ID,
        "min_size": 80 * 1024 * 1024,  # Minimum 80MB
        "max_size": 200 * 1024 * 1024,  # Maximum 200MB
    }

def is_model_valid():
    """Check if the current model file is valid"""
    if not os.path.exists(MODEL_PATH):
        return False
    
    # Check file size
    file_size = os.path.getsize(MODEL_PATH)
    model_info = get_model_info()
    
    if file_size < model_info["min_size"] or file_size > model_info["max_size"]:
        logger.warning(f"Model file size {file_size} bytes is outside expected range")
        return False
    
    # Check if model info file exists and matches
    if os.path.exists(MODEL_INFO_PATH):
        with open(MODEL_INFO_PATH, 'r') as f:
            stored_info = f.read().strip()
            if stored_info == GOOGLE_DRIVE_FILE_ID:
                logger.info("Model validation passed - using cached model")
                return True
    
    return False

def save_model_info():
    """Save model information after successful download"""
    with open(MODEL_INFO_PATH, 'w') as f:
        f.write(GOOGLE_DRIVE_FILE_ID)

def download_model_from_drive():
    """Download model from Google Drive with caching"""
    if is_model_valid():
        logger.info("Valid cached model found, skipping download")
        return True
        
    logger.info("Downloading fresh model from Google Drive...")
    
    try:
        # Download with progress tracking
        response = requests.get(MODEL_URL, stream=True, timeout=300)
        
        # Handle Google Drive's virus scan warning
        if "virus scan warning" in response.text.lower():
            import re
            confirm_token = re.search(r'confirm=([0-9A-Za-z_]+)', response.text)
            if confirm_token:
                confirm_url = f"https://drive.google.com/uc?export=download&confirm={confirm_token.group(1)}&id={GOOGLE_DRIVE_FILE_ID}"
                response = requests.get(confirm_url, stream=True, timeout=300)
        
        response.raise_for_status()
        
        # Download with progress
        total_size = 0
        start_time = time.time()
        
        with open(MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    total_size += len(chunk)
                    
                    # Log progress every 10MB
                    if total_size % (10 * 1024 * 1024) < 8192:
                        elapsed = time.time() - start_time
                        logger.info(f"Downloaded {total_size / (1024*1024):.1f} MB ({elapsed:.1f}s)")
        
        download_time = time.time() - start_time
        logger.info(f"Model download completed! Size: {total_size / (1024*1024):.1f} MB in {download_time:.1f}s")
        
        # Validate downloaded model
        if is_model_valid():
            save_model_info()
            return True
        else:
            logger.error("Downloaded model failed validation")
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)
            return False
        
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        return False

def load_model_safe():
    """Safely load the model with error handling"""
    try:
        model = load_model(MODEL_PATH)
        logger.info("Model loaded successfully!")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

# Download and load model
model = None
logger.info("Initializing Brain Tumor Detection API...")

if download_model_from_drive():
    model = load_model_safe()
else:
    logger.error("Failed to download model. API will not work properly.")

def preprocess_image(image, image_size=128):
    """Preprocess image for model prediction"""
    img = load_img(image, target_size=(image_size, image_size))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(image_path):
    """Make prediction on the uploaded image"""
    if model is None:
        return {"error": "Model not loaded. Please try again later."}
    
    try:
        img_array = preprocess_image(image_path)
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        confidence_score = np.max(predictions)
        
        predicted_label = class_labels[predicted_class_index]
        
        if predicted_label.lower() == 'notumor':
            result = "No Tumor"
        else:
            result = f"Tumor Found: {predicted_label}"
        
        return {
            "result": result,
            "confidence": f"{confidence_score * 100:.2f}%"
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"error": f"Prediction failed: {str(e)}"}

# Routes (keep all your existing routes)
@app.route('/', methods=['GET'])
def home():
    """Root endpoint"""
    return jsonify({
        "message": "Brain Tumor Detection API",
        "status": "healthy",
        "model_cached": is_model_valid(),
        "endpoints": {
            "/health": "Check API health",
            "/predict": "Upload image for prediction (POST)"
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "model_cached": is_model_valid(),
        "class_labels": class_labels,
        "model_path": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH)
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        logger.info("Prediction request received")
        
        # Check if model is loaded
        if model is None:
            logger.error("Model not loaded")
            return jsonify({"error": "Model not available. Please try again later."}), 503
        
        # Check if 'image' is part of the request
        if 'image' not in request.files:
            logger.warning("No image in request")
            return jsonify({"error": "No image uploaded"}), 400

        image_file = request.files['image']
        if image_file.filename == '':
            logger.warning("Empty filename")
            return jsonify({"error": "No file selected"}), 400

        # Create uploads directory
        uploads_dir = os.path.join(BASE_DIR, 'uploads')
        os.makedirs(uploads_dir, exist_ok=True)
        
        # Generate safe filename
        file_extension = os.path.splitext(image_file.filename)[1]
        if not file_extension:
            file_extension = '.jpg'
        safe_filename = str(uuid.uuid4()) + file_extension
        image_path = os.path.join(uploads_dir, safe_filename)
        
        # Save image
        image_file.save(image_path)
        logger.info(f"Image saved: {safe_filename}")

        try:
            # Make prediction
            output = predict_image(image_path)
            logger.info(f"Prediction completed: {output}")
            return jsonify(output)
            
        finally:
            # Clean up
            try:
                if os.path.exists(image_path):
                    os.remove(image_path)
                    logger.info(f"Cleaned up: {safe_filename}")
            except Exception as cleanup_error:
                logger.warning(f"Cleanup failed: {cleanup_error}")
    
    except Exception as e:
        logger.error(f"Prediction endpoint error: {e}")
        return jsonify({"error": "Internal server error. Please try again."}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Create uploads directory
    os.makedirs('uploads', exist_ok=True)
    
    port = int(os.environ.get('PORT', 5000))
    
    logger.info("=" * 50)
    logger.info("Brain Tumor Detection API Starting")
    logger.info("=" * 50)
    logger.info(f"Port: {port}")
    logger.info(f"Model loaded: {model is not None}")
    logger.info(f"Model cached: {is_model_valid()}")
    logger.info(f"Class labels: {class_labels}")
    logger.info("=" * 50)
    
    if model is None:
        logger.warning("⚠️  WARNING: Model not loaded!")
    else:
        logger.info("✅ Model ready for predictions!")
    
    app.run(host='0.0.0.0', port=port, debug=False)
