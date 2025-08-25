#uvicorn app:app --reload
import os
import uuid
from datetime import datetime
from mangum import Mangum
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tensorflow as tf
import logging
import tempfile
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

# Config
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
BUCKET_NAME = os.getenv('BUCKET_NAME')

# Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
logger.info("✅ Supabase client created successfully")

# ------------------------
# FIXED MODEL LOADING FOR TF 2.20 COMPATIBILITY
# ------------------------
def safe_load_model(model_path):
    """
    Load model with TensorFlow 2.20 compatibility fixes for BatchNormalization axis issue
    """
    try:
        # Method 1: Load without compilation (most common fix)
        logger.info(f"🔄 Attempting to load model: {model_path}")
        model = load_model(model_path, compile=False)
        
        # Recompile the model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy', 
            metrics=['accuracy']
        )
        logger.info("✅ Model loaded successfully (compile=False method)")
        return model
        
    except Exception as e1:
        logger.warning(f"⚠️ Standard loading failed: {e1}")
        
        try:
            # Method 2: Custom BatchNormalization fix
            def fix_batch_normalization(*args, **kwargs):
                # Fix axis parameter if it's a list instead of int
                if 'axis' in kwargs and isinstance(kwargs['axis'], list):
                    kwargs['axis'] = kwargs['axis'][0] if len(kwargs['axis']) == 1 else -1
                return tf.keras.layers.BatchNormalization(*args, **kwargs)
            
            model = load_model(
                model_path,
                custom_objects={'BatchNormalization': fix_batch_normalization}
            )
            logger.info("✅ Model loaded with BatchNormalization fix")
            return model
            
        except Exception as e2:
            logger.error(f"❌ All loading methods failed: {e2}")
            raise Exception(f"Failed to load model: {e2}")

# Load your 4-grade model with the fix
try:
    model = safe_load_model("mv2_tobacco_model.keras")
    logger.info("✅ 4-grade tobacco model loaded successfully")
    logger.info(f"📊 Model input shape: {model.input_shape}")
    logger.info(f"🧠 TensorFlow version: {tf.__version__}")
except Exception as e:
    logger.error(f"❌ Model loading failed: {e}")
    raise

app = FastAPI(
    title="ToboGrade API - 4 Grade System",
    description="Tobacco Leaf Quality Detection API (4 Grades) - TF 2.20 Compatible",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        'https://tobograde-tobacco-grading-system.netlify.app/',
        "https://leafgrade-tobacco-grading-system.netlify.app",  # Fixed: removed trailing slash
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Updated class list for 4 grades
TOBACCO_CLASSES = ['Grade_1', 'Grade_4', 'Grade_5', 'RedThargu']

def preprocess_image_enhanced(temp_path, target_size=(640, 640)):
    """
    Enhanced preprocessing for MobileNetV2 model
    """
    try:
        # Load and resize image
        img = image.load_img(temp_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Apply MobileNetV2 preprocessing (better than simple division by 255)
        img_array = preprocess_input(img_array)
        
        logger.info(f"📷 Image preprocessed: shape {img_array.shape}")
        return img_array
    except Exception as e:
        logger.error(f"❌ Image preprocessing failed: {e}")
        raise

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    temp_path = None
    try:
        logger.info(f"📤 Running 4-grade prediction for: {file.filename}")

        # Validate file
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Save temporary file
        file_bytes = await file.read()
        if len(file_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
            
        filename = f"{uuid.uuid4()}_{file.filename}"
        temp_path = os.path.join(tempfile.gettempdir(), filename)
        
        with open(temp_path, "wb") as f:
            f.write(file_bytes)

        # Enhanced preprocessing
        img_array = preprocess_image_enhanced(temp_path)

        # Predict using 4-class model
        logger.info("🔮 Running model prediction...")
        preds = model.predict(img_array, verbose=0)
        if isinstance(preds, list):
            preds = preds[0]
        preds_batch = preds[0]  # first (and only) image
        class_index = int(np.argmax(preds_batch))
        confidence = float(np.max(preds_batch) * 100)
        result = TOBACCO_CLASSES[class_index]

        # Get all class probabilities
        all_probabilities = {
            TOBACCO_CLASSES[i]: round(float(preds[0][i] * 100), 2)
            for i in range(len(TOBACCO_CLASSES))
        }

        logger.info(f"🎯 4-Grade Prediction: {result} ({confidence:.2f}%)")

        return {
            "success": True,
            "filename": file.filename,
            "result": result,
            "confidence": round(confidence, 2),
            "confidence_threshold_met": confidence >= 70.0,
            "all_probabilities": all_probabilities,
            "message": "4-grade prediction completed successfully",
            "available_grades": TOBACCO_CLASSES,
            "total_classes": len(TOBACCO_CLASSES),
            "model_version": "tf2.20-mobilenetv2",
            "processed_at": datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.info(f"🧹 Cleaned up: {filename}")
            except Exception as cleanup_error:
                logger.warning(f"⚠️ Cleanup failed: {cleanup_error}")

@app.post("/save")
async def save_to_history(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    result: str = Form(...),
    confidence: float = Form(...)
):
    try:
        logger.info(f"💾 Saving 4-grade result for user {user_id}")

        # Validate inputs
        if result not in TOBACCO_CLASSES:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid grade. Must be one of: {TOBACCO_CLASSES}"
            )
            
        if not (0 <= confidence <= 100):
            raise HTTPException(status_code=400, detail="Invalid confidence value")

        file_bytes = await file.read()
        if len(file_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file")

        # Generate safe filename
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        file_extension = os.path.splitext(file.filename)[1].lower()
        safe_filename = f"{timestamp}_{uuid.uuid4()}{file_extension}"
        folder_path = f"user/{user_id}/predictions/{safe_filename}"

        # Upload to Supabase storage
        logger.info(f"📤 Uploading to: {folder_path}")
        upload_response = supabase.storage.from_(BUCKET_NAME).upload(
            folder_path,
            file_bytes,
            file_options={
                "content-type": file.content_type or "image/jpeg",
                "cache-control": "3600"
            }
        )

        if hasattr(upload_response, "error") and upload_response.error:
            raise Exception(f"Storage upload failed: {upload_response.error}")

        image_url = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET_NAME}/{folder_path}"

        # Insert record into DB
        insert_data = {
            "user_id": user_id,
            "image_url": image_url,
            "result": result,  # Changed from 'result' to 'predicted_grade'
            "confidence": confidence,
            "status": "processed",
            "processed_at": datetime.utcnow().isoformat(),
            "created_at": datetime.utcnow().isoformat(),
            "model_version": "tf2.20-mobilenetv2"
        }
        
        logger.info("💾 Saving to database...")
        insert_response = supabase.table("upload_history").insert(insert_data).execute()

        if hasattr(insert_response, "error") and insert_response.error:
            raise Exception(f"Database insert failed: {insert_response.error}")

        logger.info(f"✅ Successfully saved for user {user_id}")
        
        return {
            "success": True,
            "message": "Saved to history successfully",
            "image_url": image_url,
            "storage_path": folder_path,
            "result": result,
            "confidence": confidence,
            "user_id": user_id,
            "model_version": "tf2.20-mobilenetv2",
            "saved_at": datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Save error: {e}")
        raise HTTPException(status_code=500, detail=f"Save failed: {str(e)}")

@app.get("/health")
async def health_check():
    try:
        # Test model with dummy prediction
        dummy_input = np.random.random((1, 640, 640, 3))
        dummy_input = preprocess_input(dummy_input)
        test_pred = model.predict(dummy_input, verbose=0)
        
        return {
            "status": "healthy", 
            "timestamp": datetime.utcnow().isoformat(),
            "model_loaded": model is not None,
            "model_functional": test_pred is not None,
            "model_input_shape": str(model.input_shape),
            "available_grades": TOBACCO_CLASSES,
            "total_classes": len(TOBACCO_CLASSES),
            "model_version": "tf2.20-mobilenetv2",
            "tensorflow_version": tf.__version__,
            "api_version": "2.1.0"
        }
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@app.get("/grades")
async def get_available_grades():
    """Get currently supported tobacco grades"""
    return {
        "current_grades": TOBACCO_CLASSES,
        "total_current": len(TOBACCO_CLASSES),
        "future_grades": ["BlackThargu", "SemiGreen", "Bright_green"],
        "total_planned": 7,
        "model_info": {
            "architecture": "MobileNetV2",
            "input_size": "640x640",
            "tensorflow_version": tf.__version__,
            "model_version": "tf2.20-mobilenetv2"
        },
        "status": "4-grade system active and compatible"
    }

@app.get("/")
async def root():
    """API root with information"""
    return {
        "message": "ToboGrade API - TensorFlow 2.20 Compatible",
        "version": "2.1.0", 
        "tensorflow_version": tf.__version__,
        "model_classes": len(TOBACCO_CLASSES),
        "endpoints": ["/predict", "/save", "/health", "/grades", "/docs"]
    }

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 ToboGrade API starting up (TF 2.20 compatible)")
    logger.info(f"🧠 TensorFlow: {tf.__version__}")
    logger.info(f"🎯 Classes: {TOBACCO_CLASSES}")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🔄 Shutting down...")
    tf.keras.backend.clear_session()
    logger.info("✅ TensorFlow session cleared")

handler = Mangum(app)
