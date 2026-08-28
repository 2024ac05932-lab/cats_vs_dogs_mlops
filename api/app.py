import io
import os
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from prometheus_fastapi_instrumentator import Instrumentator   # 👈 Add this import

# Create the FastAPI app for Cats Vs Dogs
app = FastAPI(title="Cats vs Dogs Classifier", version="1.0")

# 👇 Add this line right after creating `app`
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# Rest of your code (model loading, endpoints, etc.)
model = None
MODEL_PATH = "models/cats_dogs_model.keras"
# Its a Assignment task for MLOPs
def load_my_model():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            model = load_model(MODEL_PATH)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            model = None
    else:
        print(f"Model not found at {MODEL_PATH}")
        model = None

@app.on_event("startup")
async def startup_event():
    load_my_model()

@app.get("/health")
async def health_check():
    if model is None:
        return {"status": "unhealthy", "message": "Model not loaded. Please train the model first."}
    return {"status": "healthy", "message": "Model is ready."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train first.")
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")
    try:
        contents = await file.read()
        img_bytes = io.BytesIO(contents)
        img = load_img(img_bytes, target_size=(128, 128))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)
        score = float(prediction[0][0])
        if score > 0.5:
            label = "Dog"
            confidence = score
        else:
            label = "Cat"
            confidence = 1 - score
        return {
            "prediction": label,
            "confidence": round(confidence * 100, 2),
            "raw_score": round(score, 4)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "Welcome to the Cats vs Dogs API!",
        "endpoints": {
            "/health": "Check API health",
            "/predict": "Upload an image to classify it (POST)",
            "/metrics": "Prometheus metrics"
        }
    }# Trigger CD
# Trigger full CD
# test after execution policy fix
# test after runner service installed
