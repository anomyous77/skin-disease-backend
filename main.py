from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "quick4_400_best_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

class_names = ["Acne", "Tinea", "Vitiligo", "Warts"]

recommendations = {
    "Acne": "Keep skin clean, avoid oily products, and do not squeeze pimples.",
    "Tinea": "Keep skin dry, avoid sharing towels, and consult a doctor for antifungal treatment.",
    "Vitiligo": "Protect skin from sunlight and consult a dermatologist for proper guidance.",
    "Warts": "Avoid scratching or touching warts and consult a doctor if they spread."
}

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((160, 160))
    img_array = np.array(image)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.get("/")
def home():
    return {"message": "Skin Disease Detection API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    processed_image = preprocess_image(image_bytes)

    prediction = model.predict(processed_image)
    predicted_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction)) * 100

    disease = class_names[predicted_index]

    if confidence < 60:
        warning = "Prediction is uncertain. Please consult a dermatologist."
    else:
        warning = "This is an AI-based result. Please consult a dermatologist for confirmation."

    return {
        "disease": disease,
        "confidence": round(confidence, 2),
        "recommendation": recommendations[disease],
        "warning": warning
    }