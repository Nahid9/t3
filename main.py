from fastapi import FastAPI, File, UploadFile, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom, RandomHeight, RandomWidth
from tensorflow.keras.utils import custom_object_scope
import numpy as np
from PIL import Image
import base64
import io
import os
import tensorflow as tf

# Initialize FastAPI
app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="static")

# Model configurations
CROP_MODELS = {
    "corn": {
        "model_path": "mobilenet_corn.h5",
        "class_names": ["Corn___Common_Rust", "Corn___Gray_Leaf_Spot", "Corn___Healthy", "Corn___Leaf_Blight"],
        "display_name": "🌽 Corn Disease Classifier"
    },
    "potato": {
        "model_path": "mobilenet_potato.h5",
        "class_names": ["Potato___Early_Blight", "Potato___Healthy", "Potato___Late_Blight"],
        "display_name": "🥔 Potato Disease Classifier"
    },
    "rice": {
        "model_path": "mobilenet_rice.h5",
        "class_names": ["Rice___Brown_Spot", "Rice___Healthy", "Rice___Hispa", "Rice___Leaf_Blast"],
        "display_name": "🌿 Rice Disease Classifier"
    },
    "wheat": {
        "model_path": "mobilenet_wheat.h5",
        "class_names": ["Wheat___Brown_Rust", "Wheat___Healthy", "Wheat___Yellow_Rust"],
        "display_name": "🌾 Wheat Disease Classifier"
    }
}

# Image preprocessing config
IMG_SIZE = (224, 224)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

# Custom objects for data augmentation layers
custom_objects = {
    "RandomFlip": RandomFlip,
    "RandomRotation": RandomRotation,
    "RandomZoom": RandomZoom,
    "RandomHeight": RandomHeight,
    "RandomWidth": RandomWidth
}

# Load all models at startup
models = {}
for crop, config in CROP_MODELS.items():
    with custom_object_scope(custom_objects):
        models[crop] = load_model(config["model_path"])
    CROP_MODELS[crop]["formatted_class_names"] = [
        name.replace("___", " ").replace("_", " ") for name in config["class_names"]
    ]

# Image preprocessing
def preprocess_image(image_file):
    img = Image.open(image_file).convert("RGB")
    img = img.resize(IMG_SIZE)
    img = np.array(img) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return np.expand_dims(img, axis=0)

# Home route
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Classifier UI
@app.get("/classifier/{crop_type}", response_class=HTMLResponse)
async def crop_classifier(request: Request, crop_type: str):
    if crop_type not in CROP_MODELS:
        return RedirectResponse(url="/")
    return templates.TemplateResponse(
        f"{crop_type}.html",
        {
            "request": request,
            "crop_type": crop_type,
            "display_name": CROP_MODELS[crop_type]["display_name"]
        }
    )

# Prediction route
@app.post("/predict/{crop_type}", response_class=HTMLResponse)
async def predict(request: Request, crop_type: str, file: UploadFile = File(...)):
    if crop_type not in CROP_MODELS:
        return RedirectResponse(url="/")

    try:
        # Read and preprocess image
        image_bytes = await file.read()
        img = preprocess_image(io.BytesIO(image_bytes))

        encoded_image = base64.b64encode(image_bytes).decode("utf-8")
        image_data_url = f"data:image/jpeg;base64,{encoded_image}"

        # Predict
        model = models[crop_type]
        preds = model.predict(img)
        class_idx = int(np.argmax(preds[0]))
        confidence = float(np.max(preds[0]))
        confidence_percent = round(confidence * 100)
        predicted_class = CROP_MODELS[crop_type]["formatted_class_names"][class_idx]

        return templates.TemplateResponse(
            f"{crop_type}.html",
            {
                "request": request,
                "crop_type": crop_type,
                "display_name": CROP_MODELS[crop_type]["display_name"],
                "prediction": predicted_class,
                "confidence": confidence_percent,
                "confidence_raw": confidence,
                "image_uploaded": True,
                "image_data_url": image_data_url 
            }
        )

    except Exception as e:
        return templates.TemplateResponse(
            f"{crop_type}.html",
            {
                "request": request,
                "crop_type": crop_type,
                "display_name": CROP_MODELS[crop_type]["display_name"],
                "error": f"Error: {str(e)}"
            }
        )
