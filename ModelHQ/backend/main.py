from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import cv2
from pydantic import BaseModel
import uvicorn

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Models
breast_cancer_model = tf.keras.models.load_model("./models/Breast_cancer_new.h5")
stock_model = tf.keras.models.load_model("./models/GOOG.h5")

# Define input schemas
class StockInput(BaseModel):
    stock_symbol: str
    days: int = 10

@app.get("/")
def home():
    return {"message": "ML Model API is running!"}

@app.post("/predict/breast_cancer")
async def predict_breast_cancer(file: UploadFile = File(...)):
    # Read and preprocess image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Make prediction
    prediction = breast_cancer_model.predict(img)[0][0]
    predicted_class = "Malignant" if prediction > 0.5 else "Benign"
    confidence = float(prediction if prediction > 0.5 else 1 - prediction)
    
    return {
        "prediction": predicted_class,
        "confidence": confidence,
        "model": "breast_cancer"
    }

@app.post("/predict/stock")
def predict_stock(data: StockInput):
    mock_predictions = [100 + i*5 for i in range(data.days)]
    return {
        "model": "stock",
        "stock": data.stock_symbol,
        "predictions": mock_predictions
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)