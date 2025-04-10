from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import cv2
from pydantic import BaseModel
import uvicorn
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import os

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load breast cancer model
breast_cancer_model = tf.keras.models.load_model("./models/Breast_cancer_new.h5")

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
    try:
        # Use GOOG.h5 model regardless of selected stock (for demo purposes)
        model_path = "./models/GOOG.h5"
        
        if not os.path.exists(model_path):
            return {
                "status": "error",
                "message": "GOOG prediction model not found"
            }
        
        model = tf.keras.models.load_model(model_path)
        
        # Get recent data for GOOG (since model was trained on GOOG data)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        stock_data = yf.download("GOOG", start=start_date, end=end_date)
        
        if stock_data.empty or len(stock_data) < 100:
            return {
                "status": "error",
                "message": "Insufficient historical data for GOOG"
            }
        
        # Preprocess data
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(stock_data[['Close']])
        last_100_days = scaled_data[-100:].reshape(1, 100, 1)
        
        # Generate predictions
        predicted_prices = []
        current_sequence = last_100_days.copy()
        
        for _ in range(data.days):
            pred = model.predict(current_sequence, verbose=0)[0][0]
            actual_price = float(scaler.inverse_transform([[pred]])[0][0])
            predicted_prices.append(actual_price)
            current_sequence = np.append(current_sequence[:,1:,:], [[[pred]]], axis=1)
        
        return {
            "status": "success",
            "stock": "GOOG",  # Always return GOOG as the predicted stock
            "predictions": predicted_prices
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)