from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from pydantic import BaseModel
import uvicorn
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import os
import xgboost as xgb
import joblib  # For loading the scaler
import pandas as pd

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
breast_cancer_model = tf.keras.models.load_model("./models/Breast Cancer/Breast_cancer_new.h5")
xgb_model = xgb.XGBClassifier()
xgb_model.load_model("./models/Heart disease/heart_disease_model.h5")
scaler = joblib.load("./models/Heart disease/scaler.pkl")  # Correct path to the scaler

# Define input schemas
class StockInput(BaseModel):
    stock_symbol: str
    days: int = 10

class HeartDiseaseInput(BaseModel):
    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

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

@app.post("/predict/heart_disease")
def predict_heart_disease(data: HeartDiseaseInput):
    try:
        print("Received data:", data.dict())  # Log the input data

        # Convert input data to a DataFrame
        input_dict = data.dict()
        input_df = pd.DataFrame([input_dict])

        # One-hot encode categorical features to match training
        input_df = pd.get_dummies(input_df, columns=['cp', 'restecg', 'slope', 'thal'], drop_first=True)

        # Load reference columns from training (X_train.columns saved during preprocessing)
        reference_cols = joblib.load("./models/Heart disease/reference_columns.pkl")

        # Add missing columns with 0
        for col in reference_cols:
            if col not in input_df.columns:
                input_df[col] = 0

        # Reorder columns to match training order
        input_df = input_df[reference_cols]

        # Scale numerical features
        numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

        # Make prediction
        prediction = xgb_model.predict(input_df)[0]
        probability = xgb_model.predict_proba(input_df)[0][1]

        return {
            "status": "success",
            "prediction": int(prediction),  # 1: Heart Disease, 0: No Heart Disease
            "probability": float(probability)
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.post("/predict/walmart_sales")
def predict_walmart_sales(data: dict):
    try:
        # Load the trained model
        model = xgb.Booster()
        model.load_model("./models/sales prediction/walmart_sales_model.h5")

        # Load the scaler
        scaler = joblib.load("./models/sales prediction/scaler.pkl")

        # Convert input data to a DataFrame
        input_df = pd.DataFrame([data])

        # Ensure all expected columns exist and are of correct type
        input_df = input_df.astype({
            "Store": int,
            "Temperature": float,
            "Fuel_Price": float,
            "CPI": float,
            "Unemployment": float,
            "Year": int,
            "WeekOfYear": int,
            "Store_Size_Category_Medium": int,
            "Store_Size_Category_Large": int,
            "IsHoliday_1": int
        })

        # Scale numerical features
        numerical_features = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'WeekOfYear']
        input_df[numerical_features] = scaler.transform(input_df[numerical_features])

        # Convert to DMatrix
        dinput = xgb.DMatrix(input_df)

        # Make prediction
        prediction = model.predict(dinput)[0]

        return {
            "status": "success",
            "prediction": float(prediction)
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.post("/predict/car_price")
def predict_car_price(data: dict):
    try:
        # Load the trained model
        model = joblib.load("./models/Car Price Prediction/car_price_model.pkl")

        # Convert input data to a DataFrame
        input_df = pd.DataFrame([data])

        # Make prediction
        prediction = model.predict(input_df)[0]

        return {
            "status": "success",
            "prediction": float(prediction)
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)