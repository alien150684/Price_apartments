from fastapi import FastAPI, HTTPException
from api.validators import Apartment, District
import pandas as pd
import joblib
from catboost import CatBoostRegressor

app = FastAPI()

@app.get("/")
def read_root():
  return {"message": "Server is working!"}

# Импортируем модель
# model = CatBoostRegressor()
# model.load_model("resources\cat_boost.json", "json")  # load model
model = joblib.load('resources\lin_reg.pkl')

@app.get("/model-info/")
def get_model_info():
  return {
    "model_type": str(type(model)),
    "model_parameters": model.get_params()
  }

@app.post("/predict/")
def predict_price(apartment: Apartment):
  try:
    input_data = pd.DataFrame([apartment.dict()])
    # Импортируем трансформер
    custom_transformer = joblib.load('resources\preprocessor.pkl')
    transform_data = custom_transformer.transform(input_data)
    
    prediction = model.predict(transform_data)
    return {"prediction": float(prediction[0])}
  except Exception as e:
    raise HTTPException(status_code=400, detail=f"Error processing the prediction: {e}")