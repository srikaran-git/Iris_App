from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel

# Load model
model = joblib.load("model.pkl")

app = FastAPI()

# Define input format
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def home():
    return {"message": "ML API is running"}

@app.post("/predict")
def predict(data: IrisInput):
    input_data = np.array([[data.sepal_length,
                            data.sepal_width,
                            data.petal_length,
                            data.petal_width]])

    prediction = model.predict(input_data)[0]
    classes = ['Setosa', 'Versicolor', 'Virginica']
    return {"prediction": classes[prediction]}

    


    