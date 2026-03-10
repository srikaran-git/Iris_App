import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load model
model = joblib.load("model.pkl")

# Flower labels
flowers = ["Setosa", "Versicolor", "Virginica"]

st.title("🌸 Iris Flower Prediction App")

st.write("Enter flower measurements below to predict the species.")

# Sidebar inputs
st.sidebar.header("Input Flower Measurements")

sepal_length = st.sidebar.slider("Sepal Length", 4.0, 8.0, 5.4)
sepal_width = st.sidebar.slider("Sepal Width", 2.0, 4.5, 3.4)
petal_length = st.sidebar.slider("Petal Length", 1.0, 7.0, 1.3)
petal_width = st.sidebar.slider("Petal Width", 0.1, 2.5, 0.2)

# Create input array
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# Predict
prediction = model.predict(input_data)[0]
probability = model.predict_proba(input_data)

st.subheader("Prediction")

st.success(f"🌼 Predicted Flower: **{flowers[prediction]}**")

if prediction == 0:
    st.image("iris_setosa.jpg", caption="Setosa")

elif prediction == 1:
    st.image("iris_versicolor.jpg", caption="Versicolor")

else:
    st.image("iris_virginica.jpg", caption="Virginica")

    

# Show probabilities
prob_df = pd.DataFrame(
    probability,
    columns=flowers
)

st.subheader("Prediction Probability")

st.bar_chart(prob_df.T)

# Show input data
st.subheader("Input Data")

input_df = pd.DataFrame({
    "Sepal Length": [sepal_length],
    "Sepal Width": [sepal_width],
    "Petal Length": [petal_length],
    "Petal Width": [petal_width]
})

st.table(input_df)