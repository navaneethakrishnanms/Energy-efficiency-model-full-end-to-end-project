import streamlit as st
import pickle
import numpy as np


with open("model.sav", "rb") as f:
    model = pickle.load(f)

st.title("ML Prediction App 🚀")
st.write("Enter feature values to make a prediction:")

X1 = st.number_input("Enter X1:", value=0.0)
X2 = st.number_input("Enter X2:", value=0.0)
X3 = st.number_input("Enter X3:", value=0.0)
X6 = st.number_input("Enter X6:", value=0.0)
X7 = st.number_input("Enter X7:", value=0.0)
X8 = st.number_input("Enter X8:", value=0.0)


features = np.array([[X1, X2, X3, X6, X7, X8]])

if st.button("Predict"):
    prediction = model.predict(features)
    st.success(f"Predicted Y1: {prediction[0]}")
