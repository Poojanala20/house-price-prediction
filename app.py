import streamlit as st
import pickle
import numpy as np

# Load your trained model
import joblib
model = joblib.load('house_price_model.pkl')


st.title("House Price Prediction")

# Input features
bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3)
bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2)
sqft = st.number_input("Total Square Footage", min_value=500, max_value=10000, value=1500)

if st.button("Predict Price"):
    features = np.array([[bedrooms, bathrooms, sqft]])
    prediction = model.predict(features)
    st.success(f"Estimated House Price: ${prediction[0]:,.2f}")
