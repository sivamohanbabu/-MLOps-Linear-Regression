import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# App Title
st.title("Advertisement Sales Prediction")

st.write("Enter the advertisement budget to predict sales.")

# Inputs
tv = st.number_input("TV Advertising Budget", min_value=0.0)
radio = st.number_input("Radio Advertising Budget", min_value=0.0)
newspaper = st.number_input("Newspaper Advertising Budget", min_value=0.0)

# Prediction Button
if st.button("Predict Sales"):
    input_data = np.array([[tv, radio, newspaper]])
    prediction = model.predict(input_data)

    st.success(f"📈 Predicted Sales: {prediction[0]:.2f}")
