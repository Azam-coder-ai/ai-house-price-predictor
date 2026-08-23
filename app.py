import streamlit as st
import pandas as pd
import numpy as np
import pickle

# 1. Page Title and Config
st.set_page_config(page_title="AI House Price Predictor", layout="centered")
st.title("🏠 Artificial Intelligence House Price Predictor")
st.write("Enter your house characteristics, and the AI will calculate its real market value.")

# 2. Load Saved Model and Scaler Assets
@st.cache_resource 
def load_assets():
    with open('HPP.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_assets()
except FileNotFoundError:
    st.error("Error: 'uy_bozori_model.pkl' or 'scaler.pkl' not found! Run 'main.py' first to save the model.")
    st.stop()

# 3. User Interface Form (English)
st.subheader("📋 House Characteristics")

area = st.number_input("House Area (Square Meters):", min_value=10, max_value=10000, value=100)
bedrooms = st.slider("Number of Bedrooms:", min_value=1, max_value=10, value=3)
bathrooms = st.slider("Number of Bathrooms:", min_value=1, max_value=5, value=1)
stories = st.slider("Number of Stories (Floors):", min_value=1, max_value=4, value=2)
parking = st.slider("Number of Parking Spaces:", min_value=0, max_value=3, value=1)

mainroad = st.selectbox("Is it on a Main Road?", ["Yes", "No"])
guestroom = st.selectbox("Does it have a Guestroom?", ["Yes", "No"])
basement = st.selectbox("Does it have a Basement?", ["Yes", "No"])
hotwaterheating = st.selectbox("Does it have Hot Water Heating?", ["Yes", "No"])
airconditioning = st.selectbox("Does it have Air Conditioning?", ["Yes", "No"])
prefarea = st.selectbox("Is it in a Preferred/Premium Neighborhood?", ["Yes", "No"])

furnishing = st.radio("Furnishing Status:", ["Fully Furnished", "Semi-Furnished", "Unfurnished"])

# 4. Predict Button Logic
if st.button("🤖 Predict Market Value"):
    
    # Text mapping helper
    def binary_map(value):
        return 1 if value == "Yes" else 0

    # Handle One-hot encoding map
    furnishing_semi = 1 if furnishing == "Semi-Furnished" else 0
    furnishing_unfurnished = 1 if furnishing == "Unfurnished" else 0

    # Compile array structure matching final_data.columns
        # 1. Update your input array to match the exact column order of final_data
    input_data = [
        0,  # dummy placeholder for price
        area, 
        bedrooms, 
        bathrooms, 
        stories, # Stories belongs right here after bathrooms!
        binary_map(mainroad), 
        binary_map(guestroom), 
        binary_map(basement),
        binary_map(hotwaterheating), 
        binary_map(airconditioning), 
        parking, 
        binary_map(prefarea),
        furnishing_semi, 
        furnishing_unfurnished
    ]

    # 2. Update your column layout structure to align perfectly
    columns_structure = [
        'price', 'area', 'bedrooms', 'bathrooms', 'stories', 
        'mainroad', 'guestroom', 'basement', 'hotwaterheating', 
        'airconditioning', 'parking', 'prefarea', 
        'furnishingstatus_semi-furnished', 'furnishingstatus_unfurnished'
    ]


    # Convert to DataFrame
    columns_structure = ['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea', 'furnishingstatus_semi-furnished', 'furnishingstatus_unfurnished']
    input_df = pd.DataFrame([input_data], columns=columns_structure)

    # 5. Apply the Scaler
    scaled_input = scaler.transform(input_df)
    
    # Drop the dummy price feature (index 0)
    X_input = np.delete(scaled_input, 0, axis=1)

    # 6. Run AI Prediction
    scaled_prediction = model.predict(X_input)

    # 7. Reverse scaling back to real values
    price_min = 1750000
    price_max = 9100000
    real_price_eur = (scaled_prediction * (price_max - price_min) + price_min)[0]

    # Display the final prediction
    st.success(f"🎯 AI Recommended Market Price: **EUR {real_price_eur:,.2f}**")
