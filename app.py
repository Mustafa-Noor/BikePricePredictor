import streamlit as st
import joblib
import google.generativeai as genai
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    st.error("Please set your GOOGLE_API_KEY in the .env file")
    st.stop()

try:
    genai.configure(api_key=GOOGLE_API_KEY)
    # Initialize the Gemini model
    gemini_model = genai.GenerativeModel(model_name='models/gemini-1.5-flash-latest')
except Exception as e:
    st.error(f"Error configuring Gemini API: {str(e)}")
    st.stop()

# Load the price prediction model and scaler
price_model = joblib.load('bike_price_model.joblib')
scaler = joblib.load('scaler.joblib')
bike_le = joblib.load('bike_label_encoder.joblib')

# Set page config
st.set_page_config(
    page_title="Bike Price Predictor",
    page_icon="🏍️",
    layout="wide"
)

# Title and description
st.title("🏍️ Bike Price Predictor")
st.write("Predict the selling price of your bike and get insights using AI!")

# Create two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("Enter Bike Details")
    
    # Input fields
    bike_name = st.selectbox("Bike Brand", sorted(bike_le.classes_))
    year = st.number_input("Year", min_value=1990, max_value=2024, value=2020)
    seller_type = st.selectbox("Seller Type", ["Individual", "Dealer"])
    owner = st.selectbox("Owner", ["1st owner", "2nd owner", "3rd owner", "4th owner", "5th owner and above"])
    km_driven = st.number_input("Kilometers Driven", min_value=0, value=10000)
    ex_showroom_price = st.number_input("Ex-showroom Price", min_value=0, value=100000)

    # Convert categorical variables
    bike_name_encoded = bike_le.transform([bike_name])[0]
    
    # Map owner values
    owner_map = {
        '1st owner': 1,
        '2nd owner': 2,
        '3rd owner': 3,
        '4th owner': 4,
        '5th owner and above': 5
    }
    owner_encoded = owner_map[owner]

    # Create feature array with exact same structure as training data
    features = np.array([[
        year,
        km_driven,
        ex_showroom_price,
        owner_encoded,
        bike_name_encoded,
        1 if seller_type == "Individual" else 0  # seller_type_Individual
    ]])

    # Scale features
    features_scaled = scaler.transform(features)

    # Make prediction
    if st.button("Predict Price"):
        prediction = price_model.predict(features_scaled)[0]
        st.success(f"Predicted Selling Price: ₹{prediction:,.2f}")

        try:
            # Prepare query for Gemini
            query = f"""
            I have a bike with the following details:
            - Bike Brand: {bike_name}
            - Year: {year}
            - Seller Type: {seller_type}
            - Owner: {owner}
            - Kilometers Driven: {km_driven}
            - Ex-showroom Price: ₹{ex_showroom_price:,.2f}
            - Predicted Selling Price: ₹{prediction:,.2f}

            Please provide insights about:
            1. Is this a good selling price?
            2. What factors might affect this price?
            3. Any recommendations for the seller?
            """

            # Get response from Gemini
            response = gemini_model.generate_content(query)
            
            with col2:
                st.subheader("AI Insights")
                st.write(response.text)
        except Exception as e:
            st.error(f"Error getting AI insights: {str(e)}")
            st.info("Please make sure you have set up your Google API key correctly in the .env file")

# Add footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and Gemini API") 