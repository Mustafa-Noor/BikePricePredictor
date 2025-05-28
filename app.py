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
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY') or st.secrets["GOOGLE_API_KEY"]["key"]
if not GOOGLE_API_KEY:
    st.error("Please set your GOOGLE_API_KEY in the Streamlit secrets")
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

# Custom CSS for biker theme
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #FF6B35, #F7931E);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
        text-align: center;
    }
    .stButton > button {
        background-color: #FF6B35;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .prediction-box {
        background-color: #1E1E1E;
        color: #FF6B35;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #FF6B35;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🏍️ BIKE PRICE PREDICTOR 🏍️</h1>', unsafe_allow_html=True)
st.markdown("### *Rev up your selling game with AI-powered bike valuations!*")
st.write("Get accurate price predictions and expert riding insights for your machine! 🔥")

# Create two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔧 Enter Your Bike's Specs")
    
    # Input fields with biker terminology
    bike_name = st.selectbox("🏍️ Your Ride (Brand)", sorted(bike_le.classes_))
    year = st.number_input("📅 Model Year", min_value=1990, max_value=2024, value=2020)
    seller_type = st.selectbox("👤 Seller Type", ["Individual", "Dealer"])
    owner = st.selectbox("🤝 Ownership History", ["1st owner", "2nd owner", "3rd owner", "4th owner", "5th owner and above"])
    km_driven = st.number_input("🛣️ Odometer Reading (KMs)", min_value=0, value=10000)
    ex_showroom_price = st.number_input("💰 Ex-showroom Price (Rs)", min_value=0, value=100000)

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
    if st.button("🚀 GET MY BIKE'S VALUE!", type="primary"):
        prediction = price_model.predict(features_scaled)[0]
        
        # Enhanced success message with biker theme
        st.markdown(f"""
        <div class="prediction-box">
            <h3>🎯 YOUR BIKE'S MARKET VALUE</h3>
            <h2>Rs {prediction:,.0f}</h2>
            <p><em>Time to make that sale! 🏁</em></p>
        </div>
        """, unsafe_allow_html=True)

        try:
            # Enhanced query for Gemini with biker terminology
            query = f"""
            You are a seasoned bike mechanic and riding expert with 20+ years in the Indian motorcycle market. 
            A rider wants to sell their machine with these specs:
            
            🏍️ BIKE DETAILS:
            - Machine: {bike_name}
            - Model Year: {year}
            - Seller: {seller_type}
            - Ownership: {owner}
            - Odometer: {km_driven:,} KMs
            - Original Price: Rs {ex_showroom_price:,.0f}
            - Predicted Value: Rs {prediction:,.0f}

            Give me your expert analysis in a biker's language. Cover:
            
            🔥 PRICE ANALYSIS:
            - Is Rs {prediction:,.0f} a solid deal or should they push for more?
            - How does the depreciation look for this machine?
            
            ⚙️ MARKET FACTORS:
            - What's affecting this bike's resale value right now?
            - How do the kilometers and ownership history impact the price?
            
            🏁 SELLING STRATEGY:
            - Best tips to get maximum value when selling
            - What paperwork and preparations are crucial?
            - Timing advice for the sale
            
            🛠️ CONDITION CHECK:
            - Key things buyers will inspect on this model
            - Maintenance records that add value
            
            Keep it real, practical, and in a rider-to-rider tone. Use Indian motorcycle market knowledge and include currency as "Rs" not "₹".
            """

            # Get response from Gemini
            response = gemini_model.generate_content(query)
            
            with col2:
                st.subheader("🔥 Expert Biker Insights")
                st.markdown("*From a veteran rider who knows the streets and the market*")
                st.write(response.text)
                
        except Exception as e:
            st.error(f"Error getting AI insights: {str(e)}")
            st.info("Please make sure you have set up your Google API key correctly in the .env file")



# Add footer with biker theme
st.markdown("---")
st.markdown("🏍️ **Built with passion by riders, for riders** | *Powered by AI & years of road experience* 🛣️")