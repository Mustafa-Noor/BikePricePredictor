# Bike Price Predictor

A web application that predicts bike prices using machine learning and provides AI-powered insights using Google's Gemini API.

## Features

- Predict bike selling prices based on various features
- Get AI-powered insights about the predicted price
- User-friendly interface built with Streamlit
- Integration with Google's Gemini API for intelligent analysis

## Setup

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Get your Google API key:
   - Go to https://makersuite.google.com/app/apikey
   - Create a new API key
   - Create a `.env` file in the project root and add:
     ```
     GOOGLE_API_KEY=your_api_key_here
     ```

4. Train the model:
   ```bash
   python train_model.py
   ```

5. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage

1. Enter the bike details in the input form
2. Click "Predict Price" to get the predicted selling price
3. View AI-powered insights about the prediction

## Project Structure

- `train_model.py`: Script to train and save the ML model
- `app.py`: Streamlit web application
- `requirements.txt`: Project dependencies
- `BIKE DETAILS.csv`: Dataset used for training

## Technologies Used

- Python
- Scikit-learn
- Streamlit
- Google Gemini API
- Pandas
- NumPy 