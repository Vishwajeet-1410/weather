# weather_app.py 🌤️ Weather Type Predictor using .sav file

import streamlit as st
import numpy as np
import pickle
import os

# ✅ Full path to your saved model (.sav file)
MODEL_PATH = "C:/summerintenship machine learning/Spider FIles/app/weather.sav"

# 🌐 Streamlit Page Setup
st.set_page_config(page_title="🌤️ Weather Predictor", page_icon="🌦️", layout="centered")
st.title("🌦️ Weather Type Prediction App")
st.markdown("Enter the weather details below to predict the likely **WeSather Type** (e.g. Sunny, Rainy, Cloudy).")

try:
    # 🔄 Load the model and encoders
    with open(MODEL_PATH, "rb") as file:
        model_data = pickle.load(file)

    model = model_data['model']
    label_encoders = model_data['label_encoders']

    # ✅ Get class names from encoders
    season_classes = label_encoders['Season'].classes_
    weathertype_classes = label_encoders['WeatherType'].classes_

    # 🧾 User Inputs (9 numeric + 1 categorical)
    col1, col2 = st.columns(2)
    with col1:
        temperature = st.number_input("🌡️ Temperature (°C)", value=25.0)
        humidity = st.number_input("💧 Humidity (%)", value=60.0)
        pressure = st.number_input("📈 Pressure (hPa)", value=1013.0)
        uv_index = st.number_input("🔆 UV Index", value=5.0)
        visibility = st.number_input("👁️ Visibility (km)", value=10.0)
    with col2:
        dew_point = st.number_input("🧊 Dew Point (°C)", value=18.0)
        wind_speed = st.number_input("💨 Wind Speed (km/h)", value=12.0)
        cloud_cover = st.number_input("☁️ Cloud Cover (%)", value=30.0)
        wind_direction = st.number_input("🧭 Wind Direction (°)", value=90.0)
        season = st.selectbox("📅 Season", season_classes)

    # 🔄 Encode season
    season_encoded = label_encoders['Season'].transform([season])[0]

    # 🧠 Prepare input data for prediction
    input_data = np.array([[temperature, humidity, pressure, uv_index,
                            visibility, dew_point, wind_speed, cloud_cover,
                            wind_direction, season_encoded]])

    # 🔍 Predict button
    if st.button("🔍 Predict Weather"):
        prediction_encoded = model.predict(input_data)[0]
        prediction_label = label_encoders['WeatherType'].inverse_transform([prediction_encoded])[0]
        st.success(f"🌤️ Predicted Weather Type: **{prediction_label}**")

except FileNotFoundError:
    st.error("❌ The model file was not found. Please check your path to `weather_model.sav`.")
except KeyError:
    st.error("❌ The `.sav` file is missing `model` or `label_encoders`. Ensure it’s properly saved.")
except Exception as e:
    st.error(f"⚠️ An unexpected error occurred: {e}")
