import streamlit as st
import numpy as np
import joblib

# Load model, scaler, and label encoder
model = joblib.load('fish_health_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Page title
st.title("🐟 Fish Pond Health Prediction System")
st.markdown("Enter the water quality and fish data to predict pond health status.")

# Input form
temperature = st.number_input("Water Temperature (°C)", step=0.1)
dissolved_oxygen = st.number_input("Dissolved Oxygen (mg/L)", step=0.1)
ph = st.number_input("pH Level", step=0.1)
turbidity = st.number_input("Turbidity (NTU)", step=0.1)
avg_fish_weight = st.number_input("Average Fish Weight (g)", step=1.0)
survival_rate = st.number_input("Survival Rate (%)", step=0.1)
disease_cases = st.number_input("Disease Occurrence (Cases)", step=1)

low_oxygen_alert = st.selectbox("Low Oxygen Alert", ["No", "Yes"])
thermal_risk = st.selectbox("Thermal Risk Index", ["Low", "Medium", "High"])

# Encode categorical fields
low_oxygen_encoded = 1 if low_oxygen_alert == "Yes" else 0
thermal_risk_encoded = {"Low": 0, "Medium": 1, "High": 2}[thermal_risk]

# Predict on button click
if st.button("Predict Health Status"):
    input_data = np.array([[temperature, dissolved_oxygen, ph, turbidity,
                            avg_fish_weight, survival_rate, disease_cases,
                            low_oxygen_encoded, thermal_risk_encoded]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    predicted_label = label_encoder.inverse_transform(prediction)

    st.success(f"✅ Predicted Health Status: **{predicted_label[0]}**")
