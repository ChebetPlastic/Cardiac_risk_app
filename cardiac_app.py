# cardiac_app.py

import streamlit as st
from datetime import datetime

# Set page configuration
st.set_page_config(page_title="Cardiac Risk Monitor", page_icon="❤️", layout="centered")

st.title("💓 Cardiac Risk Monitor (Wearable Prototype)")
st.markdown("Monitor a patient’s cardiac risk using simulated smart device data.")

# Simulated Input (can later be connected to wearable inputs)
bmi = st.slider("Patient BMI", 15.0, 45.0, 25.0)
spo2 = st.slider("Blood Oxygen Level (SpO2 %)", 80.0, 100.0, 96.0)
hr = st.slider("Heart Rate (bpm)", 30, 160, 75)
ecg = st.selectbox("ECG Classification", [0, 1, 3], index=0)

def calculate_risk(spo2, hr, ecg, bmi):
    score = 0

    if spo2 < 92:
        score += 3
    elif spo2 < 94:
        score += 2
    elif spo2 < 96:
        score += 1

    if hr > 130 or hr <= 40:
        score += 3
    elif hr > 110 or hr <= 50:
        score += 2
    elif hr > 90 or hr <= 60:
        score += 1

    if ecg == 3:
        score += 3

    if bmi >= 40 or bmi < 18.5:
        score += 3
    elif bmi >= 30:
        score += 2
    elif bmi >= 25:
        score += 1

    if score == 0:
        return score, "🟢 Normal"
    elif score <= 2:
        return score, "🟡 Low Risk"
    elif score <= 5:
        return score, "🟠 Medium Risk"
    else:
        return score, "🔴 High Risk"

score, level = calculate_risk(spo2, hr, ecg, bmi)

# Display the results
st.markdown(f"### 📊 Risk Score: **{score}**")
st.markdown(f"### 🔥 Risk Level: **{level}**")
st.caption(f"🕒 Assessed at: {datetime.now().strftime('%H:%M:%S')}")

# Recommendation messages
if score >= 6:
    st.error("🚨 High Risk: Immediate medical attention advised.")
elif score >= 3:
    st.warning("⚠️ Moderate Risk: Monitor closely.")
else:
    st.success("✅ Stable condition.")
