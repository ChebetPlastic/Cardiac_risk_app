
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from streamlit_autorefresh import st_autorefresh

# 1. Page Config
st.set_page_config(page_title="Sleep-Cardiac Risk Monitor", page_icon="❤️", layout="centered")

# 2. Auto-Refresh
count = st_autorefresh(interval=3 * 60 * 1000, key="data_refresh")

# 3. Logic Functions
def calculate_news_score(hr, spo2, deep_sleep, wakeups, snoring_cat):
    vital_score = 0
    if spo2 < 95: vital_score += 2
    if hr > 100 or hr < 50: vital_score += 1
    
    sleep_score = 0
    if deep_sleep < 1.0: sleep_score += 2
    elif deep_sleep < 2.5: sleep_score += 1
    
    if wakeups >= 2: sleep_score += 1
    
    if snoring_cat == "Significant": sleep_score += 2
    elif snoring_cat == "Moderate": sleep_score += 1
    
    total = vital_score + sleep_score
    
    if total <= 2: band = "Low"
    elif total <= 5: band = "Medium"
    else: band = "High"
    
    return vital_score, sleep_score, total, band

@st.cache_resource
def load_components():
    data = {"model": None, "scaler": None}
    if os.path.exists('sleep_cardiac_rf.pkl'):
        data["model"] = joblib.load('sleep_cardiac_rf.pkl')
    elif os.path.exists('rf_model.pkl'):
        data["model"] = joblib.load('rf_model.pkl')
        
    if os.path.exists('sleep_cardiac_scaler.pkl'):
        data["scaler"] = joblib.load('sleep_cardiac_scaler.pkl')
    return data

# ==========================================
# 4. USER INTERFACE
# ==========================================

st.title("Sleep-Cardiac Risk Monitor")

mode = st.toggle("Enable Auto-Simulation Mode", value=False)

if mode:
    st.info(f"🔄 Auto-Mode Active. Refreshing every 3 mins.")
    hr = np.random.randint(60, 110)
    spo2 = np.random.randint(90, 99)
    deep_sleep = np.random.uniform(0.5, 3.0)
    rem_sleep = np.random.uniform(0.5, 2.5)
    sleep_dur = deep_sleep + rem_sleep + np.random.uniform(2.0, 4.0) 
    wakeups = np.random.randint(0, 4)
    snoring = np.random.choice(["No / minimal", "Moderate", "Significant"])
    ecg_status = "Normal"
else:
    with st.expander("Patient Vital Signs (HR / SpO2 / ECG)", expanded=False):
        hr = st.slider("Heart Rate (BPM)", 40, 140, 72)
        spo2 = st.slider("SpO2 (%)", 85, 100, 96)
        ecg_status = st.radio("ECG Status", ["Normal", "Abnormal"], horizontal=True)

    st.write("### Sleep Architecture Input")
    sleep_dur = st.slider("Total Sleep Duration (hrs)", 4.0, 12.0, 7.5, step=0.5)
    deep_sleep = st.slider("Deep sleep (hours)", 0.0, 5.0, 2.0, step=0.1)
    rem_sleep = st.slider("REM sleep (hours)", 0.0, 5.0, 1.5, step=0.1)
    wakeups = st.slider("Night-time awakenings (0-3+)", 0, 5, 1)
    snoring = st.selectbox("Snoring / OSA risk", ["No / minimal", "Moderate", "Significant"])

# --- Processing ---

if mode or st.button("Assess risk", type="primary", use_container_width=True):
    
    # 1. Calculate Rule-Based
    v_score, s_score, total_score, band = calculate_news_score(hr, spo2, deep_sleep, wakeups, snoring)
    
    # 2. Calculate ML Prediction
    components = load_components()
    model = components["model"]
    scaler = components["scaler"]
    ml_class = "Unknown"
    
    # --- DISPLAY RESULTS ---
    st.divider()
    
    # SECTION A: Rule-Based
    st.subheader("Rule-based NEWS + sleep result")
    color_map = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
    icon = color_map.get(band, "⚪")
    st.markdown(f"### {icon} {band} Risk")
    st.caption(f"Vital score: {v_score} | Sleep score: {s_score} | Total score: {total_score}")
    
    if band == "Medium" or band == "High":
        st.warning(f"Risk Band (rules): **{band}**")
    else:
        st.success(f"Risk Band (rules): **{band}**")

    # SECTION B: Random Forest
    st.write("")
    st.subheader("🤖 RandomForest prediction")
    
    if model:
        try:
            # Prepare Data
            ecg_val = 1 if ecg_status == "Abnormal" else 0
            snoring_map = {"No / minimal": 0, "Moderate": 5, "Significant": 10}
            snoring_val = snoring_map.get(snoring, 0)

            # Create DataFrame
            input_df = pd.DataFrame([[
                hr, spo2, ecg_val, sleep_dur, deep_sleep, rem_sleep, 
                wakeups, snoring_val, total_score
            ]], columns=['HeartRate', 'SpO2', 'ECG_Class', 'SleepDur', 
                         'DeepSleep', 'REM_Sleep', 'Wakeups', 'Snoring', 'TotalScore'])
            
            # Scale
            if scaler:
                final_input = scaler.transform(input_df)
            else:
                final_input = input_df
            
            # Predict
            raw_pred = model.predict(final_input)[0]
            
            # --- FINAL CORRECTED MAPPING ---
            # Based on your classes [1, 2, 3]
            label_map = {
                1: "Low Risk",
                2: "Medium Risk",
                3: "High Risk"
            }
            # Fallback for 0 just in case
            if raw_pred == 0: 
                ml_class = "Low Risk"
            else:
                ml_class = label_map.get(raw_pred, f"Class {raw_pred}")

            # Display Result
            if "High" in str(ml_class) or "Severe" in str(ml_class):
                st.error(f"RF result: {ml_class}")
            elif "Medium" in str(ml_class):
                st.warning(f"RF result: {ml_class}")
            else:
                st.success(f"RF result: {ml_class}")

        except Exception as e:
            st.error(f"Prediction Error: {e}")
    else:
        st.info("Model not loaded.")

    if mode:
        st.caption("Values simulated. Next update in 3 minutes.")
