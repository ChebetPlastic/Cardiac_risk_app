import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from streamlit_autorefresh import st_autorefresh

# 1. Page Config
st.set_page_config(page_title="Sleep-Cardiac Risk Monitor", page_icon="❤️", layout="centered")

# 2. Auto-Refresh (3 minutes)
count = st_autorefresh(interval=3 * 60 * 1000, key="data_refresh")

# 3. Logic Functions
def calculate_news_score(hr, spo2, deep_sleep, wakeups, snoring_cat):
    # --- Vital Score ---
    vital_score = 0
    if spo2 < 95: vital_score += 2
    if hr > 100 or hr < 50: vital_score += 1
    
    # --- Sleep Score ---
    sleep_score = 0
    if deep_sleep < 1.0: sleep_score += 2
    elif deep_sleep < 2.5: sleep_score += 1
    
    if wakeups >= 2: sleep_score += 1
    
    if snoring_cat == "Significant": sleep_score += 2
    elif snoring_cat == "Moderate": sleep_score += 1
    
    total = vital_score + sleep_score
    
    # Risk Band
    if total <= 2: band = "Low"
    elif total <= 5: band = "Medium"
    else: band = "High"
    
    return vital_score, sleep_score, total, band

@st.cache_resource
def load_components():
    """Loads both the Model and the Scaler"""
    data = {"model": None, "scaler": None}
    
    # Load Model
    if os.path.exists('sleep_cardiac_rf.pkl'):
        data["model"] = joblib.load('sleep_cardiac_rf.pkl')
    elif os.path.exists('rf_model.pkl'):
        data["model"] = joblib.load('rf_model.pkl')
        
    # Load Scaler (Crucial for correct predictions)
    if os.path.exists('sleep_cardiac_scaler.pkl'):
        data["scaler"] = joblib.load('sleep_cardiac_scaler.pkl')
        
    return data

# ==========================================
# 4. USER INTERFACE
# ==========================================

st.title("Sleep-Cardiac Risk Monitor")

mode = st.toggle("Enable Auto-Simulation Mode", value=False)

if mode:
    st.info(f"🔄 Auto-Mode Active. Refreshing every 3 mins. (Refreshes: {count})")
    # Simulate data
    hr = np.random.randint(60, 110)
    spo2 = np.random.randint(90, 99)
    deep_sleep = np.random.uniform(0.5, 3.0)
    rem_sleep = np.random.uniform(0.5, 2.5)
    sleep_dur = deep_sleep + rem_sleep + np.random.uniform(2.0, 4.0) 
    wakeups = np.random.randint(0, 4)
    snoring = np.random.choice(["No / minimal", "Moderate", "Significant"])
    ecg_status = "Normal"
else:
    # Manual Inputs
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
    
    if model:
        try:
            # --- DATA PREPARATION ---
            ecg_val = 1 if ecg_status == "Abnormal" else 0
            snoring_map = {"No / minimal": 0, "Moderate": 5, "Significant": 10}
            snoring_val = snoring_map.get(snoring, 0)

            # 1. Create Raw DataFrame
            # IMPORTANT: The order of columns here MUST match the order used during training!
            input_df = pd.DataFrame([[
                hr, spo2, ecg_val, sleep_dur, deep_sleep, rem_sleep, 
                wakeups, snoring_val, total_score
            ]], columns=['HeartRate', 'SpO2', 'ECG_Class', 'SleepDur', 
                         'DeepSleep', 'REM_Sleep', 'Wakeups', 'Snoring', 'TotalScore'])
            
            # 2. Scale the Data (if scaler exists)
            if scaler:
                # This translates "72" into the scaled format (e.g., 0.5)
                final_input = scaler.transform(input_df)
            else:
                st.warning("⚠️ Scaler file not found. Prediction may be inaccurate (Raw data used).")
                final_input = input_df
            
            # 3. Predict
            raw_pred = model.predict(final_input)[0]
            
            # --- MAPPING FIX ---
            label_map = {
                0: "Low Risk",
                1: "Medium Risk",
                2: "High Risk",
                3: "Severe Risk"
            }
            ml_class = label_map.get(raw_pred, f"Class {raw_pred}")
            
        except Exception as e:
            st.error(f"Prediction Error: {e}")
            ml_class = "Error"
    else:
        ml_class = band # Fallback
        
    # --- DISPLAY RESULTS ---
    st.divider()
    
    # SECTION A: Rule-Based Result
    st.subheader("Rule-based NEWS + sleep result")
    
    color_map = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
    icon = color_map.get(band, "⚪")
    
    st.markdown(f"### {icon} {band} Risk")
    st.caption(f"Vital score: {v_score} | Sleep score: {s_score} | Total score: {total_score}")
    
    if band == "Medium" or band == "High":
        st.warning(f"Risk Band (rules): **{band}**")
    else:
        st.success(f"Risk Band (rules): **{band}**")

    # SECTION B: Random Forest Result
    st.write("")
    st.subheader("🤖 RandomForest prediction")
    
    if "High" in str(ml_class) or "Severe" in str(ml_class) or "Class 3" in str(ml_class):
        st.error(f"RF result: {ml_class}")
    elif "Medium" in str(ml_class) or "Class 2" in str(ml_class):
        st.warning(f"RF result: {ml_class}")
    elif "Error" in str(ml_class):
        st.info("Model error. Check logs.")
    else:
        st.success(f"RF result: {ml_class}")

    if mode:
        st.caption("Values simulated. Next update in 3 minutes.")
