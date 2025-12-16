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
def load_rf_model():
    # Try both potential filenames to be safe
    if os.path.exists('sleep_cardiac_rf.pkl'):
        return joblib.load('sleep_cardiac_rf.pkl')
    elif os.path.exists('rf_model.pkl'):
        return joblib.load('rf_model.pkl')
    return None

# ==========================================
# 4. USER INTERFACE
# ==========================================

st.title("Sleep-Cardiac Risk Monitor")

# Toggle for Auto/Manual
mode = st.toggle("Enable Auto-Simulation Mode", value=False)

if mode:
    st.info(f"🔄 Auto-Mode Active. Refreshing every 3 mins. (Refreshes: {count})")
    # Simulate data
    hr = np.random.randint(60, 110)
    spo2 = np.random.randint(90, 99)
    deep_sleep = np.random.uniform(0.5, 3.0)
    rem_sleep = np.random.uniform(0.5, 2.5)
    sleep_dur = deep_sleep + rem_sleep + np.random.uniform(2.0, 4.0) # Est total sleep
    wakeups = np.random.randint(0, 4)
    snoring = np.random.choice(["No / minimal", "Moderate", "Significant"])
    ecg_status = "Normal" 
else:
    # Manual Inputs
    with st.expander("Patient Vital Signs (HR / SpO2 / ECG)", expanded=False):
        hr = st.slider("Heart Rate (BPM)", 40, 140, 72)
        spo2 = st.slider("SpO2 (%)", 85, 100, 96)
        # RESTORED: ECG Input (Required for 9 features)
        ecg_status = st.radio("ECG Status", ["Normal", "Abnormal"], horizontal=True)

    st.write("### Sleep Architecture Input")
    # RESTORED: Sleep Duration (Required for 9 features)
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
    model = load_rf_model()
    ml_class = "Unknown"
    
    if model:
        try:
            # --- DATA PREPARATION (FIXED TO 9 FEATURES) ---
            
            # Map Categorical inputs to numbers for the model
            ecg_val = 1 if ecg_status == "Abnormal" else 0
            
            snoring_map = {"No / minimal": 0, "Moderate": 5, "Significant": 10}
            snoring_val = snoring_map.get(snoring, 0)

            # Create DataFrame with EXACTLY 9 columns in the likely order
            # We include 'TotalScore' (from rules) as the 9th feature
            input_data = pd.DataFrame([[
                hr,             # 1. HeartRate
                spo2,           # 2. SpO2
                ecg_val,        # 3. ECG_Class
                sleep_dur,      # 4. SleepDur
                deep_sleep,     # 5. DeepSleep
                rem_sleep,      # 6. REM_Sleep
                wakeups,        # 7. Wakeups
                snoring_val,    # 8. Snoring
                total_score     # 9. VitalScore / TotalScore (The engineered feature)
            ]], columns=['HeartRate', 'SpO2', 'ECG_Class', 'SleepDur', 'DeepSleep', 'REM_Sleep', 'Wakeups', 'Snoring', 'TotalScore'])
            
            ml_pred = model.predict(input_data)[0]
            ml_class = ml_pred
            
        except ValueError as e:
            st.error(f"Model Shape Error: {e}")
            ml_class = "Error"
        except Exception as e:
            st.error(f"Prediction Error: {e}")
            ml_class = "Error"
    else:
        ml_class = band # Fallback for demo
        
    # --- DISPLAY RESULTS (Black UI Style) ---
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
    
    if ml_class == "High":
        st.error(f"RF class: {ml_class}")
    elif ml_class == "Medium":
        st.warning(f"RF class: {ml_class}")
    elif ml_class == "Error":
        st.info("Could not load model. Showing Rule-Based result only.")
    else:
        st.success(f"RF class: {ml_class}")

    if mode:
        st.caption("Values simulated. Next update in 3 minutes.")
