# sleep_cardiac_app.py
# Simple Sleep–Cardiac Risk Monitor (Rules + RandomForest)

import streamlit as st
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from collections import deque
import random

# --------------------------------------------------
# Load trained models and artefacts (RF ONLY)
# --------------------------------------------------
rf = joblib.load("sleep_cardiac_rf.pkl")
scaler = joblib.load("sleep_cardiac_scaler.pkl")
label_to_idx = joblib.load("sleep_cardiac_label_map.pkl")

idx_to_label = {v: k for k, v in label_to_idx.items()}

FEATURE_ORDER = [
    "HR", "SPO2", "BMI",
    "SleepDur", "DeepSleep", "RemSleep", "Wakeups",
    "Snoring_score", "ECG_class"
]

# --------------------------------------------------
# Rule-based risk (same logic as training)
# --------------------------------------------------
def rule_based_risk(inputs):
    results = {
        "SPO2":      {"value": inputs["SPO2"],          "score": 0, "risk": "Normal"},
        "HR":        {"value": inputs["HR"],            "score": 0, "risk": "Normal"},
        "BMI":       {"value": inputs["BMI"],           "score": 0, "risk": "Normal"},
        "ECG":       {"value": inputs["ECG_class"],     "score": 0, "risk": "Normal"},
        "SleepDur":  {"value": inputs["SleepDur"],      "score": 0, "risk": "Normal"},
        "DeepSleep": {"value": inputs["DeepSleep"],     "score": 0, "risk": "Normal"},
        "RemSleep":  {"value": inputs["RemSleep"],      "score": 0, "risk": "Normal"},
        "Wakeups":   {"value": inputs["Wakeups"],       "score": 0, "risk": "Normal"},
        "Snoring":   {"value": inputs["Snoring_score"], "score": 0, "risk": "Normal"},
    }

    spo2 = inputs["SPO2"]
    hr = inputs["HR"]
    bmi = inputs["BMI"]
    ecg = inputs["ECG_class"]
    sleep = inputs["SleepDur"]
    deep = inputs["DeepSleep"]
    rem = inputs["RemSleep"]
    wakeups = inputs["Wakeups"]
    snore_score = inputs["Snoring_score"]

    # ----- SPO2 -----
    if spo2 >= 96:
        pass
    elif spo2 >= 94:
        results["SPO2"].update({"score": 1, "risk": "Low"})
    elif spo2 >= 92:
        results["SPO2"].update({"score": 2, "risk": "Medium"})
    else:
        results["SPO2"].update({"score": 3, "risk": "High"})

    # ----- HR -----
    if hr <= 40:
        results["HR"].update({"score": 3, "risk": "High"})
    elif hr <= 50:
        results["HR"].update({"score": 1, "risk": "Low"})
    elif hr <= 90:
        pass
    elif hr <= 110:
        results["HR"].update({"score": 1, "risk": "Low"})
    elif hr <= 130:
        results["HR"].update({"score": 2, "risk": "Medium"})
    else:
        results["HR"].update({"score": 3, "risk": "High"})

    # ----- BMI -----
    if bmi < 18.5:
        results["BMI"].update({"score": 3, "risk": "High"})
    elif bmi < 25:
        pass
    elif bmi < 30:
        results["BMI"].update({"score": 1, "risk": "Low"})
    elif bmi < 40:
        results["BMI"].update({"score": 2, "risk": "Medium"})
    else:
        results["BMI"].update({"score": 3, "risk": "High"})

    # ----- ECG -----
    if ecg == 3:
        results["ECG"].update({"score": 3, "risk": "High"})
    elif ecg == 1:
        results["ECG"].update({"score": 1, "risk": "Low"})

    # ----- Sleep duration -----
    if 7 <= sleep <= 9:
        pass
    elif 6 <= sleep < 7 or 9 < sleep <= 10:
        results["SleepDur"].update({"score": 1, "risk": "Low"})
    elif 5 <= sleep < 6 or sleep > 10:
        results["SleepDur"].update({"score": 2, "risk": "Medium"})
    else:
        results["SleepDur"].update({"score": 3, "risk": "High"})

    # ----- Deep sleep -----
    if deep >= 3:
        pass
    elif 2 <= deep < 3:
        results["DeepSleep"].update({"score": 1, "risk": "Low"})
    elif 1 <= deep < 2:
        results["DeepSleep"].update({"score": 2, "risk": "Medium"})
    else:
        results["DeepSleep"].update({"score": 3, "risk": "High"})

    # ----- REM -----
    if 1.5 <= rem <= 3.5:
        pass
    elif 1.0 <= rem < 1.5 or 3.5 < rem <= 4.5:
        results["RemSleep"].update({"score": 1, "risk": "Low"})
    elif 0.5 <= rem < 1.0 or rem > 4.5:
        results["RemSleep"].update({"score": 2, "risk": "Medium"})
    else:
        results["RemSleep"].update({"score": 3, "risk": "High"})

    # ----- Wakeups -----
    if wakeups <= 0:
        pass
    elif wakeups == 1:
        results["Wakeups"].update({"score": 1, "risk": "Low"})
    elif wakeups == 2:
        results["Wakeups"].update({"score": 2, "risk": "Medium"})
    else:
        results["Wakeups"].update({"score": 3, "risk": "High"})

    # ----- Snoring_score -----
    if snore_score >= 2:
        results["Snoring"].update({"score": 2, "risk": "Medium"})

    # Aggregate scores
    vital_params = ["SPO2", "HR", "BMI", "ECG"]
    sleep_params = ["SleepDur", "DeepSleep", "RemSleep", "Wakeups", "Snoring"]

    vital_score = sum(results[p]["score"] for p in vital_params)
    sleep_score = sum(results[p]["score"] for p in sleep_params)
    total_score = vital_score + sleep_score

    red_flag = any(results[p]["score"] == 3 for p in vital_params)

    # NEWS-style band for vitals only
    if vital_score >= 7:
        vital_band = "High"
    elif vital_score >= 5:
        vital_band = "Medium"
    elif red_flag:
        vital_band = "Low-Medium"
    elif vital_score == 0 and sleep_score == 0:
        vital_band = "Normal"
    else:
        vital_band = "Low"

    # Escalation with sleep
    if vital_band in ["Medium", "High"] and sleep_score >= 3:
        risk_label = "High_Severe"
    elif vital_band in ["Low", "Low-Medium"] and sleep_score >= 3:
        risk_label = "Low-Medium_Sleep"
    else:
        risk_label = vital_band

    # Collapsed label for ML (3 classes)
    if risk_label in ["Normal", "Low"]:
        ml_label = "Low"
    elif risk_label in ["Low-Medium", "Low-Medium_Sleep", "Medium"]:
        ml_label = "Medium"
    else:
        ml_label = "High"

    return {
        "component_scores": results,
        "vital_score": vital_score,
        "sleep_score": sleep_score,
        "total_score": total_score,
        "risk_label": risk_label,
        "ml_label": ml_label,
    }

# --------------------------------------------------
# RF helper
# --------------------------------------------------
def prepare_feature_vector(inputs):
    x = np.array([inputs[f] for f in FEATURE_ORDER], dtype=float).reshape(1, -1)
    x_scaled = scaler.transform(x)
    return x_scaled

def predict_rf(inputs):
    x_scaled = prepare_feature_vector(inputs)
    idx = int(rf.predict(x_scaled)[0])
    return idx_to_label[idx]

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
st.set_page_config(page_title="Sleep–Cardiac Risk Monitor", page_icon="💤")

st.title("💤 Sleep–Cardiac Risk Monitor")

st.markdown("Enter current **vital signs** and **sleep metrics** to estimate risk.")

# --- Input controls ---
c1, c2 = st.columns(2)

with c1:
    spo2 = st.slider("SpO₂ (%)", 80.0, 100.0, 96.0)
    hr = st.slider("Heart Rate (bpm)", 30, 180, 75)
    bmi = st.slider("BMI", 15.0, 50.0, 27.0)
    ecg_label = st.selectbox(
        "ECG Classification",
        ["0 = Normal", "1 = Borderline", "3 = Abnormal"],
        index=0,
    )

with c2:
    sleepdur = st.slider("Total sleep (hours)", 3.0, 12.0, 7.0, 0.25)
    deepsleep = st.slider("Deep sleep (hours)", 0.0, 5.0, 2.0, 0.25)
    remsleep = st.slider("REM sleep (hours)", 0.0, 5.0, 1.5, 0.25)
    wakeups = st.slider("Night-time awakenings (0–3)", 0, 3, 1)
    snoring = st.selectbox("Snoring / OSA risk", ["No / minimal", "Yes / significant"])

ecg_map = {"0 = Normal": 0, "1 = Borderline": 1, "3 = Abnormal": 3}
snore_score = 0 if snoring.startswith("No") else 2

inputs = {
    "HR": hr,
    "SPO2": spo2,
    "BMI": bmi,
    "SleepDur": sleepdur,
    "DeepSleep": deepsleep,
    "RemSleep": remsleep,
    "Wakeups": wakeups,
    "Snoring_score": snore_score,
    "ECG_class": ecg_map[ecg_label],
}

# --- Risk icon mapping (for display) ---
def risk_icon_and_colour(ml_label):
    if ml_label == "Low":
        return "🟢", "Low Risk"
    elif ml_label == "Medium":
        return "🟡", "Medium Risk"
    else:
        return "🔴", "High Risk"

# History in session_state
if "history" not in st.session_state:
    st.session_state.history = deque(maxlen=30)

# --- Assess risk button ---
if st.button("Assess risk", type="primary"):
    rule = rule_based_risk(inputs)
    rf_label = predict_rf(inputs)

    # Manual (rule-based) risk result – big icon like your other app
    icon, text = risk_icon_and_colour(rule["ml_label"])

    st.subheader("Manual Risk Result")
    st.markdown(f"### {icon} **{text}**")
    st.caption(f"Score: {rule['total_score']}  (Vital: {rule['vital_score']} | Sleep: {rule['sleep_score']})")

    # RandomForest prediction (simple line)
    st.subheader("RandomForest prediction")
    rf_icon, rf_text = risk_icon_and_colour(rf_label)
    st.markdown(f"**RF class:** {rf_icon} {rf_text}")

    # Save to history
    st.session_state.history.appendleft({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "spo2": spo2,
        "hr": hr,
        "bmi": bmi,
        "sleep_h": sleepdur,
        "deep_h": deepsleep,
        "rem_h": remsleep,
        "wakeups": wakeups,
        "snoring_score": snore_score,
        "rule_class": rule["ml_label"],
        "rf_class": rf_label,
    })

# --- Reading history table ---
st.markdown("### 📋 Reading History")
if st.session_state.history:
    df_hist = pd.DataFrame(list(st.session_state.history))
    st.dataframe(df_hist)
else:
    st.info("No readings yet. Enter values and press **Assess risk**.")


