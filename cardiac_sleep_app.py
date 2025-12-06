# cardiac_sleep_app.py
# Streamlit App: Sleep–Cardiac Risk Monitor (NEWS + Sleep + RF + LSTM)

import numpy as np
import pandas as pd
import streamlit as st
import joblib
from datetime import datetime
from collections import deque
import random
import os

# Try to import TF for LSTM; if not available, app will still run with RF + RULES
try:
    import tensorflow as tf
except ImportError:
    tf = None


# =========================================================
# STREAMLIT PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Sleep–Cardiac Risk Monitor",
    page_icon="❤️",
    layout="centered"
)

st.title("💓 Sleep–Cardiac Risk Monitor")
st.caption("NEWS vitals + Sleep scoring + RandomForest + LSTM prediction")


# =========================================================
# MODEL LOADING
# =========================================================

FEATURE_ORDER = [
    "HR", "SPO2", "BMI",
    "SleepDur", "DeepSleep", "RemSleep", "Wakeups",
    "Snoring_score", "ECG_class"
]

SEQ_LEN = 10  # LSTM synthetic sequence length


@st.cache_resource
def load_models():
    """Load RF, scaler, label map and optionally LSTM."""
    base = os.path.dirname(os.path.abspath(__file__))

    rf = joblib.load(os.path.join(base, "sleep_cardiac_rf.pkl"))
    scaler = joblib.load(os.path.join(base, "sleep_cardiac_scaler.pkl"))
    label_to_idx = joblib.load(os.path.join(base, "sleep_cardiac_label_map.pkl"))
    idx_to_label = {v: k for k, v in label_to_idx.items()}

    lstm_model = None
    lstm_path = os.path.join(base, "sleep_cardiac_lstm.keras")

    if tf is not None and os.path.exists(lstm_path):
        lstm_model = tf.keras.models.load_model(lstm_path)

    return rf, scaler, idx_to_label, lstm_model


# Attempt to load
try:
    rf_model, scaler, idx_to_label, lstm_model = load_models()
    MODELS_AVAILABLE = True
except Exception as e:
    MODELS_AVAILABLE = False
    st.error("❌ Could not load trained model files (.pkl, .keras). Upload them to GitHub.")
    st.exception(e)


# =========================================================
# RULE-BASED NEWS + SLEEP SCORING
# (same logic used during model training)
# =========================================================

def rule_based_risk(inputs):

    results = {
        "SPO2": {"value": inputs["SPO2"], "score": 0, "risk": "Normal", "highlight": ""},
        "HR": {"value": inputs["HR"], "score": 0, "risk": "Normal", "highlight": ""},
        "BMI": {"value": inputs["BMI"], "score": 0, "risk": "Normal", "highlight": ""},
        "ECG": {"value": inputs["ECG_class"], "score": 0, "risk": "Normal", "highlight": ""},
        "SleepDur": {"value": inputs["SleepDur"], "score": 0, "risk": "Normal", "highlight": ""},
        "DeepSleep": {"value": inputs["DeepSleep"], "score": 0, "risk": "Normal", "highlight": ""},
        "RemSleep": {"value": inputs["RemSleep"], "score": 0, "risk": "Normal", "highlight": ""},
        "Wakeups": {"value": inputs["Wakeups"], "score": 0, "risk": "Normal", "highlight": ""},
        "Snoring": {"value": inputs["Snoring_score"], "score": 0, "risk": "Normal", "highlight": ""},
    }

    spo2 = inputs["SPO2"]
    hr = inputs["HR"]
    bmi = inputs["BMI"]
    ecg = inputs["ECG_class"]
    sleep = inputs["SleepDur"]
    deep = inputs["DeepSleep"]
    rem = inputs["RemSleep"]
    wakeups = inputs["Wakeups"]
    snore = inputs["Snoring_score"]

    # SPO2
    if spo2 >= 96: pass
    elif spo2 >= 94: results["SPO2"].update({"score": 1, "risk": "Low", "highlight": "⚠️"})
    elif spo2 >= 92: results["SPO2"].update({"score": 2, "risk": "Medium", "highlight": "🔶"})
    else: results["SPO2"].update({"score": 3, "risk": "High", "highlight": "🔴"})

    # HR
    if hr <= 40: results["HR"].update({"score": 3, "risk": "High", "highlight": "🆘"})
    elif hr <= 50: results["HR"].update({"score": 1, "risk": "Low", "highlight": "⚠️"})
    elif hr <= 90: pass
    elif hr <= 110: results["HR"].update({"score": 1, "risk": "Mild", "highlight": "⚠️"})
    elif hr <= 130: results["HR"].update({"score": 2, "risk": "Medium", "highlight": "🔶"})
    else: results["HR"].update({"score": 3, "risk": "High", "highlight": "🆘"})

    # BMI
    if bmi < 18.5: results["BMI"].update({"score": 3, "risk": "High", "highlight": "🔴"})
    elif bmi < 25: pass
    elif bmi < 30: results["BMI"].update({"score": 1, "risk": "Low", "highlight": "⚠️"})
    elif bmi < 40: results["BMI"].update({"score": 2, "risk": "Medium", "highlight": "🔶"})
    else: results["BMI"].update({"score": 3, "risk": "High", "highlight": "🔴"})

    # ECG
    if ecg == 3: results["ECG"].update({"score": 3, "risk": "High", "highlight": "🔴"})
    elif ecg == 1: results["ECG"].update({"score": 1, "risk": "Low", "highlight": "⚠️"})

    # Sleep Duration
    if 7 <= sleep <= 9: pass
    elif 6 <= sleep < 7 or 9 < sleep <= 10:
        results["SleepDur"].update({"score": 1, "risk": "Low", "highlight": "⚠️"})
    elif 5 <= sleep < 6 or sleep > 10:
        results["SleepDur"].update({"score": 2, "risk": "Medium", "highlight": "🔶"})
    else:
        results["SleepDur"].update({"score": 3, "risk": "High", "highlight": "🔴"})

    # Deep Sleep
    if deep >= 3: pass
    elif deep >= 2: results["DeepSleep"].update({"score": 1, "risk": "Low", "highlight": "⚠️"})
    elif deep >= 1: results["DeepSleep"].update({"score": 2, "risk": "Medium", "highlight": "🔶"})
    else: results["DeepSleep"].update({"score": 3, "risk": "High", "highlight": "🔴"})

    # REM
    if 1.5 <= rem <= 3.5: pass
    elif 1 <= rem < 1.5 or 3.5 < rem <= 4.5:
        results["RemSleep"].update({"score": 1, "risk": "Low", "highlight": "⚠️"})
    elif 0.5 <= rem < 1 or rem > 4.5:
        results["RemSleep"].update({"score": 2, "risk": "Medium", "highlight": "🔶"})
    else:
        results["RemSleep"].update({"score": 3, "risk": "High", "highlight": "🔴"})

    # Wakeups
    if wakeups == 1: results["Wakeups"].update({"score": 1, "risk": "Low", "highlight": "⚠️"})
    elif wakeups == 2: results["Wakeups"].update({"score": 2, "risk": "Medium", "highlight": "🔶"})
    elif wakeups == 3: results["Wakeups"].update({"score": 3, "risk": "High", "highlight": "🔴"})

    # Snoring
    if snore >= 2: results["Snoring"].update({"score": 2, "risk": "Medium", "highlight": "🔶"})

    # Aggregate Scoring
    vital_params = ["SPO2", "HR", "BMI", "ECG"]
    sleep_params = ["SleepDur", "DeepSleep", "RemSleep", "Wakeups", "Snoring"]

    vital_score = sum(results[p]["score"] for p in vital_params)
    sleep_score = sum(results[p]["score"] for p in sleep_params)
    total_score = vital_score + sleep_score

    red_flag = any(results[p]["score"] == 3 for p in vital_params)

    # Vitals band (NEWS logic)
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

    # Sleep escalation
    if vital_band in ["Medium", "High"] and sleep_score >= 3:
        risk_label = "High_Severe"
    elif vital_band in ["Low", "Low-Medium"] and sleep_score >= 3:
        risk_label = "Low-Medium_Sleep"
    else:
        risk_label = vital_band

    # Isolated abnormality
    abnormal_params = [p for p in results if results[p]["score"] > 0]
    if len(abnormal_params) == 1:
        p = abnormal_params[0]
        results[p]["highlight"] = "ISOLATED❗ → " + results[p]["highlight"]

    return {
        "component_scores": results,
        "vital_score": vital_score,
        "sleep_score": sleep_score,
        "total_score": total_score,
        "risk_label": risk_label
    }


# =========================================================
# PREDICTION FUNCTIONS
# =========================================================
def prepare_vector(inputs):
    x = np.array([inputs[f] for f in FEATURE_ORDER]).reshape(1, -1)
    return scaler.transform(x)


def predict_rf(inputs):
    x = prepare_vector(inputs)
    idx = int(rf_model.predict(x)[0])
    return idx_to_label[idx]


def predict_lstm(inputs):
    if lstm_model is None or tf is None:
        return "Unavailable", 0.0
    x_scaled = prepare_vector(inputs)
    seq = np.repeat(x_scaled.reshape(1, 1, -1), SEQ_LEN, axis=1)
    probs = lstm_model.predict(seq, verbose=0)[0]
    idx = np.argmax(probs)
    return idx_to_label[idx], float(probs[idx])


# =========================================================
# WEARABLE SIMULATION
# =========================================================
def simulate_wearable(bmi):
    return {
        "HR": random.randint(50, 150),
        "SPO2": random.uniform(90, 100),
        "BMI": bmi,
        "SleepDur": random.uniform(4, 9),
        "DeepSleep": random.uniform(0.5, 4),
        "RemSleep": random.uniform(0.3, 4),
        "Wakeups": random.choice([0, 1, 2, 3]),
        "Snoring_score": random.choice([0, 2]),
        "ECG_class": random.choice([0, 0, 0, 3])
    }


# =========================================================
# UI TABS
# =========================================================
tab1, tab2 = st.tabs(["📝 Manual Input", "📡 Wearable Simulation"])

# ================================
# TAB 1 – Manual Input
# ================================
with tab1:
    st.subheader("Manual Entry")

    c1, c2 = st.columns(2)
    with c1:
        hr = st.slider("HR (bpm)", 30, 180, 80)
        spo2 = st.slider("SpO₂ (%)", 80.0, 100.0, 96.0)
        bmi = st.slider("BMI", 15.0, 50.0, 27.0)
        ecg_choice = st.selectbox("ECG", ["Normal", "Borderline", "Abnormal"])

    with c2:
        sleepdur = st.slider("Sleep Duration (h)", 3.0, 12.0, 7.0)
        deep = st.slider("Deep Sleep (h)", 0.0, 5.0, 2.0)
        rem = st.slider("REM Sleep (h)", 0.0, 5.0, 1.5)
        wakeups = st.slider("Night wakings", 0, 3, 1)
        snore = st.selectbox("Snoring", ["None", "Significant"])

    ecg_map = {"Normal": 0, "Borderline": 1, "Abnormal": 3}

    sample = {
        "HR": hr,
        "SPO2": spo2,
        "BMI": bmi,
        "SleepDur": sleepdur,
        "DeepSleep": deep,
        "RemSleep": rem,
        "Wakeups": wakeups,
        "Snoring_score": 0 if snore == "None" else 2,
        "ECG_class": ecg_map[ecg_choice]
    }

    if st.button("🩺 Assess Risk"):
        rules = rule_based_risk(sample)
        rf_pred = predict_rf(sample)
        lstm_pred, conf = predict_lstm(sample)

        st.write(f"### Rule-based Risk: `{rules['risk_label']}`")
        st.write(f"Total Score: **{rules['total_score']}**")

        st.write("### RF Prediction:", rf_pred)
        st.write(f"### LSTM Prediction: {lstm_pred} (conf {conf:.2f})")

        df_rows = []
        for p, info in rules["component_scores"].items():
            df_rows.append({
                "Parameter": p,
                "Value": round(info["value"], 2),
                "Score": info["score"],
                "Risk": info["risk"],
                "Flag": info["highlight"]
            })
        st.dataframe(pd.DataFrame(df_rows))


# ================================
# TAB 2 – Simulation
# ================================
with tab2:
    st.subheader("Wearable-style Simulation")

    if "sim_history" not in st.session_state:
        st.session_state.sim_history = deque(maxlen=30)
        st.session_state.sim_bmi = 27.0

    st.session_state.sim_bmi = st.slider(
        "Set BMI",
        15.0, 50.0,
        st.session_state.sim_bmi
    )

    if st.button("Generate Reading"):
        reading = simulate_wearable(st.session_state.sim_bmi)
        rules = rule_based_risk(reading)
        rf_pred = predict_rf(reading)
        lstm_pred, _ = predict_lstm(reading)

        st.session_state.sim_history.appendleft({
            "time": datetime.now().strftime("%H:%M:%S"),
            "HR": reading["HR"],
            "SpO2": round(reading["SPO2"], 1),
            "BMI": round(reading["BMI"], 1),
            "Rule": rules["risk_label"],
            "RF": rf_pred,
            "LSTM": lstm_pred
        })

    if st.session_state.sim_history:
        st.dataframe(pd.DataFrame(list(st.session_state.sim_history)))
    else:
        st.info("Click 'Generate Reading' to start.")



