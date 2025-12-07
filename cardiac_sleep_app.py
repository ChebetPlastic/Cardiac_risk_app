# sleep_cardiac_app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
import random
from collections import deque

# --------------------------------------------------
# Load trained models and artefacts (RF ONLY)
# --------------------------------------------------
rf = joblib.load("sleep_cardiac_rf.pkl")
scaler = joblib.load("sleep_cardiac_scaler.pkl")
LABELS = ["High", "Low", "Medium"]
idx_to_label = {i: lab for i, lab in enumerate(LABELS)}

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
        "SPO2":      {"value": inputs["SPO2"],          "score": 0, "risk": "Normal", "highlight": ""},
        "HR":        {"value": inputs["HR"],            "score": 0, "risk": "Normal", "highlight": ""},
        "BMI":       {"value": inputs["BMI"],           "score": 0, "risk": "Normal", "highlight": ""},
        "ECG":       {"value": inputs["ECG_class"],     "score": 0, "risk": "Normal", "highlight": ""},
        "SleepDur":  {"value": inputs["SleepDur"],      "score": 0, "risk": "Normal", "highlight": ""},
        "DeepSleep": {"value": inputs["DeepSleep"],     "score": 0, "risk": "Normal", "highlight": ""},
        "RemSleep":  {"value": inputs["RemSleep"],      "score": 0, "risk": "Normal", "highlight": ""},
        "Wakeups":   {"value": inputs["Wakeups"],       "score": 0, "risk": "Normal", "highlight": ""},
        "Snoring":   {"value": inputs["Snoring_score"], "score": 0, "risk": "Normal", "highlight": ""},
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

    # SPO2
    if spo2 >= 96:
        pass
    elif spo2 >= 94:
        results["SPO2"].update({"score": 1, "risk": "Low", "highlight": "⚠️"})
    elif spo2 >= 92:
        results["SPO2"].update({"score": 2, "risk": "Medium", "highlight": "🔶"})
    else:
        results["SPO2"].update({"score": 3, "risk": "High", "highlight": "🔴"})

    # HR
    if hr <= 40:
        results["HR"].update({"score": 3, "risk": "High", "highlight": "🆘"})
    elif hr <= 50:
        results["HR"].update({"score": 1, "risk": "Low", "highlight": "⚠️"})
    elif hr <= 90:
        pass
    elif hr <= 110:
        results["HR"].update({"score": 1, "risk": "Mild", "highlight": "⚠️"})
    elif hr <= 130:
        results["HR"].update({"score": 2, "risk": "Medium", "highlight": "🔶"})
    else:
        results["HR"].update({"score": 3, "risk": "High", "highlight": "🆘"})

    # BMI
    if bmi < 18.5:
        results["BMI"].update({"score": 3, "risk": "High", "highlight": "🔴"})
    elif bmi < 25:
        pass
    elif bmi < 30:
        results["BMI"].update({"score": 1, "risk": "Low", "highlight": "⚠️"})
    elif bmi < 40:
        results["BMI"].update({"score": 2, "risk": "Medium", "highlight": "🔶"})
    else:
        results["BMI"].update({"score": 3, "risk": "High", "highlight": "🔴"})

    # ECG
    if ecg == 3:
        results["ECG"].update({"score": 3, "risk": "High", "highlight": "🔴"})
    elif ecg == 1:
        results["ECG"].update({"score": 1, "risk": "Low", "highlight": "⚠️"})

    # Sleep duration
    if 7 <= sleep <= 9:
        pass
    elif 6 <= sleep < 7 or 9 < sleep <= 10:
        results["SleepDur"].update({"score": 1, "risk": "Low", "highlight": "⚠️"})
    elif 5 <= sleep < 6 or sleep > 10:
        results["SleepDur"].update({"score": 2, "risk": "Medium", "highlight": "🔶"})
    else:
        results["SleepDur"].update({"score": 3, "risk": "High", "highlight": "🔴"})

    # Deep sleep
    if deep >= 3:
        pass
    elif 2 <= deep < 3:
        results["DeepSleep"].update({"score": 1, "risk": "Low", "highlight": "⚠️"})
    elif 1 <= deep < 2:
        results["DeepSleep"].update({"score": 2, "risk": "Medium", "highlight": "🔶"})
    else:
        results["DeepSleep"].update({"score": 3, "risk": "High", "highlight": "🔴"})

    # REM
    if 1.5 <= rem <= 3.5:
        pass
    elif 1.0 <= rem < 1.5 or 3.5 < rem <= 4.5:
        results["RemSleep"].update({"score": 1, "risk": "Low", "highlight": "⚠️"})
    elif 0.5 <= rem < 1.0 or rem > 4.5:
        results["RemSleep"].update({"score": 2, "risk": "Medium", "highlight": "🔶"})
    else:
        results["RemSleep"].update({"score": 3, "risk": "High", "highlight": "🔴"})

    # Wakeups
    if wakeups <= 0:
        pass
    elif wakeups == 1:
        results["Wakeups"].update({"score": 1, "risk": "Low", "highlight": "⚠️"})
    elif wakeups == 2:
        results["Wakeups"].update({"score": 2, "risk": "Medium", "highlight": "🔶"})
    else:
        results["Wakeups"].update({"score": 3, "risk": "High", "highlight": "🔴"})

    # Snoring_score
    if snore_score >= 2:
        results["Snoring"].update({"score": 2, "risk": "Medium", "highlight": "🔶"})

    vital_params = ["SPO2", "HR", "BMI", "ECG"]
    sleep_params = ["SleepDur", "DeepSleep", "RemSleep", "Wakeups", "Snoring"]

    vital_score = sum(results[p]["score"] for p in vital_params)
    sleep_score = sum(results[p]["score"] for p in sleep_params)
    total_score = vital_score + sleep_score

    red_flag = any(results[p]["score"] == 3 for p in vital_params)

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

    if vital_band in ["Medium", "High"] and sleep_score >= 3:
        risk_label = "High_Severe"
    elif vital_band in ["Low", "Low-Medium"] and sleep_score >= 3:
        risk_label = "Low-Medium_Sleep"
    else:
        risk_label = vital_band

    abnormal_params = [p for p in results if results[p]["score"] > 0]
    if len(abnormal_params) == 1:
        param = abnormal_params[0]
        results[param]["highlight"] = "! ISOLATED ABNORMALITY❗ → " + results[param]["highlight"]

    return {
        "component_scores": results,
        "vital_score": vital_score,
        "sleep_score": sleep_score,
        "total_score": total_score,
        "risk_label": risk_label
    }

# --------------------------------------------------
# RF prediction helpers
# --------------------------------------------------
def prepare_feature_vector(inputs):
    x = np.array([inputs[f] for f in FEATURE_ORDER], dtype=float).reshape(1, -1)
    x_scaled = scaler.transform(x)
    return x_scaled

def predict_rf(inputs):
    x_scaled = prepare_feature_vector(inputs)
    idx = rf.predict(x_scaled)[0]
    return idx_to_label[idx]

# --------------------------------------------------
# Wearable-style simulation
# --------------------------------------------------
def generate_wearable_sample(bmi):
    return {
        "HR": random.randint(50, 150),
        "SPO2": random.uniform(90, 100),
        "BMI": bmi,
        "SleepDur": random.uniform(4.0, 9.5),
        "DeepSleep": random.uniform(0.5, 4.0),
        "RemSleep": random.uniform(0.3, 4.0),
        "Wakeups": random.choice([0, 1, 2, 3]),
        "Snoring_score": random.choice([0, 2]),
        "ECG_class": random.choice([0, 0, 0, 3])
    }

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
st.set_page_config(page_title="Sleep–Cardiac Risk Monitor", page_icon="❤️")
st.title("💓 Sleep–Cardiac Risk Monitor (RandomForest + Rules)")
st.caption("NEWS vitals + sleep rules with a RandomForest classifier. LSTM removed in this lightweight deployment.")

tab_manual, tab_sim = st.tabs(["🔢 Manual input", "📶 Wearable simulation"])

# Manual input
with tab_manual:
    st.subheader("Enter vitals and sleep metrics")

    c1, c2 = st.columns(2)
    with c1:
        hr = st.slider("Heart Rate (bpm)", 30, 180, 80)
        spo2 = st.slider("SpO₂ (%)", 80.0, 100.0, 96.0)
        bmi = st.slider("BMI", 15.0, 50.0, 27.0)
        ecg_label = st.selectbox("ECG classification", ["Normal", "Borderline", "Abnormal"])
    with c2:
        sleepdur = st.slider("Total sleep (hours)", 3.0, 12.0, 7.0, 0.25)
        deepsleep = st.slider("Deep sleep (hours)", 0.0, 5.0, 2.0, 0.25)
        remsleep = st.slider("REM sleep (hours)", 0.0, 5.0, 1.5, 0.25)
        wakeups = st.slider("Night-time awakenings (0–3)", 0, 3, 1)
        snoring = st.selectbox("Snoring / OSA risk", ["No / minimal", "Yes / significant"])

    ecg_map = {"Normal": 0, "Borderline": 1, "Abnormal": 3}
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

    if st.button("Assess risk", type="primary"):
        rule = rule_based_risk(inputs)
        rf_label = predict_rf(inputs)

        st.markdown("### 🧮 Rule-based NEWS + sleep result")
        st.write(f"**Vital score:** {rule['vital_score']}  |  "
                 f"**Sleep score:** {rule['sleep_score']}  |  "
                 f"**Total score:** {rule['total_score']}")
        st.write(f"**Risk band (rules):** `{rule['risk_label']}`")

        st.markdown("### 🤖 RandomForest prediction")
        st.write(f"**RF class:** `{rf_label}`")

        st.info("This app uses the RandomForest model plus rule-based scoring. "
                "The LSTM component has been removed for easier deployment.")

        rows = []
        for name, info in rule["component_scores"].items():
            rows.append({
                "Parameter": name,
                "Value": round(info["value"], 2) if isinstance(info["value"], (int, float)) else info["value"],
                "Score": info["score"],
                "Risk": info["risk"],
                "Highlight": info["highlight"]
            })
        st.markdown("### 🔍 Parameter breakdown")
        st.dataframe(pd.DataFrame(rows))

# Simulation tab
with tab_sim:
    st.subheader("Wearable-style simulation (click to get a new reading)")

    if "sim_history" not in st.session_state:
        st.session_state.sim_history = deque(maxlen=20)
        st.session_state.sim_bmi = 27.0

    st.session_state.sim_bmi = st.slider(
        "Baseline BMI for simulated patient", 15.0, 50.0, float(st.session_state.sim_bmi), 0.5
    )

    if st.button("Generate new wearable reading"):
        sample = generate_wearable_sample(st.session_state.sim_bmi)
        rule = rule_based_risk(sample)
        rf_label = predict_rf(sample)

        st.session_state.sim_history.appendleft({
            "time": datetime.now().strftime("%H:%M:%S"),
            "HR": int(sample["HR"]),
            "SpO2": round(sample["SPO2"], 1),
            "BMI": round(sample["BMI"], 1),
            "ECG": sample["ECG_class"],
            "Total_score": rule["total_score"],
            "Rule_band": rule["risk_label"],
            "RF_pred": rf_label,
        })

    if st.session_state.sim_history:
        st.dataframe(pd.DataFrame(list(st.session_state.sim_history)))
    else:
        st.info("Click **Generate new wearable reading** to start the stream.")


