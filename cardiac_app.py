import streamlit as st
import pandas as pd
import sqlite3
import random
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# ========== Page Setup ==========
st.set_page_config(page_title="Cardiac Risk Monitor", page_icon="🫀", layout="centered")
st.title("💓 Cardiac Risk Monitor 2.0")
st.caption("Simulates auto-readings or allows manual input • Refreshes every 3 minutes")

refresh_count = st_autorefresh(interval=180000, key="auto-refresh")

# ========== Database Setup ==========
DB_NAME = "cardiac_monitor.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS readings
                 (timestamp TEXT, spo2 REAL, hr INTEGER, ecg INTEGER,
                  bmi REAL, total_risk INTEGER, risk_level TEXT)''')
    conn.commit()
    conn.close()

init_db()

# ========== Core Logic ==========
def calculate_risk(spo2, hr, ecg, bmi):
    score = 0
    if spo2 < 92: score += 3
    elif spo2 < 94: score += 2
    elif spo2 < 96: score += 1
    if hr > 130 or hr <= 40: score += 3
    elif hr > 110 or hr <= 50: score += 2
    elif hr > 90 or hr <= 60: score += 1
    if ecg == 3: score += 3
    if bmi >= 40 or bmi < 18.5: score += 3
    elif bmi >= 30: score += 2
    elif bmi >= 25: score += 1
    if score == 0: return score, "🟢 Normal"
    elif score <= 2: return score, "🟡 Low Risk"
    elif score <= 5: return score, "🟠 Medium Risk"
    else: return score, "🔴 High Risk"

def simulate_vitals():
    spo2 = round(random.uniform(89, 99), 1)
    hr = random.randint(45, 140)
    ecg = random.choices([0, 1, 3], weights=[0.8, 0.15, 0.05])[0]
    return spo2, hr, ecg

def save_reading(ts, spo2, hr, ecg, bmi, score, level):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('INSERT INTO readings VALUES (?, ?, ?, ?, ?, ?, ?)',
              (ts, spo2, hr, ecg, bmi, score, level))
    conn.commit()
    conn.close()

def load_history(limit=10):
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql(f'SELECT * FROM readings ORDER BY timestamp DESC LIMIT {limit}', conn)
    conn.close()
    return df

# ========== Mode Switch ==========
mode = st.radio("Select Input Mode", ["🔁 Auto Mode", "✍️ Manual Mode"], horizontal=True)

# ========== Shared BMI ==========
bmi = st.number_input("Set Patient BMI (used in risk scoring)", min_value=15.0, max_value=50.0, value=25.0, step=0.1)

# ========== Auto Mode ==========
if mode == "🔁 Auto Mode":
    if refresh_count > 0:
        spo2, hr, ecg = simulate_vitals()
        score, level = calculate_risk(spo2, hr, ecg, bmi)
        timestamp = datetime.now().isoformat()
        save_reading(timestamp, spo2, hr, ecg, bmi, score, level)
        st.metric(label="Current Risk", value=level, delta=f"Score: {score}")
    else:
        st.info("Waiting for first auto-refresh...")

# ========== Manual Mode ==========
if mode == "✍️ Manual Mode":
    st.markdown("### 🧪 Enter Vital Signs")
    spo2 = st.slider("SpO₂ (%)", 85, 100, 96)
    hr = st.slider("Heart Rate (bpm)", 30, 160, 75)
    ecg = st.selectbox("ECG Classification", [0, 1, 3])
    if st.button("🧠 Assess Risk"):
        score, level = calculate_risk(spo2, hr, ecg, bmi)
        timestamp = datetime.now().isoformat()
        save_reading(timestamp, spo2, hr, ecg, bmi, score, level)
        st.metric(label="Manual Risk Result", value=level, delta=f"Score: {score}")

# ========== History Table ==========
st.markdown("---")
st.markdown("### 📋 History of Readings")
history_df = load_history(limit=10)
if history_df.empty:
    st.info("No readings yet.")
else:
    st.dataframe(history_df, use_container_width=True)
    csv = history_df.to_csv(index=False).encode("utf-8")
    st.download_button("📤 Export Readings as CSV", data=csv, file_name="cardiac_readings.csv", mime="text/csv", use_container_width=True)

    # Optional: Chart
    history_df["timestamp"] = pd.to_datetime(history_df["timestamp"])
    history_df = history_df.sort_values("timestamp")
    st.line_chart(data=history_df.set_index("timestamp")[["total_risk"]])
