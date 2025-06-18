import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# ========== Basic Configuration ==========
st.set_page_config(page_title="Cardiac Risk 2.0", page_icon="🫀", layout="centered")
st.title("💓 Cardiac Risk Monitor 2.0")
st.caption("Optimized for touchscreen and mobile devices")

# ========== Auto-refresh every 3 minutes ==========
st_autorefresh(interval=180000, key="auto_refresh")

DB_NAME = "cardiac_monitor.db"

# ========== Database Setup ==========
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS readings
                 (timestamp TEXT, spo2 REAL, hr INTEGER, ecg INTEGER,
                  bmi REAL, total_risk INTEGER, risk_level TEXT)''')
    conn.commit()
    conn.close()

init_db()

# ========== Risk Calculation Logic ==========
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
        level = "🟢 Normal"
    elif score <= 2:
        level = "🟡 Low Risk"
    elif score <= 5:
        level = "🟠 Medium Risk"
    else:
        level = "🔴 High Risk"

    return score, level

def save_reading(ts, spo2, hr, ecg, bmi, total_risk, level):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('INSERT INTO readings VALUES (?, ?, ?, ?, ?, ?, ?)',
              (ts, spo2, hr, ecg, bmi, total_risk, level))
    conn.commit()
    conn.close()

def load_history(limit=10):
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql(f'SELECT * FROM readings ORDER BY timestamp DESC LIMIT {limit}', conn)
    conn.close()
    return df

# ========== Input Section ==========
st.markdown("### 🩺 Enter Vital Readings")

spo2 = st.slider("SpO₂ (%)", 85, 100, 96, help="Oxygen saturation level")
hr = st.slider("Heart Rate (bpm)", 30, 160, 75, help="Beats per minute")
ecg = st.selectbox("ECG Classification", [0, 1, 3], help="0=Normal, 3=Abnormal")
bmi = st.slider("Body Mass Index (BMI)", 15.0, 50.0, 25.0)

st.markdown("—")

# ========== Risk Calculation Trigger ==========
if st.button("🧠 Assess Risk Now", use_container_width=True):
    timestamp = datetime.now().isoformat()
    total_risk, risk_level = calculate_risk(spo2, hr, ecg, bmi)
    save_reading(timestamp, spo2, hr, ecg, bmi, total_risk, risk_level)

    st.success(f"🕒 {datetime.now().strftime('%H:%M:%S')} — Risk: {risk_level} • Score: {total_risk}")
    st.markdown("---")

# ========== History Table ==========
st.markdown("### 📋 Recent Readings")
history = load_history()

if history.empty:
    st.info("No readings saved yet.")
else:
    st.dataframe(history, use_container_width=True)
    csv = history.to_csv(index=False).encode("utf-8")
    st.download_button("📤 Download CSV", data=csv, file_name="cardiac_history.csv", mime="text/csv", use_container_width=True)
