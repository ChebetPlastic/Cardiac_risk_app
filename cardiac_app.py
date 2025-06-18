import streamlit as st
import pandas as pd
import sqlite3
import random
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# ========== Page Config ==========
st.set_page_config(page_title="Cardiac Risk Monitor", page_icon="🫀", layout="centered")
st.title("💓 Cardiac Risk Monitor 2.0")
st.caption("Auto-refreshes every 3 minutes with simulated readings")

# ========== Refresh Trigger ==========
refresh_counter = st_autorefresh(interval=180000, key="auto-refresh")  # every 3 min

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

# ========== Risk Logic ==========
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
    ecg = random.choices([0, 1, 3], weights=[0.85, 0.1, 0.05])[0]
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
    df = pd.read_sql(f"SELECT * FROM readings ORDER BY timestamp DESC LIMIT {limit}", conn)
    conn.close()
    return df

# ========== BMI Input ==========
bmi = st.number_input("Set Patient's BMI (used for all automatic readings)", min_value=15.0, max_value=50.0, value=25.0, step=0.1)

# ========== Auto Simulate on Every Refresh ==========
if refresh_counter > 0:
    spo2, hr, ecg = simulate_vitals()
    score, level = calculate_risk(spo2, hr, ecg, bmi)
    timestamp = datetime.now().isoformat()
    save_reading(timestamp, spo2, hr, ecg, bmi, score, level)

    st.success(f"🕒 {datetime.now().strftime('%H:%M:%S')} | SpO₂: {spo2}% | HR: {hr} bpm | ECG: {ecg} → Risk: {level}")

# ========== Show Table ==========
st.markdown("### 📋 Reading History")
history = load_history()

if history.empty:
    st.info("No readings yet.")
else:
    st.dataframe(history, use_container_width=True)
    csv = history.to_csv(index=False).encode("utf-8")
    st.download_button("📤 Export CSV", data=csv, file_name="cardiac_readings.csv", mime="text/csv", use_container_width=True)
