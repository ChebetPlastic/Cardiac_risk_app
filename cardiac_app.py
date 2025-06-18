import streamlit as st
import pandas as pd
import sqlite3
import random
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# ========== Config ==========
st.set_page_config(page_title="Cardiac Risk Monitor", page_icon="🫀", layout="centered")
st.title("💓 Cardiac Risk Monitor 2.0")
st.caption("Simulated auto-readings every 3 minutes — set BMI once")

# ========== Auto-refresh every 3 minutes ==========
count = st_autorefresh(interval=180000, key="refresh_key")

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

    if score == 0: level = "🟢 Normal"
    elif score <= 2: level = "🟡 Low Risk"
    elif score <= 5: level = "🟠 Medium Risk"
    else: level = "🔴 High Risk"

    return score, level

def save_reading(ts, spo2, hr, ecg, bmi, risk_score, risk_level):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('INSERT INTO readings VALUES (?, ?, ?, ?, ?, ?, ?)',
              (ts, spo2, hr, ecg, bmi, risk_score, risk_level))
    conn.commit()
    conn.close()

def load_history(limit=10):
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql(f'SELECT * FROM readings ORDER BY timestamp DESC LIMIT {limit}', conn)
    conn.close()
    return df

# ========== Persistent BMI ==========
bmi = st.number_input("Enter Patient's BMI (used for all future simulations)", min_value=15.0, max_value=50.0, value=25.0, key="bmi_input")

# ========== Simulate Readings Automatically ==========
def simulate_and_save():
    spo2 = round(random.uniform(89, 99), 1)
    hr = random.randint(45, 140)
    ecg = random.choices([0, 1, 3], weights=[0.85, 0.1, 0.05])[0]
    risk_score, risk_level = calculate_risk(spo2, hr, ecg, bmi)
    ts = datetime.now().isoformat()
    save_reading(ts, spo2, hr, ecg, bmi, risk_score, risk_level)
    return ts, spo2, hr, ecg, risk_score, risk_level

# ========== Trigger Simulation ==========
ts, spo2, hr, ecg, score, risk_level = simulate_and_save()

st.success(f"🕒 {datetime.now().strftime('%H:%M:%S')} • HR: {hr} • SpO₂: {spo2}% • ECG: {ecg} → Risk: {risk_level}")

# ========== Display History ==========
st.markdown("### 📋 Recent Readings")
history = load_history()
if history.empty:
    st.info("No data yet")
else:
    st.dataframe(history, use_container_width=True)
    csv = history.to_csv(index=False).encode("utf-8")
    st.download_button("📤 Export CSV", data=csv, file_name="cardiac_auto_log.csv", mime="text/csv", use_container_width=True)
