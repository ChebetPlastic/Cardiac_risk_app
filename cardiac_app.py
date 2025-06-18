import streamlit as st
import pandas as pd
import sqlite3
import random
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
import altair as alt
import os

# ========== Page Setup ==========
st.set_page_config(page_title="Cardiac Risk Monitor", page_icon="🫀", layout="centered")

# Safely show logo only if available
if os.path.exists("logo.png"):
    st.sidebar.image("logo.png", width=120)

st.title("💓 Cardiac Risk Monitor 2.0")
st.caption("Switch between Auto and Manual input • Visualize trends • Refreshes every 3 minutes")

refresh_count = st_autorefresh(interval=180000, key="auto-refresh")

# ========== Database ==========
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

# ========== UI ==========
mode = st.radio("Select Input Mode", ["🔁 Auto Mode", "✍️ Manual Mode"], horizontal=True)
bmi = st.number_input("Patient BMI (used in risk scoring)", 15.0, 50.0, step=0.1, value=25.0)

# ========== Auto Mode ==========
if mode == "🔁 Auto Mode":
    if refresh_count > 0:
        spo2, hr, ecg = simulate_vitals()
        score, level = calculate_risk(spo2, hr, ecg, bmi)
        timestamp = datetime.now().isoformat()
        save_reading(timestamp, spo2, hr, ecg, bmi, score, level)
        st.metric(label="Current Auto Risk", value=level, delta=f"Score: {score}")
    else:
        st.info("Waiting for auto-refresh...")

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

# ========== History + Chart ==========
st.markdown("---")
st.markdown("### 📋 Reading History")
df = load_history()

if df.empty:
    st.info("No readings yet.")
else:
    st.dataframe(df, use_container_width=True)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📤 Export CSV", data=csv, file_name="cardiac_readings.csv", mime="text/csv")

    st.markdown("### 📈 Risk Score Trend (Color Coded)")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    def risk_color(score):
        if score == 0: return "#21BF73"  # green
        elif score <= 2: return "#FFC300"
        elif score <= 5: return "#FF6F00"
        else: return "#C70039"

    df["color"] = df["total_risk"].apply(risk_color)

    chart = alt.Chart(df).mark_line(point=alt.OverlayMarkDef(filled=True, size=80)).encode(
        x=alt.X("timestamp:T", title="Time"),
        y=alt.Y("total_risk:Q", title="Risk Score", scale=alt.Scale(domain=[0, 9])),
        color=alt.Color("color:N", scale=None, legend=None)
    ).properties(height=300, width=700)

    st.altair_chart(chart, use_container_width=True)
