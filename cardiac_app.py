import streamlit as st
import pandas as pd
import sqlite3
import random
from datetime import datetime, date
from streamlit_autorefresh import st_autorefresh
import altair as alt
import os

# ========== Page Setup ==========
st.set_page_config(page_title="Cardiac Risk Monitor", page_icon="🫀", layout="centered")
if os.path.exists("logo.png"):
    st.sidebar.image("logo.png", width=120)

st.title("💓 Cardiac Risk Monitor 2.0")
st.caption("Full patient profiles • Auto/manual input • Risk trends • 3-minute refresh")

refresh_count = st_autorefresh(interval=180000, key="auto-refresh")

# ========== Database Setup ==========
DB_NAME = "cardiac_monitor.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS readings (
        timestamp TEXT, patient_id TEXT, first_name TEXT, last_name TEXT,
        dob TEXT, age INTEGER, consultant TEXT, location TEXT,
        weight REAL, height REAL, bmi REAL,
        spo2 REAL, hr INTEGER, ecg INTEGER,
        total_risk INTEGER, risk_level TEXT
    )
    ''')
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
    ecg = random.choices([0, 1, 3], weights=[0.8, 0.15, 0.05])[0]
    return spo2, hr, ecg

def save_reading(ts, pid, fname, lname, dob, age, consultant, loc, wt, ht, bmi, spo2, hr, ecg, score, level):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
    INSERT INTO readings VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (ts, pid, fname, lname, dob, age, consultant, loc, wt, ht, bmi, spo2, hr, ecg, score, level))
    conn.commit()
    conn.close()

def load_history(pid, limit=20):
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql('SELECT * FROM readings WHERE patient_id = ? ORDER BY timestamp DESC LIMIT ?', conn, params=(pid, limit))
    conn.close()
    return df

# ========== Patient Profile ==========
st.markdown("## 🧑‍⚕️ Patient Profile")

pid = st.text_input("Patient ID", value="P001")
fname = st.text_input("First Name")
lname = st.text_input("Surname")
dob = st.date_input("Date of Birth")
today = date.today()
age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
st.write(f"🧮 Age: **{age}** years")

consultant = st.text_input("Consultant Name")
location = st.text_input("Ward / Location")
weight = st.number_input("Weight (kg)", 2.0, 250.0, value=70.0)
height = st.number_input("Height (cm)", 30.0, 250.0, value=170.0)

bmi = round(weight / ((height / 100) ** 2), 1)
st.write(f"📏 Calculated BMI: **{bmi}**")

# ========== Input Mode ==========
mode = st.radio("Select Input Mode", ["🔁 Auto Mode", "✍️ Manual Mode"], horizontal=True)

# ========== Auto Mode ==========
if mode == "🔁 Auto Mode":
    if refresh_count > 0:
        spo2, hr, ecg = simulate_vitals()
        score, level = calculate_risk(spo2, hr, ecg, bmi)
        ts = datetime.now().isoformat()
        save_reading(ts, pid, fname, lname, dob.isoformat(), age, consultant, location, weight, height, bmi, spo2, hr, ecg, score, level)
        st.metric("Auto Risk", value=level, delta=f"Score: {score}")
    else:
        st.info("⏳ Waiting for auto-refresh...")

# ========== Manual Mode ==========
if mode == "✍️ Manual Mode":
    st.markdown("### 🧪 Manual Entry")
    spo2 = st.slider("SpO₂ (%)", 85, 100, 96)
    hr = st.slider("Heart Rate (bpm)", 30, 160, 75)
    ecg = st.selectbox("ECG", [0, 1, 3])
    if st.button("🧠 Assess Risk"):
        score, level = calculate_risk(spo2, hr, ecg, bmi)
        ts = datetime.now().isoformat()
        save_reading(ts, pid, fname, lname, dob.isoformat(), age, consultant, location, weight, height, bmi, spo2, hr, ecg, score, level)
        st.metric("Manual Risk", value=level, delta=f"Score: {score}")

# ========== History & Chart ==========
st.markdown("---")
st.markdown(f"### 📋 History for Patient: `{pid}`")

df = load_history(pid)
if df.empty:
    st.info("No readings yet for this patient.")
else:
    st.dataframe(df, use_container_width=True)
    st.download_button("📤 Export CSV", data=df.to_csv(index=False).encode("utf-8"), file_name=f"{pid}_readings.csv", mime="text/csv")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    df["color"] = df["total_risk"].apply(lambda s: "#21BF73" if s == 0 else "#FFC300" if s <= 2 else "#FF6F00" if s <= 5 else "#C70039")

    st.markdown("### 📈 Risk Trend")
    chart = alt.Chart(df).mark_line(point=alt.OverlayMarkDef(filled=True, size=80)).encode(
        x=alt.X("timestamp:T", title="Time"),
        y=alt.Y("total_risk:Q", title="Risk Score", scale=alt.Scale(domain=[0, 9])),
        color=alt.Color("color:N", scale=None, legend=None)
    ).properties(height=300, width=700)

    st.altair_chart(chart, use_container_width=True)
