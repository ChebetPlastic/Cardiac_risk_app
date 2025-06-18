# 💓 Cardiac Risk Monitor
...
Made with ❤️ by Maren

A mobile-friendly Streamlit app that simulates or manually logs vital sign readings to assess cardiac risk in real time.

## 🔁 Features
- **Auto Mode**: Automatically generates vitals every 3 minutes
- **Manual Mode**: Enter SpO₂, Heart Rate, ECG, and BMI manually
- **Risk scoring** based on input
- **Downloadable reading history**
- **Mobile-first design**, suitable for tablets and wearables

## ⚙️ Tech Stack
- Streamlit
- Python
- SQLite (local lightweight storage)
- Auto-refresh via `streamlit-autorefresh`

## 🚀 Launch the App

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://<your-app-name>.streamlit.app)

> Replace `<your-app-name>` with your deployed app's subdomain.

## 📦 Installation (Optional: run locally)

```bash
pip install streamlit streamlit-autorefresh pandas
streamlit run cardiac_risk_2.py
