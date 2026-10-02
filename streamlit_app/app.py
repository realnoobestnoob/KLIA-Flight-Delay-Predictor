"""Thin Streamlit front-end. All intelligence lives in the API; this file only collects inputs and shows the answer.

    streamlit run streamlit_app/app.py
Set KLIA_API_URL (default http://localhost:8000) and, if you enabled it, KLIA_API_KEY.
"""
import datetime as dt
import os

import requests
import streamlit as st

API = os.environ.get("KLIA_API_URL", "http://localhost:8000").rstrip("/")
HEADERS = {"X-API-Key": os.environ["KLIA_API_KEY"]} if os.environ.get("KLIA_API_KEY") else {}

st.set_page_config(page_title="KLIA Flight Delay", page_icon="✈️")
st.title("✈️ KLIA departure delay risk")


@st.cache_data(ttl=600, show_spinner=False)
def options():
    r = requests.get(f"{API}/v1/options", headers=HEADERS, timeout=60)   # first call may wake a sleeping host
    r.raise_for_status()
    return r.json()


try:
    opt = options()
except Exception as e:
    st.error(f"Cannot reach the API at {API}: {e}")
    st.stop()

airline = st.selectbox("Airline", opt["airlines"])
dests = opt["routes"].get(airline) or opt["destinations"]
destination = st.selectbox("Destination", dests)
c1, c2 = st.columns(2)
day = c1.date_input("Date", dt.date.today())
time_ = c2.time_input("Scheduled departure (KLIA local time)", dt.time(9, 0))
aircraft = st.text_input("Aircraft (optional, e.g. A320)")

if st.button("Predict", type="primary"):
    body = {"airline": airline, "destination": destination,
            "scheduled_departure": dt.datetime.combine(day, time_).isoformat(), "aircraft": aircraft or None}
    try:
        r = requests.post(f"{API}/v1/predict", json=body, headers=HEADERS, timeout=60)
        r.raise_for_status()
        j = r.json()
    except Exception as e:
        st.error(f"Prediction failed: {e}")
    else:
        st.metric("Delay probability", f"{j['delay_probability']:.0%}")
        msg = {"low": "Unlikely to be delayed", "elevated": "Possible delay", "high": "Likely to be delayed"}[j["risk"]]
        {"low": st.success, "elevated": st.warning, "high": st.error}[j["risk"]](msg)
        st.caption(f"model v{j['model_version']} · decision threshold {j['threshold']:.0%} · weather: {j['weather_source']}"
                   + ("" if j["known_route"] else " · new route for this model, so the estimate is less certain"))
