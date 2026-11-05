"""Streamlit dashboard (2026 data): `streamlit run app.py`"""
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import streamlit as st
from src.congestion import TARGET_CITY, run

st.set_page_config(page_title="Metro Manila Traffic", page_icon="🚦", layout="wide")
PHT = timezone(timedelta(hours=8))


@st.cache_data(ttl=1800)
def load(refresh: bool, start, days: int, city: str):
    data, te, fut, m, _ = run(refresh=refresh, start=start, days=days, city=city)
    return data, te, fut, m


now = datetime.now(PHT)
st.sidebar.header("🔮 Forecast")
f_start = st.sidebar.date_input("Start date", now.date(), min_value=now.date() - timedelta(days=7))
f_days = st.sidebar.slider("Days ahead", 1, 30, 7)
f_city = st.sidebar.selectbox("City", ["Metro Manila", "Davao City"])
refresh = st.sidebar.button("🔄 Refresh data",
                            help="Optional: pull the newest hours from the public TomTom Traffic Index (no key)")

st.title("🚦 Metro Manila Traffic Forecaster")
st.caption(f"{now:%a %d %b %Y, %I:%M %p} PHT · Real 2026 data: TomTom Traffic Index, "
           "Open-Meteo rainfall, PH holidays")

try:
    data, te, fut, m = load(refresh, str(f_start), f_days, f_city)
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()

mm = data[data.city == TARGET_CITY]
last = mm.iloc[-1]
c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Congestion at {last.timestamp:%I %p, %b %d}", f"{last.congestion:.0f}%",
          f"{last.congestion - last.usual:+.0f} pts vs usual", delta_color="inverse")
c2.metric("Traffic jams now", f"{last.jams_count:.0f}", f"{last.jams_length_km:.0f} km total", delta_color="off")
c3.metric("Peak in period", f"{mm.congestion.max():.0f}%", f"{mm.loc[mm.congestion.idxmax(), 'timestamp']:%a %b %d %I %p}",
          delta_color="off")
c4.metric("Period vs usual", f"{mm.congestion.mean() / mm.usual.mean() - 1:+.0%}", delta_color="off")

tab2, tab1, tab3 = st.tabs(["🔮 Forecast", "📅 Collected data", "📊 Patterns"])


tab1, tab2, tab3 = st.tabs(["📅 This month", "🔮 Next 48 h", "📊 Patterns"])

with tab1:
    st.subheader(f"Hourly congestion, {mm.timestamp.min():%b %d} to {mm.timestamp.max():%b %d, %Y}")
    st.line_chart(mm.set_index("timestamp")[["congestion", "usual"]]
                  .rename(columns={"congestion": "Live", "usual": "Usual for that hour"}), height=320)
    st.subheader("Traffic-jam length (km)")
    st.area_chart(mm.set_index("timestamp")[["jams_length_km"]], height=200)
    st.subheader("Metro Manila vs Davao City")
    st.line_chart(data.pivot_table("congestion", "timestamp", "city"), height=260)
    st.caption(f"{len(mm):,} hours collected so far. Click 'Refresh data' in the sidebar to pull the newest hours.")

with tab2:
    st.subheader(f"{f_city}: {fut.timestamp.min():%a %b %d} to {fut.timestamp.max():%a %b %d, %Y}")
    st.line_chart(fut.set_index("timestamp")[["forecast", "lo", "hi"]]
                  .rename(columns={"forecast": "Forecast", "lo": "Low (10%)", "hi": "High (90%)"}), height=360)
    daily = fut.groupby(fut.timestamp.dt.date).agg(
        avg_congestion=("forecast", "mean"), peak=("forecast", "max"),
        holiday=("is_holiday", "max"), rain_mm=("rain_mm", "sum"))
    daily["peak_hour"] = fut.loc[fut.groupby(fut.timestamp.dt.date).forecast.idxmax()].timestamp.dt.strftime("%I %p").values
    daily["day"] = pd.to_datetime(daily.index).strftime("%a")
    daily["holiday"] = daily.holiday.map({1: "🎉 yes", 0: ""})
    st.write("**Day by day**")
    st.dataframe(daily[["day", "avg_congestion", "peak", "peak_hour", "holiday", "rain_mm"]].round(0),
                 width="stretch")
    st.write("**Worst hours in this range**")
    st.dataframe(fut.nlargest(8, "forecast")[["timestamp", "forecast", "rain_mm", "is_holiday"]].round(0),
                 hide_index=True, width="stretch")
    a, b = st.columns(2)
    a.metric("Model error on held-out 48 h", f"{m['MAE_model']:.1f} pts")
    b.metric("TomTom 'usual' error", f"{m['MAE_tomtom_usual']:.1f} pts")
    st.caption(f"Rain input: {fut.rain_source.iloc[0]}. The forecast uses the weekday/hour pattern, official "
               "PH holidays (any year) and rain. Further than ~16 days ahead, rain is assumed dry.")

with tab3:
    k = mm.assign(kind=np.select([mm.is_holiday == 1, mm.dow >= 5], ["Holiday", "Weekend"], "Weekday"))
    st.subheader("Average congestion by hour")
    st.line_chart(k.pivot_table("congestion", "hour", "kind"), height=320)
    st.subheader("Congestion by day")
    st.bar_chart(mm.groupby(mm.timestamp.dt.date).congestion.mean(), height=260)
    st.subheader("Rain vs extra congestion")
    st.scatter_chart(mm.assign(extra=mm.congestion - mm.usual), x="rain_mm", y="extra", height=260)
