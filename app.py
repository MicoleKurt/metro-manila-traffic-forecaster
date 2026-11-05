"""Streamlit dashboard (2026 data): `streamlit run app.py`"""
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import streamlit as st
from src.congestion import TARGET_CITY, run

st.set_page_config(page_title="Metro Manila Traffic", page_icon="🚦", layout="wide")
PHT = timezone(timedelta(hours=8))


@st.cache_data(ttl=1800)
def load(refresh: bool = False):
    data, te, fut, m, _ = run(refresh=refresh)
    return data, te, fut, m


refresh = st.sidebar.button("🔄 Refresh data",
                            help="Pull the newest hours from the public TomTom Traffic Index (no key needed)")


now = datetime.now(PHT)
st.title("🚦 Metro Manila Traffic Forecaster")
st.caption(f"{now:%a %d %b %Y, %I:%M %p} PHT · Real 2026 data: TomTom Traffic Index, "
           "Open-Meteo rainfall, PH holidays")

try:
    data, te, fut, m = load(refresh)
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
    hist = mm[mm.timestamp >= mm.timestamp.max() - pd.Timedelta(hours=48)]
    chart = pd.concat([hist.set_index("timestamp")[["congestion"]].rename(columns={"congestion": "Actual"}),
                       fut.set_index("timestamp")[["forecast", "lo", "hi"]]
                       .rename(columns={"forecast": "Forecast", "lo": "Low (10%)", "hi": "High (90%)"})])
    st.line_chart(chart, height=360)
    a, b = st.columns(2)
    a.metric("Model error, last 48 h", f"{m['MAE_model']:.1f} pts")
    b.metric("TomTom 'usual' error", f"{m['MAE_tomtom_usual']:.1f} pts")
    st.write("**Worst hours ahead**")
    st.dataframe(fut.nlargest(5, "forecast")[["timestamp", "forecast", "rain_mm", "is_holiday"]].round(1),
                 hide_index=True, width="stretch")

with tab3:
    k = mm.assign(kind=np.select([mm.is_holiday == 1, mm.dow >= 5], ["Holiday", "Weekend"], "Weekday"))
    st.subheader("Average congestion by hour")
    st.line_chart(k.pivot_table("congestion", "hour", "kind"), height=320)
    st.subheader("Congestion by day")
    st.bar_chart(mm.groupby(mm.timestamp.dt.date).congestion.mean(), height=260)
    st.subheader("Rain vs extra congestion")
    st.scatter_chart(mm.assign(extra=mm.congestion - mm.usual), x="rain_mm", y="extra", height=260)
