"""Streamlit dashboard: `streamlit run app.py`"""
import json
import os
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import holidays
import pandas as pd
import pydeck as pdk
import streamlit as st
from sklearn.ensemble import HistGradientBoostingRegressor

from src.data import download, hourly_table, load_incidents

st.set_page_config(page_title="Metro Manila Traffic", page_icon="🚦", layout="wide")
PHT = timezone(timedelta(hours=8))
FEATS = ["hour", "dow", "month", "is_holiday", "is_payday", "quarantine", "rain_mm", "rain_3h"]
LIVE_WX = ("https://api.open-meteo.com/v1/forecast?latitude=14.5995&longitude=120.9842"
           "&hourly=precipitation&past_days=7&forecast_days=7&timezone=Asia%2FManila")


@st.cache_resource
def load():
    download()
    inc = load_incidents()
    df = hourly_table(inc)
    ok = df[df.gap_day == 0]
    model = HistGradientBoostingRegressor(loss="poisson", max_iter=300, learning_rate=0.05,
                                          max_leaf_nodes=15, random_state=42)
    model.fit(ok[FEATS], ok.incidents)
    return inc, ok, model


@st.cache_data(ttl=1800)
def live_weather():
    with urllib.request.urlopen(LIVE_WX, timeout=20) as r:
        h = json.load(r)["hourly"]
    return pd.DataFrame({"timestamp": pd.to_datetime(h["time"]), "rain_mm": h["precipitation"]})


def week_features(wx: pd.DataFrame) -> pd.DataFrame:
    f = wx.copy()
    ts = f.timestamp
    ph = holidays.PH(years=sorted(ts.dt.year.unique()))
    f["rain_mm"] = f.rain_mm.fillna(0)
    f["rain_3h"] = f.rain_mm.rolling(3, min_periods=1).sum()
    f["hour"], f["dow"], f["month"] = ts.dt.hour, ts.dt.dayofweek, ts.dt.month
    f["is_holiday"] = ts.dt.date.map(lambda d: d in ph).astype(int)
    f["is_payday"] = ((ts.dt.day == 15) | ts.dt.is_month_end | (ts.dt.day == 30)).astype(int)
    f["quarantine"] = 0
    return f


inc, ok, model = load()
now = datetime.now(PHT)
st.title("🚦 Metro Manila Traffic Incident Forecaster")
st.caption(f"Updated {now:%a %d %b %Y, %I:%M %p} PHT · Model trained on {len(inc):,} real MMDA incident alerts")

tab1, tab2, tab3, tab4 = st.tabs(["📅 This week", "🛰️ Live speeds", "🗺️ Hotspot map", "📊 Patterns"])

with tab1:
    try:
        f = week_features(live_weather())
        f["expected_incidents"] = model.predict(f[FEATS])
        start = pd.Timestamp(now.date()) - pd.Timedelta(days=now.weekday())
        wk = f[(f.timestamp >= start) & (f.timestamp < start + pd.Timedelta(days=7))]
        cur = f[f.timestamp.dt.floor("h") == pd.Timestamp(now.replace(tzinfo=None)).floor("h")]
        c1, c2, c3 = st.columns(3)
        if len(cur):
            c1.metric("Expected incidents this hour", f"{cur.expected_incidents.iloc[0]:.1f}")
            c2.metric("Rain this hour (live)", f"{cur.rain_mm.iloc[0]:.1f} mm")
        c3.metric("Expected incidents this week", f"{wk.expected_incidents.sum():.0f}")
        st.subheader(f"Week of {start:%b %d}: hourly forecast with live rainfall")
        st.line_chart(wk.set_index("timestamp")[["expected_incidents"]], height=300)
        st.bar_chart(wk.set_index("timestamp")[["rain_mm"]], height=180)
        hol = wk[wk.is_holiday == 1].timestamp.dt.date.unique()
        if len(hol):
            st.info("Holiday this week: " + ", ".join(str(d) for d in hol))
        worst = wk.nlargest(5, "expected_incidents")[["timestamp", "expected_incidents", "rain_mm"]]
        st.write("**Riskiest hours this week**")
        st.dataframe(worst, hide_index=True, width="stretch")
        st.caption("Rainfall: Open-Meteo live observations (past 7 days) + forecast (next 7 days). "
                   "Expected counts are at the pre-pandemic baseline rate of MMDA-reported incidents.")
    except Exception as e:
        st.error(f"Could not fetch live weather: {e}")

with tab2:
    p = Path("live/speeds.csv")
    if p.exists():
        s = pd.read_csv(p, parse_dates=["timestamp"])
        last = s[s.timestamp == s.timestamp.max()].copy()
        last["congestion_%"] = (1 - last.current_speed_kph / last.free_flow_speed_kph).clip(0) * 100
        st.subheader(f"Latest reading: {s.timestamp.max():%a %d %b, %I:%M %p} PHT (TomTom)")
        st.dataframe(last[["road", "segment", "current_speed_kph", "free_flow_speed_kph", "congestion_%"]]
                     .round(0), hide_index=True, width="stretch")
        wk = s[s.timestamp >= s.timestamp.max() - pd.Timedelta(days=7)]
        st.subheader("EDSA speeds, last 7 days")
        st.line_chart(wk[wk.road == "EDSA"].pivot_table("current_speed_kph", "timestamp", "segment"),
                      height=320)
        st.caption(f"{len(s):,} real readings since {s.timestamp.min():%b %d, %Y}")
    else:
        st.info("No live speeds yet. Add a free TomTom key as the `TOMTOM_API_KEY` repo secret; "
                "the hourly GitHub Action then fills `live/speeds.csv` with real speeds.")

with tab3:
    years = sorted(inc.timestamp.dt.year.unique())
    yr = st.multiselect("Year", years, default=years)
    cats = st.multiselect("Type", sorted(inc.category.unique()), default=sorted(inc.category.unique()))
    hr = st.slider("Hour of day", 0, 23, (0, 23))
    m = inc[inc.timestamp.dt.year.isin(yr) & inc.category.isin(cats)
            & inc.timestamp.dt.hour.between(*hr)].dropna(subset=["Latitude", "Longitude"])
    st.pydeck_chart(pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(latitude=14.6, longitude=121.03, zoom=11, pitch=45),
        layers=[pdk.Layer("HexagonLayer", m[["Longitude", "Latitude"]], get_position=["Longitude", "Latitude"],
                          radius=200, elevation_scale=8, extruded=True, pickable=True)]))
    st.caption(f"{len(m):,} incidents shown")
    st.write("**Top locations**")
    st.dataframe(m.Location.str.upper().str.strip().value_counts().head(15), width="stretch")

with tab4:
    pre = ok[ok.quarantine == 0]
    st.subheader("Average incidents per hour by weekday (pre-pandemic)")
    piv = pre.pivot_table("incidents", "hour", "dow").rename(columns=dict(enumerate(
        ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])))
    st.line_chart(piv, height=320)
    st.subheader("Daily incidents")
    d = ok.groupby(ok.timestamp.dt.date).incidents.sum()
    st.line_chart(d.rolling(7).mean(), height=280)
