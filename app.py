"""Streamlit dashboard: `streamlit run app.py`"""
from datetime import datetime, timedelta, timezone

import pandas as pd
import pydeck as pdk
import streamlit as st
from sklearn.ensemble import HistGradientBoostingRegressor

from src.congestion import run as run_live
from src.data import download, hourly_table, load_incidents

st.set_page_config(page_title="Metro Manila Traffic", page_icon="🚦", layout="wide")
PHT = timezone(timedelta(hours=8))
FEATS = ["hour", "dow", "month", "is_holiday", "is_payday", "quarantine", "rain_mm", "rain_3h"]


@st.cache_resource
def incidents():
    download()
    inc = load_incidents()
    df = hourly_table(inc)
    ok = df[df.gap_day == 0]
    model = HistGradientBoostingRegressor(loss="poisson", max_iter=300, learning_rate=0.05,
                                          max_leaf_nodes=15, random_state=42)
    model.fit(ok[FEATS], ok.incidents)
    return inc, ok, model


@st.cache_data(ttl=1800)
def live():
    return run_live()


now = datetime.now(PHT)
st.title("🚦 Metro Manila Traffic Forecaster")
st.caption(f"{now:%a %d %b %Y, %I:%M %p} PHT · Live data: TomTom Traffic Index, Open-Meteo · "
           "History: 17k MMDA incident alerts")

tab1, tab2, tab3, tab4 = st.tabs(["📅 This week (live)", "🔮 Next 48 h", "🗺️ Incident hotspots", "📊 2025 patterns"])

try:
    data, te, fut, m, live_all, prof = live()
    ok_live = True
except Exception as e:
    ok_live = False
    st.error(f"Live feed unavailable right now: {e}")

with tab1:
    if ok_live:
        last = data.iloc[-1]
        mm = live_all[live_all.city == "Metro Manila"]
        lj = mm.iloc[-1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"Congestion at {last.timestamp:%I %p, %b %d}", f"{last.congestion:.0f}%",
                  f"{last.congestion - last.usual:+.0f} pts vs usual", delta_color="inverse")
        c2.metric("Traffic jams", f"{lj.jams_count:.0f}", f"{lj.jams_length_km:.0f} km total")
        c3.metric("Week peak", f"{data.congestion.max():.0f}%",
                  f"{data.loc[data.congestion.idxmax(), 'timestamp']:%a %I %p}")
        c4.metric("This week vs usual", f"{(data.congestion.mean() / data.usual.mean() - 1):+.0%}")
        st.subheader(f"Hourly congestion, {data.timestamp.min():%b %d} to {data.timestamp.max():%b %d}")
        st.line_chart(data.set_index("timestamp")[["congestion", "usual"]]
                      .rename(columns={"congestion": "Live", "usual": "Usual for that hour"}), height=320)
        st.subheader("Metro Manila vs Davao City")
        st.line_chart(live_all.pivot_table("traffic_index_live", "timestamp", "city"), height=260)
        st.subheader("Traffic-jam length (km)")
        st.area_chart(mm.set_index("timestamp")[["jams_length_km"]], height=200)
        st.caption("Congestion % = extra travel time vs free-flow. Source: TomTom Traffic Index live feed.")

with tab2:
    if ok_live:
        st.subheader("Congestion forecast for the next 48 hours")
        hist = data[data.timestamp >= data.timestamp.max() - pd.Timedelta(hours=48)]
        chart = pd.concat([hist.set_index("timestamp")[["congestion"]].rename(columns={"congestion": "Actual"}),
                           fut.set_index("timestamp")[["forecast", "lo", "hi"]]
                           .rename(columns={"forecast": "Forecast", "lo": "Low (10%)", "hi": "High (90%)"})])
        st.line_chart(chart, height=360)
        a, b = st.columns(2)
        a.metric("Model error (last 48 h)", f"{m['MAE_model']:.1f} pts")
        b.metric("TomTom 'usual' error", f"{m['MAE_tomtom_usual']:.1f} pts")
        st.write("**Worst hours ahead**")
        st.dataframe(fut.nlargest(5, "forecast")[["timestamp", "forecast", "rain_mm", "is_holiday"]].round(1),
                     hide_index=True, width="stretch")
        inc, ok, imodel = incidents()
        f2 = fut.assign(month=fut.timestamp.dt.month,
                        is_payday=((fut.timestamp.dt.day == 15) | fut.timestamp.dt.is_month_end).astype(int),
                        quarantine=0)
        st.metric("Expected MMDA-reported incidents, next 48 h", f"{imodel.predict(f2[FEATS]).sum():.0f}")
        st.caption("Features: hour, weekday, TomTom 2025 typical congestion, live and forecast rainfall "
                   "(Open-Meteo), PH holidays. Retrained on every refresh.")

with tab3:
    inc, ok, _ = incidents()
    years = sorted(inc.timestamp.dt.year.unique())
    yr = st.multiselect("Year", years, default=years)
    cats = st.multiselect("Type", sorted(inc.category.unique()), default=sorted(inc.category.unique()))
    hr = st.slider("Hour of day", 0, 23, (0, 23))
    sel = inc[inc.timestamp.dt.year.isin(yr) & inc.category.isin(cats)
              & inc.timestamp.dt.hour.between(*hr)].dropna(subset=["Latitude", "Longitude"])
    st.pydeck_chart(pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(latitude=14.6, longitude=121.03, zoom=11, pitch=45),
        layers=[pdk.Layer("HexagonLayer", sel[["Longitude", "Latitude"]], get_position=["Longitude", "Latitude"],
                          radius=200, elevation_scale=8, extruded=True, pickable=True)]))
    st.caption(f"{len(sel):,} MMDA incident alerts (2018-2020)")
    st.dataframe(sel.Location.str.upper().str.strip().value_counts().head(15), width="stretch")

with tab4:
    if ok_live:
        st.subheader("Average speed by hour and weekday, 2025 (km/h, TomTom)")
        piv = prof.pivot_table("typical_speed_2025", "hour", "dow").rename(
            columns=dict(enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])))
        st.line_chart(piv, height=320)
        st.subheader("Congestion by hour and weekday, 2025 (%)")
        st.line_chart(prof.pivot_table("typical_2025", "hour", "dow").rename(
            columns=dict(enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]))), height=320)
