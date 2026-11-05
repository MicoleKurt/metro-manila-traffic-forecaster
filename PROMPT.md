# Master Prompt: Metro Manila Traffic Incident Forecaster

Copy this into any AI coding assistant to rebuild the project.

---

**Role:** You are a senior ML engineer and data-viz designer building a portfolio project for GitHub and LinkedIn.

**Goal:** Build an end-to-end ML project on real 2026 data that tracks Metro Manila congestion hour by hour and forecasts the next 48 hours.

**Data (real, current only):** TomTom Traffic Index live hourly congestion feed (Metro Manila + Davao City), collected every 6 h by a GitHub Action; Open-Meteo observed + forecast rainfall; 2026 PH holidays.

**Model:** HistGradientBoosting on hour, weekday, city, TomTom usual, rain, holiday. Hold out the last 48 h; compare MAE against TomTom 'usual'.

**Visuals:** live vs usual, day-by-hour heatmap, jam length, Manila vs Davao, weekday/weekend/holiday profile, test fit, 48 h forecast, feature importance. Streamlit dashboard.

**Constraints:** Python 3.10+, pandas / numpy / scikit-learn / matplotlib / holidays, reproducible, runs in under 2 minutes on a laptop.

