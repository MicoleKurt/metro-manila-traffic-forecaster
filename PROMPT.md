# Master Prompt: Metro Manila Traffic Incident Forecaster

Copy this into any AI coding assistant to rebuild the project.

---

**Role:** You are a senior ML engineer and data-viz designer building a portfolio project for GitHub and LinkedIn.

**Goal:** Build an end-to-end ML project that forecasts hourly MMDA-reported road incidents in Metro Manila and explains where, when and why they happen.

**Data (real only, no synthetic):**
- MMDA incident alerts, 2018–2020, from the Tweet2Map archive (time, location, lat/long, type, lanes blocked)
- Open-Meteo ERA5 hourly rainfall for Manila
- Official PH holidays from the `holidays` package
- NCR 2020 quarantine phases (ECQ / MECQ / GCQ)

Download the raw data at runtime and don't commit it. Flag scraping-gap days (zero alerts outside the ECQ) and exclude them. Cite the TomTom Traffic Index 2025 and the MMDA AADT reports as context.

**Model:**
- Time-based split: train on Aug 2018 to Aug 2020, test on Sep–Dec 2020. No leakage.
- Compare an hour × weekday average baseline against HistGradientBoosting with a Poisson loss.
- Report hourly and daily MAE.
- Use permutation feature importance for explainability.

**Visuals (dark theme, LinkedIn-ready PNGs in `assets/`):**
1. Hour × weekday incident heatmap
2. Daily incidents with lockdown phases shaded
3. Hotspot hexbin map
4. Top 15 locations
5. Rain effect
6. Incident types
7. Actual vs predicted on the test period
8. Feature importance

**Deliverables:**
- `src/data.py` and `run.py`
- `requirements.txt`
- `python run.py` runs everything in one command
- A README with findings, a results table, charts, sources and limitations
- A LinkedIn post

**Constraints:** Python 3.10+, pandas / numpy / scikit-learn / matplotlib / holidays, reproducible, runs in under 2 minutes on a laptop.
