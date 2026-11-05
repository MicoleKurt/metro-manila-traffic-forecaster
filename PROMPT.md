# Master Prompt â€” Philippine Traffic Forecaster

Copy this into any AI coding assistant to (re)build the project.

---

**Role:** You are a senior ML engineer and data-viz designer building a portfolio project for GitHub and LinkedIn.

**Goal:** Build *Metro Manila Traffic Forecaster* â€” an end-to-end ML project that predicts hourly average travel speed (km/h) on major Metro Manila corridors (EDSA, C5, Commonwealth Ave, EspaÃ±a Blvd, Roxas Blvd, SLEX, NLEX) and explains *why* traffic gets bad.

**Data (real only, no synthetic):** MMDA incident alerts 2018-2020 (Tweet2Map archive), Open-Meteo ERA5 hourly rainfall for Manila, official PH holidays, NCR 2020 quarantine phases. Flag scraping-gap days and exclude them. Cite TomTom Traffic Index 2025 and MMDA AADT as context.

**Model:**
- Time-aware train/test split (train on earlier dates, test on the latest months â€” no leakage).
- Baseline (historical hourÃ—weekday mean) vs. gradient boosting (scikit-learn HistGradientBoosting). Report MAE, RMSE, RÂ² and % improvement over baseline.
- Permutation feature importance for explainability.
- 7-day forward forecast for EDSA.

**Visuals (dark theme, LinkedIn-ready 1600Ã—900 PNGs in `assets/`):**
1. Hour Ã— weekday congestion heatmap
2. Actual vs predicted (one test week, EDSA)
3. Feature importance bar chart
4. Rain & typhoon impact by hour
5. Monthly seasonality ("ber" months effect)
6. Corridor comparison at rush hour
7. 7-day EDSA forecast with uncertainty band

**Deliverables:** `src/` (data, train, visualize), `requirements.txt`, one-command run (`python run.py`), README with results table, charts, data-source citations, limitations and next steps, plus a ready-to-post LinkedIn caption.

**Constraints:** Python 3.10+, pandas/numpy/scikit-learn/matplotlib only, reproducible seed, runs on a laptop in under 2 minutes.

