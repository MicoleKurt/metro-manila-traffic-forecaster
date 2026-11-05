# Master Prompt — Philippine Traffic Forecaster

Copy this into any AI coding assistant to (re)build the project.

---

**Role:** You are a senior ML engineer and data-viz designer building a portfolio project for GitHub and LinkedIn.

**Goal:** Build *Metro Manila Traffic Forecaster* — an end-to-end ML project that predicts hourly average travel speed (km/h) on major Metro Manila corridors (EDSA, C5, Commonwealth Ave, España Blvd, Roxas Blvd, SLEX, NLEX) and explains *why* traffic gets bad.

**Data (be honest about sources):**
1. Use real public sources where available and cite them in the README: MMDA traffic volume / AADT reports, TomTom Traffic Index (Manila), PAGASA rainfall & typhoon records, Official Gazette public-holiday proclamations, MMDA UVVRP (number coding) rules.
2. Where no free hourly dataset exists, generate a **synthetic dataset calibrated to those public patterns** and label it clearly as synthetic. Provide a loader so real data (e.g., TomTom/HERE/Google API exports, MMDA CSVs) can be dropped in with the same schema.
3. Encode PH-specific drivers: weekday AM/PM rush, Friday-evening peak, payday (15th & 30th), Christmas rush ("ber" months, December peak), Holy Week & long-weekend exodus (lighter city traffic), rainy season (Jun–Nov) and typhoon days, number coding hours, mall-sale days.

**Model:**
- Time-aware train/test split (train on earlier dates, test on the latest months — no leakage).
- Baseline (historical hour×weekday mean) vs. gradient boosting (scikit-learn HistGradientBoosting). Report MAE, RMSE, R² and % improvement over baseline.
- Permutation feature importance for explainability.
- 7-day forward forecast for EDSA.

**Visuals (dark theme, LinkedIn-ready 1600×900 PNGs in `assets/`):**
1. Hour × weekday congestion heatmap
2. Actual vs predicted (one test week, EDSA)
3. Feature importance bar chart
4. Rain & typhoon impact by hour
5. Monthly seasonality ("ber" months effect)
6. Corridor comparison at rush hour
7. 7-day EDSA forecast with uncertainty band

**Deliverables:** `src/` (data, train, visualize), `requirements.txt`, one-command run (`python run.py`), README with results table, charts, data-source citations, limitations and next steps, plus a ready-to-post LinkedIn caption.

**Constraints:** Python 3.10+, pandas/numpy/scikit-learn/matplotlib only, reproducible seed, runs on a laptop in under 2 minutes.
