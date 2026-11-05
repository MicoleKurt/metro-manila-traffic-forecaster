"""2026 congestion model: real TomTom hourly congestion (Metro Manila + Davao City)
+ live/forecast Open-Meteo rainfall + 2026 PH holidays.
"""
import json
import urllib.request
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src import live_traffic

WX = ("https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
      "&hourly=precipitation&past_days={past}&forecast_days=3&timezone=Asia%2FManila")
COORDS = {"Metro Manila": (14.5995, 120.9842), "Davao City": (7.0731, 125.6128)}
CITY_ID = {"Metro Manila": 0, "Davao City": 1}
FEATS = ["hour", "dow", "city_id", "usual", "rain_mm", "rain_3h", "is_holiday"]
TARGET_CITY = "Metro Manila"


def weather(city: str, past_days: int) -> pd.DataFrame:
    lat, lon = COORDS[city]
    url = WX.format(lat=lat, lon=lon, past=min(max(past_days, 1), 92))
    with urllib.request.urlopen(url, timeout=30) as r:
        h = json.load(r)["hourly"]
    w = pd.DataFrame({"timestamp": pd.to_datetime(h["time"]), "rain_mm": h["precipitation"]})
    w["rain_mm"] = w.rain_mm.fillna(0)
    w["rain_3h"] = w.rain_mm.rolling(3, min_periods=1).sum()
    w["city"] = city
    return w


def add_calendar(f: pd.DataFrame) -> pd.DataFrame:
    f["hour"], f["dow"] = f.timestamp.dt.hour, f.timestamp.dt.dayofweek
    ph = holidays.PH(years=sorted(f.timestamp.dt.year.unique()))
    f["is_holiday"] = f.timestamp.dt.date.map(lambda d: d in ph).astype(int)
    f["city_id"] = f.city.map(CITY_ID)
    return f


WX_FILE = Path("live/weather.csv")


def run(test_hours: int = 48, refresh: bool = False):
    """Self-contained by default: trains on the data snapshot bundled in live/.
    refresh=True pulls the newest public data first (no API key needed)."""
    if refresh or not live_traffic.OUT.exists():
        live = live_traffic.collect()
        days = (pd.Timestamp.now().normalize() - live.timestamp.min().normalize()).days + 2
        pd.concat([weather(c, days) for c in COORDS]).to_csv(WX_FILE, index=False)
    live = live_traffic.load()
    live = live[live.timestamp.dt.year == 2026].copy()
    wx = pd.read_csv(WX_FILE, parse_dates=["timestamp"])

    data = live.rename(columns={"traffic_index_live": "congestion", "traffic_index_usual": "usual"})
    data = add_calendar(data.merge(wx, on=["city", "timestamp"], how="left")
                        .fillna({"rain_mm": 0, "rain_3h": 0}))
    data["excess"] = data.congestion - data.usual

    # hold out the last `test_hours` hours of Metro Manila
    cutoff = data[data.city == TARGET_CITY].timestamp.max() - pd.Timedelta(hours=test_hours - 1)
    tr, te = data[data.timestamp < cutoff], data[(data.timestamp >= cutoff) & (data.city == TARGET_CITY)].copy()
    model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=8,
                                          min_samples_leaf=5, random_state=42)
    model.fit(tr[FEATS], tr.congestion)
    te["pred"] = model.predict(te[FEATS])
    # baselines that need no model
    last_week = data.set_index(["city", "timestamp"]).congestion
    te["same_hour_last_week"] = [last_week.get((TARGET_CITY, t - pd.Timedelta(days=7)), np.nan)
                                 for t in te.timestamp]
    def mae(p):  # None until there is enough history (e.g. a week-ago baseline)
        err = np.abs(te.congestion - p).dropna()
        return round(float(err.mean()), 2) if len(err) else None
    metrics = {"data_range": f"{data.timestamp.min()} -> {data.timestamp.max()}",
               "hours_collected": int(len(data)), "train_rows": int(len(tr)), "test_hours": int(len(te)),
               "test_period": f"{te.timestamp.min()} -> {te.timestamp.max()}",
               "MAE_model": mae(te.pred), "MAE_tomtom_usual": mae(te.usual),
               "MAE_same_hour_last_week": mae(te.same_hour_last_week)}

    # refit on everything; forecast 48 h. TomTom "usual" repeats weekly -> take it from t-7d.
    model.fit(data[FEATS], data.congestion)
    usual = data.set_index(["city", "timestamp"]).usual
    mm = data[data.city == TARGET_CITY]
    fut = pd.DataFrame({"timestamp": pd.date_range(mm.timestamp.max() + pd.Timedelta(hours=1), periods=48, freq="h"),
                        "city": TARGET_CITY})
    fut["usual"] = [usual.get((TARGET_CITY, t - pd.Timedelta(days=7)), np.nan) for t in fut.timestamp]
    fut = add_calendar(fut.merge(wx, on=["city", "timestamp"], how="left").fillna({"rain_mm": 0, "rain_3h": 0}))
    fut["forecast"] = model.predict(fut[FEATS]).clip(0)
    resid = te.congestion - te.pred
    fut["lo"] = (fut.forecast + resid.quantile(0.1)).clip(0)
    fut["hi"] = fut.forecast + resid.quantile(0.9)
    return data, te, fut, metrics, model
