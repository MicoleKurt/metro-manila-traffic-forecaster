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
FEATS = ["hour", "is_weekend", "city_id", "usual", "rain_mm", "rain_3h", "is_holiday"]
# The model predicts the EXCESS over TomTom's usual (congestion - usual). "usual" is TomTom's
# long-run typical level for that weekday & hour, so the model never has to learn what a
# normal Monday looks like from one week of data; it learns how holidays, weekends, rain
# and time of day push traffic above or below normal.
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
    f["is_weekend"] = (f.dow >= 5).astype(int)
    f["city_id"] = f.city.map(CITY_ID)
    return f


WX_FILE = Path("live/weather.csv")


def run(test_hours: int = 48, refresh: bool = False, start=None, days: float = 7, city: str = TARGET_CITY):
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
    model.fit(tr[FEATS], tr.congestion - tr.usual)
    te["pred"] = (te.usual + model.predict(te[FEATS])).clip(0)
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

    model.fit(data[FEATS], data.congestion - data.usual)  # final model uses every hour
    resid = te.congestion - te.pred
    fut = forecast(model, data, wx, start=start, days=days, city=city)
    fut["lo"] = (fut.forecast + resid.quantile(0.1)).clip(0)
    fut["hi"] = fut.forecast + resid.quantile(0.9)
    return data, te, fut, metrics, model


def forecast(model, data, wx_snapshot, start=None, days=7, city=TARGET_CITY) -> pd.DataFrame:
    """Predict congestion for any future dates. Needs no new traffic data:
    - hour / weekday / holiday come from the calendar (official PH holidays, any year)
    - TomTom 'usual' repeats weekly, so it is learned per city x weekday x hour from the data
    - rain: live Open-Meteo forecast if online (up to 16 days), else the bundled snapshot,
      else assumed dry. `rain_source` says which one was used.
    """
    start = pd.Timestamp(start or pd.Timestamp.now(tz="Asia/Manila").tz_localize(None)).floor("D")
    fut = pd.DataFrame({"timestamp": pd.date_range(start, periods=int(days * 24), freq="h"), "city": city})
    prof = data.groupby(["city", "dow", "hour"]).usual.mean().rename("usual").reset_index()
    fut = add_calendar(fut).merge(prof, on=["city", "dow", "hour"], how="left")

    rain = None
    end_days = (fut.timestamp.max().normalize() - pd.Timestamp.now().normalize()).days + 1
    if 0 < end_days <= 16:
        try:
            lat, lon = COORDS[city]
            url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
                   f"&hourly=precipitation&forecast_days={end_days}&past_days=1&timezone=Asia%2FManila")
            with urllib.request.urlopen(url, timeout=15) as r:
                h = json.load(r)["hourly"]
            rain = pd.DataFrame({"timestamp": pd.to_datetime(h["time"]), "rain_mm": h["precipitation"]})
            src = "Open-Meteo live forecast"
        except Exception:
            rain = None
    if rain is None:
        rain = wx_snapshot[wx_snapshot.city == city][["timestamp", "rain_mm"]]
        src = "bundled snapshot / assumed dry"
    rain = rain.copy()
    rain["rain_mm"] = rain.rain_mm.fillna(0)
    rain["rain_3h"] = rain.rain_mm.rolling(3, min_periods=1).sum()
    fut = fut.merge(rain, on="timestamp", how="left")
    fut["rain_source"] = np.where(fut.rain_mm.isna(), "assumed dry", src)
    fut = fut.fillna({"rain_mm": 0, "rain_3h": 0})
    fut["forecast"] = (fut.usual + model.predict(fut[FEATS])).clip(0)
    return fut
