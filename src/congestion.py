"""Live congestion model: real TomTom hourly congestion for Metro Manila (this week)
+ TomTom 2025 typical hour x weekday profile + live Open-Meteo rainfall + PH holidays.
"""
import json
import urllib.request

import holidays
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src import live_traffic, tomtom_index

WX = ("https://api.open-meteo.com/v1/forecast?latitude=14.5995&longitude=120.9842"
      "&hourly=precipitation&past_days={past}&forecast_days=3&timezone=Asia%2FManila")
DAYS = ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"]
FEATS = ["hour", "dow", "typical_2025", "rain_mm", "rain_3h", "is_holiday"]


def typical_profile() -> pd.DataFrame:
    wh = pd.DataFrame(tomtom_index.fetch()["metro"]["all"]["weekHour"])
    wh[["day", "hour"]] = wh.time.str.split("-", expand=True)
    wh["dow"] = wh.day.map(DAYS.index)
    wh["hour"] = wh.hour.astype(int)
    return wh.rename(columns={"c": "typical_2025", "v": "typical_speed_2025"})[
        ["dow", "hour", "typical_2025", "typical_speed_2025"]]


def weather(past_days: int) -> pd.DataFrame:
    with urllib.request.urlopen(WX.format(past=min(past_days, 92)), timeout=30) as r:
        h = json.load(r)["hourly"]
    w = pd.DataFrame({"timestamp": pd.to_datetime(h["time"]), "rain_mm": h["precipitation"]})
    w["rain_mm"] = w.rain_mm.fillna(0)
    w["rain_3h"] = w.rain_mm.rolling(3, min_periods=1).sum()
    return w


def features(ts: pd.Series, wx: pd.DataFrame, prof: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame({"timestamp": ts.values})
    f["hour"], f["dow"] = f.timestamp.dt.hour, f.timestamp.dt.dayofweek
    ph = holidays.PH(years=sorted(f.timestamp.dt.year.unique()))
    f["is_holiday"] = f.timestamp.dt.date.map(lambda d: d in ph).astype(int)
    f = f.merge(prof, on=["dow", "hour"], how="left").merge(wx, on="timestamp", how="left")
    return f.fillna({"rain_mm": 0, "rain_3h": 0})


def run(test_hours: int = 48):
    live = live_traffic.collect()
    mm = live[live.city == "Metro Manila"].sort_values("timestamp").reset_index(drop=True)
    days = (pd.Timestamp.now().normalize() - mm.timestamp.min().normalize()).days + 2
    wx, prof = weather(days), typical_profile()
    data = features(mm.timestamp, wx, prof)
    data["congestion"] = mm.traffic_index_live.values
    data["usual"] = mm.traffic_index_usual.values

    tr, te = data.iloc[:-test_hours], data.iloc[-test_hours:].copy()
    model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=8,
                                          min_samples_leaf=5, random_state=42)
    model.fit(tr[FEATS], tr.congestion)
    te["pred"] = model.predict(te[FEATS])
    mae = lambda p: float(np.mean(np.abs(te.congestion - p)))
    metrics = {"train_hours": len(tr), "test_hours": len(te),
               "test_period": f"{te.timestamp.min()} -> {te.timestamp.max()}",
               "MAE_model": mae(te.pred), "MAE_tomtom_usual": mae(te.usual),
               "MAE_typical_2025": mae(te.typical_2025)}

    # refit on everything, forecast the next 48 hours
    model.fit(data[FEATS], data.congestion)
    fut_ts = pd.Series(pd.date_range(data.timestamp.max() + pd.Timedelta(hours=1), periods=48, freq="h"))
    fut = features(fut_ts, wx, prof)
    fut["forecast"] = model.predict(fut[FEATS]).clip(0)
    resid = te.congestion - te.pred
    fut["lo"] = (fut.forecast + resid.quantile(0.1)).clip(0)
    fut["hi"] = fut.forecast + resid.quantile(0.9)
    return data, te, fut, metrics, live, prof
