"""Real-data loaders for the Metro Manila traffic-incident forecaster.

Sources (all real, all public):
- MMDA traffic incident alerts posted on @MMDA (Aug 2018 - Dec 2020), parsed and
  geocoded by the Tweet2Map project: https://github.com/pbrotoisworo/tweet2map (GPL-3.0)
- Hourly precipitation for Manila from the Open-Meteo historical archive (ERA5):
  https://open-meteo.com/en/docs/historical-weather-api (CC BY 4.0)
- Philippine public holidays from the `holidays` package (official proclamations)
- NCR community-quarantine phases (IATF resolutions, 2020)
"""
import json
import urllib.request
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

RAW = Path("data/raw")
INCIDENTS_URL = "https://raw.githubusercontent.com/pbrotoisworo/tweet2map/master/data/data.sqlite"
WEATHER_URL = ("https://archive-api.open-meteo.com/v1/archive?latitude=14.5995&longitude=120.9842"
               "&start_date=2018-08-01&end_date=2020-12-31&hourly=precipitation"
               "&timezone=Asia%2FManila")

# NCR quarantine level: 0 none, 1 GCQ, 2 MECQ, 3 ECQ (IATF resolutions)
QUARANTINE = [
    ("2020-03-15", "2020-03-16", 1),  # "community quarantine" before ECQ
    ("2020-03-17", "2020-05-15", 3),
    ("2020-05-16", "2020-05-31", 2),
    ("2020-06-01", "2020-08-03", 1),
    ("2020-08-04", "2020-08-18", 2),
    ("2020-08-19", "2020-12-31", 1),
]


def download():
    RAW.mkdir(parents=True, exist_ok=True)
    for url, name in [(INCIDENTS_URL, "mmda_incidents.sqlite"), (WEATHER_URL, "weather.json")]:
        p = RAW / name
        if not p.exists():
            print("  downloading", name)
            urllib.request.urlretrieve(url, p)


def load_incidents() -> pd.DataFrame:
    import sqlite3
    with sqlite3.connect(RAW / "mmda_incidents.sqlite") as c:
        df = pd.read_sql("select * from INCIDENTS", c)
    df["timestamp"] = pd.to_datetime(df.Date + " " + df.Time.str.strip(),
                                     format="%Y-%m-%d %I:%M %p", errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["Type"] = df.Type.str.upper().str.strip()
    df["category"] = np.select(
        [df.Type.str.contains("ACCIDENT"), df.Type.str.contains("STALL|MECHANICAL|FLAT|BREAK"),
         df.Type.str.contains("FLOOD")],
        ["Vehicular accident", "Stalled vehicle", "Flooding"], "Other")
    df["on_edsa"] = df.Location.str.upper().str.contains("EDSA", na=False)
    return df


def load_weather() -> pd.Series:
    w = json.loads((RAW / "weather.json").read_text())["hourly"]
    return pd.Series(w["precipitation"], index=pd.to_datetime(w["time"]), name="rain_mm")


def hourly_table(inc: pd.DataFrame) -> pd.DataFrame:
    start = inc.timestamp.min().normalize()
    end = inc.timestamp.max().normalize() + pd.Timedelta(hours=23)
    idx = pd.date_range(start, end, freq="h")
    counts = inc.set_index("timestamp").resample("h").size().reindex(idx, fill_value=0)
    df = pd.DataFrame({"timestamp": idx, "incidents": counts.values})
    rain = load_weather().reindex(idx).fillna(0).values
    df["rain_mm"] = rain
    df["rain_3h"] = pd.Series(rain).rolling(3, min_periods=1).sum().values
    ph = holidays.PH(years=range(2018, 2021))
    ts = df.timestamp
    df["hour"] = ts.dt.hour
    df["dow"] = ts.dt.dayofweek
    df["month"] = ts.dt.month
    df["is_holiday"] = ts.dt.date.map(lambda d: d in ph).astype(int)
    df["is_payday"] = ((ts.dt.day == 15) | ts.dt.is_month_end | (ts.dt.day == 30)).astype(int)
    df["quarantine"] = 0
    for a, b, lvl in QUARANTINE:
        df.loc[(ts >= a) & (ts < pd.Timestamp(b) + pd.Timedelta(days=1)), "quarantine"] = lvl
    # Days with zero alerts outside the ECQ are scraping gaps, not quiet roads
    daily = df.groupby(ts.dt.date).incidents.transform("sum")
    df["gap_day"] = ((daily == 0) & (df.quarantine < 3)).astype(int)
    return df
