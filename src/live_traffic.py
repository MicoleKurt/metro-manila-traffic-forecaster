"""Collect REAL live hourly Metro Manila congestion from the public TomTom Traffic
Index feed (no API key) and append it to live/manila_hourly.csv.

The feed returns roughly the last 7 days hour by hour, so running this at least
once a day (see .github/workflows/collect.yml) builds a continuous history.

Fields (TomTom):
  traffic_index_live      congestion % right now (extra travel time vs free flow)
  traffic_index_usual     TomTom's historic typical congestion for that hour
  rel_travel_time_live    travel time as % of free-flow (e.g. 180 = 1.8x)
  jams_count / jams_length_km / jams_delay_min
Data © TomTom, https://www.tomtom.com/traffic-index/city/manila/
"""
import json
import urllib.request
from pathlib import Path

import pandas as pd

FEED = "https://www.tomtom.com/traffic-index/api/live-traffic.json?city={region}"
CITIES = {"Metro Manila": "PHL_manila_new", "Davao City": "PHL_davao-city_new"}
OUT = Path("live/manila_hourly.csv")
COLS = {"trafficIndexLive": "traffic_index_live", "trafficIndexHistoric": "traffic_index_usual",
        "relTravelTimeLive": "rel_travel_time_live", "relTravelTimeHistoric": "rel_travel_time_usual",
        "jamsCount": "jams_count", "jamsLength": "jams_length_km", "jamsDelay": "jams_delay_min"}


def fetch(region: str) -> pd.DataFrame:
    req = urllib.request.Request(FEED.format(region=region), headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        d = json.load(r)["data"]
    df = pd.DataFrame(d).rename(columns=COLS)
    df["timestamp"] = pd.to_datetime(df.date) + pd.to_timedelta(df.hour, unit="h")
    return df[["timestamp", *COLS.values()]]


def collect() -> pd.DataFrame:
    frames = []
    for city, region in CITIES.items():
        f = fetch(region)
        f.insert(1, "city", city)
        frames.append(f)
    new = pd.concat(frames)
    OUT.parent.mkdir(exist_ok=True)
    if OUT.exists():
        old = pd.read_csv(OUT, parse_dates=["timestamp"])
        new = pd.concat([old, new])
    # keep the latest reading per city-hour
    new = new.drop_duplicates(["city", "timestamp"], keep="last").sort_values(["city", "timestamp"])
    new.to_csv(OUT, index=False)
    return new


def load() -> pd.DataFrame:
    return pd.read_csv(OUT, parse_dates=["timestamp"])


if __name__ == "__main__":
    df = collect()
    for c, g in df.groupby("city"):
        print(f"{c}: {len(g)} hours, {g.timestamp.min()} -> {g.timestamp.max()}")
