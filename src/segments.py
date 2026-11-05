"""Per-road-segment live speeds (which part of EDSA is jammed) from the TomTom
Traffic Flow API. Appends to live/segments.csv.

Needs a free key from developer.tomtom.com in TOMTOM_API_KEY.
25 points x 24 runs/day = 600 calls/day (free tier: 2,500/day).
"""
import json
import os
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

URL = ("https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/12/json"
       "?point={lat},{lon}&unit=KMPH&key={key}")
OUT = Path("live/segments.csv")

# (road, segment, lat, lon), ordered north -> south along each road
SEGMENTS = [
    ("EDSA", "Monumento", 14.6545, 120.9840),
    ("EDSA", "Balintawak", 14.6574, 121.0040),
    ("EDSA", "Roosevelt / Muñoz", 14.6573, 121.0200),
    ("EDSA", "North Ave / SM North", 14.6566, 121.0315),
    ("EDSA", "Quezon Ave", 14.6430, 121.0380),
    ("EDSA", "Kamuning", 14.6330, 121.0435),
    ("EDSA", "Cubao / Aurora", 14.6200, 121.0520),
    ("EDSA", "Santolan / Camp Crame", 14.6080, 121.0560),
    ("EDSA", "Ortigas", 14.5880, 121.0570),
    ("EDSA", "Shaw Blvd", 14.5810, 121.0535),
    ("EDSA", "Guadalupe", 14.5665, 121.0455),
    ("EDSA", "Buendia", 14.5540, 121.0345),
    ("EDSA", "Ayala", 14.5490, 121.0280),
    ("EDSA", "Magallanes", 14.5420, 121.0195),
    ("EDSA", "Taft / Pasay Rotonda", 14.5375, 121.0010),
    ("EDSA", "Roxas Blvd / MOA", 14.5355, 120.9925),
    ("C5", "Katipunan", 14.6380, 121.0740),
    ("C5", "Libis / Eastwood", 14.6080, 121.0800),
    ("C5", "Kalayaan / BGC", 14.5560, 121.0580),
    ("Commonwealth Ave", "Philcoa", 14.6545, 121.0560),
    ("Commonwealth Ave", "Fairview", 14.6990, 121.0850),
    ("Quezon Ave", "Welcome Rotonda", 14.6210, 121.0010),
    ("España Blvd", "UST", 14.6100, 120.9890),
    ("Marcos Highway", "Santolan", 14.6220, 121.0950),
    ("SLEX", "Alabang", 14.4230, 121.0450),
]


def collect(key: str) -> pd.DataFrame:
    ts = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:00")
    rows = []
    for road, seg, lat, lon in SEGMENTS:
        try:
            with urllib.request.urlopen(URL.format(lat=lat, lon=lon, key=key), timeout=20) as r:
                d = json.load(r)["flowSegmentData"]
        except Exception as e:
            print("failed", road, seg, e)
            continue
        rows.append({"timestamp": ts, "road": road, "segment": seg, "lat": lat, "lon": lon,
                     "speed_kph": d["currentSpeed"], "free_flow_kph": d["freeFlowSpeed"],
                     "travel_time_s": d["currentTravelTime"], "free_flow_time_s": d["freeFlowTravelTime"],
                     "confidence": d["confidence"], "closed": d.get("roadClosure", False)})
    new = pd.DataFrame(rows)
    OUT.parent.mkdir(exist_ok=True)
    if OUT.exists():
        new = pd.concat([pd.read_csv(OUT), new]).drop_duplicates(["timestamp", "road", "segment"], keep="last")
    new.to_csv(OUT, index=False)
    return new


def load() -> pd.DataFrame | None:
    if not OUT.exists():
        return None
    s = pd.read_csv(OUT, parse_dates=["timestamp"])
    s["congestion_pct"] = (s.travel_time_s / s.free_flow_time_s - 1).clip(lower=0) * 100
    s["speed_ratio"] = s.speed_kph / s.free_flow_kph
    return s


if __name__ == "__main__":
    k = os.environ.get("TOMTOM_API_KEY")
    if not k:
        sys.exit("TOMTOM_API_KEY not set; skipping segment collection")
    df = collect(k)
    print(f"{len(df)} segment readings stored")
