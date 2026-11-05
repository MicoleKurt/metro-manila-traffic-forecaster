"""Collect live real traffic speeds on Metro Manila corridors from the TomTom
Traffic Flow API and append them to live/speeds.csv.

Run hourly (see .github/workflows/collect.yml) to build a real speed history.
Needs a free TomTom key (developer.tomtom.com, 2,500 free calls/day) in the
TOMTOM_API_KEY environment variable. 12 points x 24 h = 288 calls/day.
"""
import csv
import json
import os
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

URL = ("https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
       "?point={lat},{lon}&unit=KMPH&key={key}")

# (road, segment, lat, lon) - points sit on the carriageway
POINTS = [
    ("EDSA", "Monumento", 14.6547, 120.9845),
    ("EDSA", "North Ave", 14.6566, 121.0320),
    ("EDSA", "Cubao", 14.6194, 121.0525),
    ("EDSA", "Ortigas", 14.5876, 121.0567),
    ("EDSA", "Guadalupe", 14.5667, 121.0459),
    ("EDSA", "Ayala", 14.5491, 121.0280),
    ("EDSA", "Taft/MOA", 14.5378, 121.0012),
    ("C5", "Libis", 14.6060, 121.0800),
    ("C5", "Market! Market!", 14.5490, 121.0560),
    ("Commonwealth Ave", "Philcoa", 14.6545, 121.0560),
    ("España Blvd", "UST", 14.6100, 120.9890),
    ("Roxas Blvd", "Baclaran", 14.5370, 120.9930),
]
FIELDS = ["timestamp", "road", "segment", "lat", "lon", "current_speed_kph",
          "free_flow_speed_kph", "current_travel_time_s", "free_flow_travel_time_s",
          "confidence", "road_closure"]
OUT = Path("live/speeds.csv")


def main():
    key = os.environ.get("TOMTOM_API_KEY")
    if not key:
        sys.exit("Set TOMTOM_API_KEY (free at developer.tomtom.com)")
    now = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M")
    rows = []
    for road, seg, lat, lon in POINTS:
        try:
            with urllib.request.urlopen(URL.format(lat=lat, lon=lon, key=key), timeout=20) as r:
                d = json.load(r)["flowSegmentData"]
        except Exception as e:  # keep going if one point fails
            print("failed", road, seg, e)
            continue
        rows.append([now, road, seg, lat, lon, d["currentSpeed"], d["freeFlowSpeed"],
                     d["currentTravelTime"], d["freeFlowTravelTime"], d["confidence"],
                     d.get("roadClosure", False)])
    OUT.parent.mkdir(exist_ok=True)
    new = not OUT.exists()
    with OUT.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(FIELDS)
        w.writerows(rows)
    print(f"{now}: wrote {len(rows)} rows")


if __name__ == "__main__":
    main()
