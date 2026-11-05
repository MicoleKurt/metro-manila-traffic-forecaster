"""Synthetic hourly traffic data for Metro Manila corridors.

Calibrated to publicly reported patterns (MMDA, TomTom Traffic Index, PAGASA).
It is SYNTHETIC. Replace `generate()` with `load_real(path)` once you have real
exports that share the same schema:
    timestamp, road, speed_kph, rain_mm, typhoon
"""
import numpy as np
import pandas as pd

SEED = 42
# Speed curve params, grid-searched so that EDSA on non-holiday weekdays averages
# ~21 km/h from 6am-10pm (MMDA, Dec 2024) and ~15 km/h at 5-8pm rush
# (TomTom Traffic Index 2025, Manila rush-hour speed 15.2 km/h).
ALPHA, BETA, MID = 4.0, 1.3, 0.6

# free-flow speed (km/h; EDSA limit 60) and relative congestion severity
ROADS = {
    "EDSA":            (60, 0.72),
    "C5":              (60, 0.62),
    "Commonwealth Ave": (60, 0.58),
    "España Blvd":     (40, 0.55),
    "Roxas Blvd":      (60, 0.45),
    "SLEX":            (80, 0.50),
    "NLEX":            (90, 0.40),
}

# Fixed-date regular + special non-working days (Proclamations 90/2022, 368/2023,
# 727/2024, 1006/2025). Movable dates are listed per year below.
HOLIDAYS = ["01-01", "05-01", "06-12", "08-21", "11-01", "11-02", "11-30",
            "12-08", "12-24", "12-25", "12-30", "12-31"]
MOVABLE_HOLIDAYS = {
    # Chinese New Year, Araw ng Kagitingan (moved if needed), National Heroes Day
    "2023-01-23", "2023-04-10", "2023-08-28",
    "2024-02-10", "2024-04-09", "2024-08-26",
    "2025-01-29", "2025-04-09", "2025-08-25",
    "2026-02-17", "2026-04-09", "2026-08-31",
}
HOLY_WEEK = {  # Maundy Thu - Easter Sun
    2023: ("2023-04-06", "2023-04-09"),
    2024: ("2024-03-28", "2024-03-31"),
    2025: ("2025-04-17", "2025-04-20"),
    2026: ("2026-04-02", "2026-04-05"),
}


def calendar_features(ts: pd.DatetimeIndex) -> pd.DataFrame:
    df = pd.DataFrame({"timestamp": ts})
    df["hour"] = ts.hour
    df["dow"] = ts.dayofweek
    df["month"] = ts.month
    df["day"] = ts.day
    df["is_weekend"] = (df.dow >= 5).astype(int)
    md = ts.strftime("%m-%d")
    df["is_holiday"] = (md.isin(HOLIDAYS) |
                        ts.strftime("%Y-%m-%d").isin(MOVABLE_HOLIDAYS)).astype(int)
    hw = np.zeros(len(ts), dtype=int)
    for y, (a, b) in HOLY_WEEK.items():
        hw |= ((ts >= a) & (ts < pd.Timestamp(b) + pd.Timedelta(days=1))).astype(int)
    df["holy_week"] = hw
    last = ts.days_in_month
    df["is_payday"] = ((ts.day == 15) | (ts.day == last) | (ts.day == 30)).astype(int)
    df["ber_month"] = (ts.month >= 9).astype(int)
    # UVVRP number coding window (7-10am, 5-8pm weekdays)
    df["coding_hours"] = (
        (df.dow < 5) & (df.hour.between(7, 9) | df.hour.between(17, 19))
    ).astype(int)
    return df


def weather(ts: pd.DatetimeIndex, rng) -> pd.DataFrame:
    days = pd.DatetimeIndex(ts.normalize().unique())
    wet = days.month.isin([6, 7, 8, 9, 10, 11])
    p_rain = np.where(wet, 0.55, 0.12)
    rainy = rng.random(len(days)) < p_rain
    # typhoon days: clustered, mostly Jul-Nov
    typh = np.zeros(len(days), dtype=int)
    for i in np.where(days.month.isin([7, 8, 9, 10, 11]))[0]:
        if rng.random() < 0.012:
            typh[i:i + rng.integers(1, 4)] = 1
    daily = pd.DataFrame({"date": days, "rainy": rainy, "typhoon": typh})
    df = pd.DataFrame({"date": ts.normalize(), "hour": ts.hour})
    df = df.merge(daily, on="date", how="left")
    # afternoon convective showers peak 2-6pm
    afternoon = 0.35 + 0.65 * np.exp(-((df.hour - 16) ** 2) / 12)
    base = rng.gamma(1.5, 3.0, len(df)) * afternoon * df.rainy
    base += df.typhoon * rng.gamma(3, 6, len(df))
    df["rain_mm"] = np.round(base, 1)
    return df[["rain_mm", "typhoon"]]


def generate(start="2023-01-01", end="2026-10-31 23:00") -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    ts = pd.date_range(start, end, freq="h")
    cal = calendar_features(ts)
    wx = weather(ts, rng)
    h = cal.hour.values

    # demand curve: AM peak ~8am, PM peak ~6-7pm, low at 3am
    am = np.exp(-((h - 8) ** 2) / 3.0)
    pm = np.exp(-((h - 18.5) ** 2) / 4.0)
    mid = MID * np.exp(-((h - 13) ** 2) / 14.0)
    demand = 0.15 + 0.85 * am + 1.0 * pm + mid

    mult = np.ones(len(ts))
    mult *= np.where(cal.is_weekend, 0.68, 1.0)
    mult *= np.where((cal.dow == 4) & (h >= 16), 1.15, 1.0)          # Friday
    mult *= np.where(cal.is_payday == 1, 1.08, 1.0)
    mult *= np.where(cal.is_holiday == 1, 0.70, 1.0)
    mult *= np.where(cal.holy_week == 1, 0.45, 1.0)
    ber = np.select([cal.month == 12, cal.month >= 9], [1.20, 1.06], 1.0)
    mult *= ber
    mult *= np.where((cal.month == 12) & (cal.day.between(15, 23)), 1.08, 1.0)
    mult *= np.where(cal.coding_hours == 1, 0.96, 1.0)
    trend = 1 + 0.02 * (ts.year - 2023)                                  # vehicle growth
    # Mar-Apr 2026: MMDA reported 5-7% lower EDSA volume amid oil-price spike
    trend = trend * np.where((ts >= "2026-03-09") & (ts < "2026-05-01"), 0.94, 1.0)
    load = demand * mult * trend

    frames = []
    for road, (free, sev) in ROADS.items():
        r = load * sev
        r = r * (1 + 0.035 * wx.rain_mm.clip(0, 20).values) * (1 + 0.35 * wx.typhoon.values)
        # BPR-style speed drop from load
        speed = free / (1 + ALPHA * r ** BETA)
        speed *= np.exp(rng.normal(0, 0.07, len(ts)))
        speed = np.clip(speed, 4, free)
        f = cal.copy()
        f["road"] = road
        f["rain_mm"] = wx.rain_mm.values
        f["typhoon"] = wx.typhoon.values
        f["speed_kph"] = speed.round(1)
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def load_real(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path, parse_dates=["timestamp"])
    cal = calendar_features(pd.DatetimeIndex(raw.timestamp))
    return pd.concat([cal, raw.drop(columns="timestamp")], axis=1)

