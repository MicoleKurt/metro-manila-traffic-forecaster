"""Real-data pipeline: download -> features -> model -> charts.  `python run.py`"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance

from src.data import download, hourly_table, load_incidents, QUARANTINE

OUT = Path("assets"); OUT.mkdir(exist_ok=True)
BG, FG, GRID, MUTED = "#0d1117", "#e6edf3", "#30363d", "#8b949e"
ACC, ACC2, ACC3 = "#f2c94c", "#2f80ed", "#eb5757"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": FG, "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG,
    "axes.edgecolor": GRID, "grid.color": GRID, "font.size": 13,
    "axes.titleweight": "bold", "axes.titlesize": 18,
})
FEATS = ["hour", "dow", "month", "is_holiday", "is_payday", "quarantine", "rain_mm", "rain_3h"]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
NOTE = "Data: MMDA incident alerts (via Tweet2Map), Open-Meteo ERA5 rainfall"


def save(fig, name):
    fig.text(0.99, 0.01, NOTE, ha="right", fontsize=9, color=MUTED)
    fig.savefig(OUT / name, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print("  saved", name)


def main():
    print("1/4 data")
    download()
    inc = load_incidents()
    df = hourly_table(inc)
    ok = df[df.gap_day == 0]
    print(f"  {len(inc):,} incidents, {ok.timestamp.dt.date.nunique()} usable days")

    print("2/4 model (Poisson gradient boosting, time split)")
    cut = pd.Timestamp("2020-09-01")
    tr, te = ok[ok.timestamp < cut], ok[ok.timestamp >= cut].copy()
    # baseline: average for this hour & weekday, recent regime (GCQ months of 2020)
    recent = tr[tr.timestamp >= "2020-06-01"]
    te["baseline"] = te.join(recent.groupby(["dow", "hour"]).incidents.mean().rename("b"),
                             on=["dow", "hour"]).b.fillna(recent.incidents.mean())
    model = HistGradientBoostingRegressor(loss="poisson", max_iter=300, learning_rate=0.05,
                                          max_leaf_nodes=15, random_state=42)
    model.fit(tr[FEATS], tr.incidents)
    te["pred"] = model.predict(te[FEATS])
    dte = te.groupby(te.timestamp.dt.date)[["incidents", "pred", "baseline"]].sum()
    metrics = {
        "test_period": "2020-09-01 to 2020-12-27",
        "hourly_MAE": {"baseline": mean_absolute_error(te.incidents, te.baseline),
                       "model": mean_absolute_error(te.incidents, te.pred)},
        "hourly_poisson_deviance": {"baseline": mean_poisson_deviance(te.incidents, te.baseline.clip(1e-6)),
                                    "model": mean_poisson_deviance(te.incidents, te.pred)},
        "daily_MAE": {"baseline": mean_absolute_error(dte.incidents, dte.baseline),
                      "model": mean_absolute_error(dte.incidents, dte.pred)},
        "mean_daily_incidents_test": dte.incidents.mean(),
    }
    Path("results.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

    print("3/4 charts")
    pre = ok[ok.quarantine == 0]

    # 1 heatmap (pre-pandemic)
    piv = pre.pivot_table("incidents", "dow", "hour", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(16, 6))
    im = ax.imshow(piv, aspect="auto", cmap="inferno")
    ax.set_yticks(range(7), DAYS)
    ax.set_xticks(range(0, 24, 2), [f"{h}:00" for h in range(0, 24, 2)])
    ax.set_title("When do Metro Manila road incidents happen? (avg per hour, pre-pandemic)")
    fig.colorbar(im, ax=ax, label="incidents / hour")
    save(fig, "01_heatmap.png")

    # 2 COVID timeline
    daily = df.groupby(df.timestamp.dt.date).agg(n=("incidents", "sum"), gap=("gap_day", "max"))
    daily.index = pd.to_datetime(daily.index)
    roll = daily.n.where(daily.gap == 0).rolling(7, min_periods=3).mean()
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(roll.index, roll, color=ACC, lw=2)
    colors = {1: "#2f80ed33", 2: "#f2994a44", 3: "#eb575755"}
    labels = {1: "GCQ", 2: "MECQ", 3: "ECQ"}
    seen = set()
    for a, b, lvl in QUARANTINE:
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=colors[lvl],
                   label=None if lvl in seen else labels[lvl])
        seen.add(lvl)
    ax.set_title("Daily MMDA-reported incidents (7-day avg) and NCR lockdown phases")
    ax.set_ylabel("incidents / day"); ax.grid(alpha=.4); ax.legend(loc="upper left")
    save(fig, "02_covid_timeline.png")

    # 3 hotspot map
    pts = inc[(inc.Latitude.between(14.35, 14.80)) & (inc.Longitude.between(120.90, 121.15))]
    fig, ax = plt.subplots(figsize=(10, 12))
    hb = ax.hexbin(pts.Longitude, pts.Latitude, gridsize=70, bins="log", cmap="inferno", mincnt=1)
    ax.set_aspect("equal"); ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title("Incident hotspots, Metro Manila\n(17k geocoded MMDA alerts, 2018-2020)")
    fig.colorbar(hb, ax=ax, label="incidents (log)", shrink=.6)
    save(fig, "03_hotspot_map.png")

    # 4 top locations
    top = inc.Location.str.upper().str.strip().value_counts().head(15)[::-1]
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.barh(top.index, top.values, color=ACC)
    ax.set_title("Top 15 incident locations (MMDA alerts, 2018-2020)")
    ax.grid(axis="x", alpha=.4)
    save(fig, "04_top_locations.png")

    # 5 rain effect
    p = pre.copy()
    p["rain_bin"] = pd.cut(p.rain_mm, [-0.01, 0.0, 1, 5, 100], labels=["Dry", "<1 mm", "1-5 mm", ">5 mm"])
    rb = p.groupby("rain_bin", observed=True).incidents.agg(["mean", "count"])
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(rb.index.astype(str), rb["mean"], color=[ACC2, ACC, "#f2994a", ACC3])
    for i, (m, c) in enumerate(zip(rb["mean"], rb["count"])):
        ax.text(i, m, f"{m:.2f}\n(n={c:,} h)", ha="center", va="bottom", fontsize=11)
    ax.set_title("Incidents per hour vs rainfall that hour (pre-pandemic)")
    ax.set_ylabel("avg incidents / hour"); ax.grid(axis="y", alpha=.4)
    save(fig, "05_rain_effect.png")

    # 6 incident types
    cat = inc.category.value_counts()[::-1]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(cat.index, cat.values, color=ACC2)
    for i, v in enumerate(cat.values):
        ax.text(v, i, f" {v:,} ({v / cat.sum():.0%})", va="center")
    ax.set_title("What kind of incidents? (MMDA alerts, 2018-2020)")
    save(fig, "06_incident_types.png")

    # 7 actual vs predicted (daily)
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(pd.to_datetime(dte.index), dte.incidents, color=FG, lw=1.5, label="Actual")
    ax.plot(pd.to_datetime(dte.index), dte.pred, color=ACC, lw=2.2, label="Model")
    ax.plot(pd.to_datetime(dte.index), dte.baseline, color=ACC2, lw=1.2, ls="--", label="Hour x weekday average")
    ax.set_title(f"Unseen test period (Sep-Dec 2020): daily incidents, model MAE "
                 f"{metrics['daily_MAE']['model']:.1f} vs baseline {metrics['daily_MAE']['baseline']:.1f}")
    ax.set_ylabel("incidents / day"); ax.grid(alpha=.4); ax.legend()
    save(fig, "07_actual_vs_pred.png")

    # 8 feature importance
    pi = permutation_importance(model, te[FEATS], te.incidents, n_repeats=5,
                                scoring="neg_mean_poisson_deviance", random_state=0)
    imp = pd.Series(pi.importances_mean, FEATS).sort_values()
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(imp.index, imp.values, color=ACC)
    ax.set_title("What drives incident counts? (permutation importance)")
    ax.grid(axis="x", alpha=.4)
    save(fig, "08_feature_importance.png")
    live_section(metrics)
    print("4/4 done")


def live_section(metrics):
    """Real live TomTom congestion (this week) + 2025 TomTom profile + 48 h forecast."""
    global NOTE
    from src.congestion import run as run_live
    print("live: TomTom congestion this week")
    data, te, fut, m, live, prof = run_live()
    metrics["live_congestion"] = m
    Path("results.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(json.dumps(m, indent=2, default=str))
    NOTE = "Data: TomTom Traffic Index (live + 2025), Open-Meteo rainfall"

    # 9 this week: live vs usual
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(data.timestamp, data.congestion, color=ACC3, lw=2.2, label="Live (this week)")
    ax.plot(data.timestamp, data.usual, color=MUTED, lw=1.5, ls="--", label="TomTom usual for that hour")
    ax.set_title(f"Metro Manila congestion, {data.timestamp.min():%b %d} - {data.timestamp.max():%b %d, %Y}")
    ax.set_ylabel("congestion level (%)"); ax.grid(alpha=.4); ax.legend()
    save(fig, "09_live_this_week.png")

    # 10 jams
    mm = live[live.city == "Metro Manila"]
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.fill_between(mm.timestamp, mm.jams_length_km, color=ACC, alpha=.6)
    ax.set_title(f"Total traffic-jam length in Metro Manila this week (peak {mm.jams_length_km.max():.0f} km)")
    ax.set_ylabel("km of jams"); ax.grid(alpha=.4)
    save(fig, "10_jams_this_week.png")

    # 11 TomTom 2025 profile heatmap (real)
    piv = prof.pivot_table("typical_speed_2025", "dow", "hour")
    fig, ax = plt.subplots(figsize=(16, 6))
    im = ax.imshow(piv, aspect="auto", cmap="RdYlGn")
    ax.set_yticks(range(7), DAYS)
    ax.set_xticks(range(0, 24, 2), [f"{h}:00" for h in range(0, 24, 2)])
    ax.set_title("Metro Manila average speed by hour and weekday, full year 2025 (km/h)")
    fig.colorbar(im, ax=ax, label="km/h")
    save(fig, "11_speed_profile_2025.png")

    # 12 forecast
    fig, ax = plt.subplots(figsize=(16, 7))
    hist = data[data.timestamp >= data.timestamp.max() - pd.Timedelta(hours=48)]
    ax.plot(hist.timestamp, hist.congestion, color=FG, lw=2, label="Actual (live)")
    ax.fill_between(fut.timestamp, fut.lo, fut.hi, color=ACC, alpha=.25, label="80% interval")
    ax.plot(fut.timestamp, fut.forecast, color=ACC, lw=2.4, label="Forecast")
    ax.set_title(f"Next 48 hours: Metro Manila congestion forecast (test MAE {m['MAE_model']:.1f} pts "
                 f"vs {m['MAE_tomtom_usual']:.1f} for TomTom 'usual')")
    ax.set_ylabel("congestion level (%)"); ax.grid(alpha=.4); ax.legend()
    save(fig, "12_forecast_48h.png")


if __name__ == "__main__":
    main()
