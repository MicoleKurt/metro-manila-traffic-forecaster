"""2026 pipeline: collect live data -> train -> evaluate -> charts.  `python run.py`"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.congestion import FEATS, TARGET_CITY, run

OUT = Path("assets"); OUT.mkdir(exist_ok=True)
BG, FG, GRID, MUTED = "#0d1117", "#e6edf3", "#30363d", "#8b949e"
ACC, ACC2, ACC3 = "#f2c94c", "#2f80ed", "#eb5757"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": FG, "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG,
    "axes.edgecolor": GRID, "grid.color": GRID, "font.size": 13,
    "axes.titleweight": "bold", "axes.titlesize": 18,
})
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
NOTE = "Data: TomTom Traffic Index live feed, Open-Meteo rainfall, PH holidays (2026)"


def save(fig, name):
    fig.text(0.99, 0.01, NOTE, ha="right", fontsize=9, color=MUTED)
    fig.savefig(OUT / name, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print("  saved", name)


def main():
    refresh = "--refresh" in sys.argv
    print("1/3 training on bundled 2026 data" + (" (refreshed)" if refresh else ""))
    data, te, fut, m, model = run(refresh=refresh)
    Path("results.json").write_text(json.dumps(m, indent=2, default=str))
    print(json.dumps(m, indent=2, default=str))
    mm = data[data.city == TARGET_CITY]
    span = f"{mm.timestamp.min():%b %d} - {mm.timestamp.max():%b %d, %Y}"

    print("2/3 charts")
    for f in OUT.glob("*.png"):
        f.unlink()

    # 1 live vs usual
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(mm.timestamp, mm.congestion, color=ACC3, lw=2.2, label="Live")
    ax.plot(mm.timestamp, mm.usual, color=MUTED, lw=1.5, ls="--", label="TomTom usual for that hour")
    for d in mm[mm.is_holiday == 1].timestamp.dt.normalize().unique():
        ax.axvspan(d, d + pd.Timedelta(days=1), color="#2f80ed22")
    ax.set_title(f"Metro Manila congestion, {span}  (blue = holiday)")
    ax.set_ylabel("congestion level (%)"); ax.grid(alpha=.4); ax.legend()
    save(fig, "01_live_vs_usual.png")

    # 2 hour x day heatmap of this period
    piv = mm.assign(day=mm.timestamp.dt.strftime("%a %b %d")).pivot_table(
        "congestion", "day", "hour", sort=False).reindex(columns=range(24))
    fig, ax = plt.subplots(figsize=(16, 7))
    im = ax.imshow(piv, aspect="auto", cmap="inferno")
    ax.set_yticks(range(len(piv)), piv.index)
    ax.set_xticks(range(0, 24, 2), [f"{h}:00" for h in range(0, 24, 2)])
    ax.set_title("Metro Manila congestion by hour, day by day (%)")
    fig.colorbar(im, ax=ax, label="congestion %")
    save(fig, "02_heatmap_by_day.png")

    # 3 jams
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.fill_between(mm.timestamp, mm.jams_length_km, color=ACC, alpha=.6)
    pk = mm.loc[mm.jams_length_km.idxmax()]
    ax.annotate(f"{pk.jams_length_km:.0f} km of jams\n{pk.timestamp:%a %I %p}", (pk.timestamp, pk.jams_length_km),
                xytext=(10, -40), textcoords="offset points", color=FG)
    ax.set_title(f"Total traffic-jam length, Metro Manila ({span})")
    ax.set_ylabel("km"); ax.grid(alpha=.4)
    save(fig, "03_jams.png")

    # 4 Manila vs Davao
    fig, ax = plt.subplots(figsize=(16, 7))
    for c, col in [("Metro Manila", ACC3), ("Davao City", ACC2)]:
        g = data[data.city == c]
        ax.plot(g.timestamp, g.congestion, color=col, lw=2, label=f"{c} (avg {g.congestion.mean():.0f}%)")
    ax.set_title("Metro Manila vs Davao City, live congestion")
    ax.set_ylabel("congestion level (%)"); ax.grid(alpha=.4); ax.legend()
    save(fig, "04_manila_vs_davao.png")

    # 5 weekday vs weekend vs holiday profile
    k = mm.assign(kind=np.select([mm.is_holiday == 1, mm.dow >= 5], ["Holiday", "Weekend"], "Weekday"))
    fig, ax = plt.subplots(figsize=(16, 7))
    for kind, col in [("Weekday", ACC3), ("Weekend", ACC), ("Holiday", ACC2)]:
        s = k[k.kind == kind].groupby("hour").congestion.mean()
        if len(s):
            ax.plot(s.index, s.values, lw=2.5, color=col, label=kind)
    ax.set_title("Average congestion by hour: weekday vs weekend vs holiday")
    ax.set_xlabel("hour"); ax.set_ylabel("congestion %"); ax.grid(alpha=.4); ax.legend()
    save(fig, "05_daytype_profile.png")

    # 6 test: actual vs predicted
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(te.timestamp, te.congestion, color=FG, lw=2, label="Actual")
    ax.plot(te.timestamp, te.pred, color=ACC, lw=2.4, label=f"Model (MAE {m['MAE_model']:.1f})")
    ax.plot(te.timestamp, te.usual, color=MUTED, lw=1.5, ls="--", label=f"TomTom usual (MAE {m['MAE_tomtom_usual']:.1f})")
    ax.set_title("Held-out test: last 48 hours, Metro Manila")
    ax.set_ylabel("congestion %"); ax.grid(alpha=.4); ax.legend()
    save(fig, "06_test_actual_vs_pred.png")

    # 7 forecast
    hist = mm[mm.timestamp >= mm.timestamp.max() - pd.Timedelta(hours=48)]
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(hist.timestamp, hist.congestion, color=FG, lw=2, label="Actual (live)")
    ax.fill_between(fut.timestamp, fut.lo, fut.hi, color=ACC, alpha=.25, label="80% interval")
    ax.plot(fut.timestamp, fut.forecast, color=ACC, lw=2.4, label="Forecast")
    ax.set_title(f"Next 48 hours: Metro Manila congestion forecast (from {fut.timestamp.min():%b %d %I %p})")
    ax.set_ylabel("congestion %"); ax.grid(alpha=.4); ax.legend()
    save(fig, "07_forecast_48h.png")

    # 8 importance
    pi = permutation_importance(model, data[FEATS], data.congestion, n_repeats=5, random_state=0,
                                scoring="neg_mean_absolute_error")
    imp = pd.Series(pi.importances_mean, FEATS).sort_values()
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(imp.index, imp.values, color=ACC)
    ax.set_title("What drives congestion? (permutation importance, MAE pts)")
    ax.grid(axis="x", alpha=.4)
    save(fig, "08_feature_importance.png")
    print("3/3 done")



if __name__ == "__main__":
    main()
