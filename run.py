"""One-command pipeline: data -> model -> metrics -> charts.  `python run.py`"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data import calendar_features, generate, ROADS

OUT = Path("assets"); OUT.mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)

BG, FG, GRID = "#0d1117", "#e6edf3", "#30363d"
ACC, ACC2, ACC3 = "#f2c94c", "#2f80ed", "#eb5757"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": FG, "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG,
    "axes.edgecolor": GRID, "grid.color": GRID, "font.size": 13,
    "axes.titleweight": "bold", "axes.titlesize": 18,
})
FEATS = ["hour", "dow", "month", "day", "is_weekend", "is_holiday", "holy_week",
         "is_payday", "ber_month", "coding_hours", "rain_mm", "typhoon", "road_id"]
ROAD_IDS = {r: i for i, r in enumerate(ROADS)}


def save(fig, name, note="Synthetic data calibrated to MMDA / TomTom / PAGASA patterns"):
    fig.text(0.99, 0.01, note, ha="right", fontsize=9, color="#8b949e")
    fig.savefig(OUT / name, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print("  saved", name)


def main():
    print("1/4 generating data")
    df = generate()
    df["road_id"] = df.road.map(ROAD_IDS)
    df.sample(5000, random_state=0).to_csv("data/sample.csv", index=False)

    print("2/4 training")
    cut = pd.Timestamp("2026-05-01")
    tr, te = df[df.timestamp < cut], df[df.timestamp >= cut].copy()
    base = tr.groupby(["road", "dow", "hour"]).speed_kph.mean().rename("baseline")
    te = te.join(base, on=["road", "dow", "hour"])
    model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.08,
                                          categorical_features=[FEATS.index("road_id")],
                                          random_state=42)
    model.fit(tr[FEATS], tr.speed_kph)
    te["pred"] = model.predict(te[FEATS])

    def m(p):
        return dict(MAE=mean_absolute_error(te.speed_kph, p),
                    RMSE=mean_squared_error(te.speed_kph, p) ** 0.5,
                    R2=r2_score(te.speed_kph, p))
    metrics = {"baseline": m(te.baseline), "gbm": m(te.pred)}
    metrics["mae_improvement_pct"] = 100 * (1 - metrics["gbm"]["MAE"] / metrics["baseline"]["MAE"])
    Path("results.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

    print("3/4 charts")
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    edsa = df[df.road == "EDSA"]

    # 1 heatmap
    piv = edsa[edsa.is_holiday == 0].pivot_table("speed_kph", "dow", "hour")
    fig, ax = plt.subplots(figsize=(16, 6))
    im = ax.imshow(piv, aspect="auto", cmap="RdYlGn")
    ax.set_yticks(range(7), days); ax.set_xticks(range(0, 24, 2), [f"{h}:00" for h in range(0, 24, 2)])
    ax.set_title("EDSA average speed (km/h) by hour and weekday")
    fig.colorbar(im, ax=ax, label="km/h")
    save(fig, "01_heatmap.png")

    # 2 actual vs predicted
    wk = te[(te.road == "EDSA") & (te.timestamp >= "2026-09-07") & (te.timestamp < "2026-09-14")]
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(wk.timestamp, wk.speed_kph, color=FG, lw=1.6, label="Actual")
    ax.plot(wk.timestamp, wk.pred, color=ACC, lw=2.2, label="Model")
    ax.plot(wk.timestamp, wk.baseline, color=ACC2, lw=1.2, ls="--", label="Hist. average")
    ax.set_title(f"EDSA, one unseen test week: model MAE {metrics['gbm']['MAE']:.2f} km/h")
    ax.set_ylabel("km/h"); ax.grid(alpha=.4); ax.legend()
    save(fig, "02_actual_vs_pred.png")

    # 3 importance
    smp = te[te.road == "EDSA"]
    pi = permutation_importance(model, smp[FEATS], smp.speed_kph, n_repeats=4,
                                scoring="neg_mean_absolute_error", random_state=0)
    imp = pd.Series(pi.importances_mean, FEATS).drop("road_id").sort_values()
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.barh(imp.index, imp.values, color=ACC)
    ax.set_title("What drives EDSA traffic? (km/h of error added when feature is shuffled)")
    ax.grid(axis="x", alpha=.4)
    save(fig, "03_feature_importance.png")

    # 4 rain impact
    wd = edsa[(edsa.is_weekend == 0) & (edsa.is_holiday == 0)].copy()
    wd["cond"] = np.select([wd.typhoon == 1, wd.rain_mm > 2], ["Typhoon", "Rain"], "Dry")
    fig, ax = plt.subplots(figsize=(16, 7))
    for c, col in [("Dry", ACC2), ("Rain", ACC), ("Typhoon", ACC3)]:
        s = wd[wd.cond == c].groupby("hour").speed_kph.mean()
        ax.plot(s.index, s.values, lw=2.5, color=col, label=c)
    ax.set_title("Weekday EDSA speed: dry vs rain vs typhoon")
    ax.set_xlabel("Hour"); ax.set_ylabel("km/h"); ax.grid(alpha=.4); ax.legend()
    save(fig, "04_rain_impact.png")

    # 5 seasonality
    pk = df[(df.hour.between(17, 19)) & (df.is_weekend == 0)]
    mo = pk.groupby("month").speed_kph.mean()
    fig, ax = plt.subplots(figsize=(16, 7))
    cols = [ACC3 if x >= 9 else ACC2 for x in mo.index]
    ax.bar(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], mo.values, color=cols)
    ax.set_title("The 'ber months' effect: weekday PM-peak speed by month (all corridors)")
    ax.set_ylabel("km/h"); ax.grid(axis="y", alpha=.4)
    save(fig, "05_seasonality.png")

    # 6 corridors
    rd = pk.groupby("road").speed_kph.mean().sort_values()
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.barh(rd.index, rd.values, color=ACC)
    for i, v in enumerate(rd.values):
        ax.text(v + .3, i, f"{v:.1f}", va="center")
    ax.set_title("Weekday 5-8 PM average speed by corridor (km/h)")
    save(fig, "06_corridors.png")

    # 7 forecast next 7 days (clear weather scenario)
    fut = pd.date_range("2026-11-09", periods=24 * 7, freq="h")
    ff = calendar_features(fut)
    ff["rain_mm"], ff["typhoon"], ff["road_id"] = 0.0, 0, ROAD_IDS["EDSA"]
    p = model.predict(ff[FEATS])
    resid = (te.speed_kph - te.pred)[te.road == "EDSA"]
    lo, hi = np.quantile(resid, [0.1, 0.9])
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.fill_between(fut, p + lo, p + hi, color=ACC, alpha=.25, label="80% interval")
    ax.plot(fut, p, color=ACC, lw=2.2, label="Forecast")
    ax.set_title("EDSA 7-day speed forecast (Nov 9-15, 2026, dry-weather scenario)")
    ax.set_ylabel("km/h"); ax.grid(alpha=.4); ax.legend()
    save(fig, "07_forecast.png")
    print("4/4 done")


if __name__ == "__main__":
    main()
