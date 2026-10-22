"""Train and evaluate the shipment ETA models."""
import warnings
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (brier_score_loss, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
SEED = 0
ROOT = Path(__file__).parent
DATA = ROOT / "data"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

DELAY_THRESHOLD_H = 6          # "delayed" = more than 6h late
LOW_MAX, MED_MAX = 0.30, 0.60  # P<0.3 Low | 0.3-0.6 Medium | >0.6 High
CHECKPOINTS = ["A", "B", "C"]
MAIN = "B"                     # checkpoint used for the detailed analysis

TS = ["sched_departure", "sched_port_arrival", "sched_customs_clear", "sched_delivery",
      "actual_departure", "actual_port_arrival", "actual_customs_clear", "actual_delivery"]
model_df = pd.read_csv(DATA / "shipments.model.csv", parse_dates=TS)


def hrs(a, b):
    return (a - b).dt.total_seconds() / 3600


df = model_df.sort_values("sched_departure").copy()
df["dep_obs"] = hrs(df.actual_departure, df.sched_departure)
df["arr_obs"] = hrs(df.actual_port_arrival, df.sched_port_arrival)
df["cc_obs"] = hrs(df.actual_customs_clear, df.sched_customs_clear)

cut = int(len(df) * 0.8)
train, test = df.iloc[:cut], df.iloc[cut:]
print(f"Train: {len(train)} rows ({train.sched_departure.min().date()} -> {train.sched_departure.max().date()})")
print(f"Test : {len(test)} rows ({test.sched_departure.min().date()} -> {test.sched_departure.max().date()})")

dep_median_by_route = train.groupby("route")["dep_obs"].median()
dep_median_global = train["dep_obs"].median()

def build_features(d, cp):
    """Build features using only information available at checkpoint cp."""
    X = pd.DataFrame(index=d.index)
    dayoff = ((d.is_weekend_arrival == 1) | (d.is_holiday_arrival == 1)).astype(int)
    month = d.sched_port_arrival.dt.month

    X["ocean_h"], X["buffer_h"] = d.ocean_h, d.buffer_h
    X["route_variability"], X["inland_h"] = d.variability, d.inland_h
    X["port_congestion"], X["weather_severity"] = d.port_congestion, d.weather_severity
    X["is_weekend_arrival"], X["is_holiday_arrival"] = d.is_weekend_arrival, d.is_holiday_arrival
    X["arrival_dow"], X["month"] = d.arrival_dow, month
    X["is_peak_season"] = month.isin([9, 10, 11]).astype(int)
    X["cong_sq"] = d.port_congestion ** 2
    X["cong_x_dayoff"] = d.port_congestion * dayoff

    dep = d.dep_obs.fillna(d.route.map(dep_median_by_route)).fillna(dep_median_global)
    X["dep_delay"] = dep
    X["dep_missing"] = d.actual_departure_was_missing
    latest = dep
    if cp in ("B", "C"):
        arr = d.arr_obs.fillna(dep)               # missing -> carry previous delay forward
        X["arrival_delay"] = arr
        X["arr_missing"] = d.actual_port_arrival_was_missing
        X["arrival_beyond_buffer"] = (arr > d.buffer_h).astype(int)   # missed berth window?
        latest = arr
    if cp == "C":
        cc = d.cc_obs.fillna(arr)
        X["customs_delay"] = cc
        X["cc_missing"] = d.actual_customs_clear_was_missing
        latest = cc
    X["late_vs_buffer"] = latest - d.buffer_h
    X["late_x_cong"] = latest * d.port_congestion
    return X, latest


FEATURE_FAMILY = {   # grouping used to explain at a human level
    "lateness": ["dep_delay", "arrival_delay", "customs_delay", "late_vs_buffer",
                 "late_x_cong", "arrival_beyond_buffer"],
    "congestion": ["port_congestion", "cong_sq", "cong_x_dayoff"],
    "weather": ["weather_severity"],
    "calendar": ["is_weekend_arrival", "is_holiday_arrival", "arrival_dow", "month", "is_peak_season"],
    "route": ["ocean_h", "buffer_h", "route_variability", "inland_h"],
    "missing_data": ["dep_missing", "arr_missing", "cc_missing"],
}
NICE = {
    "dep_delay": "departure delay", "arrival_delay": "port-arrival delay",
    "customs_delay": "customs delay", "late_vs_buffer": "lateness vs buffer",
    "late_x_cong": "lateness x congestion", "arrival_beyond_buffer": "missed berth window",
    "port_congestion": "port congestion", "cong_sq": "congestion (non-linear)",
    "cong_x_dayoff": "congestion x weekend/holiday", "weather_severity": "weather severity",
    "is_weekend_arrival": "weekend arrival", "is_holiday_arrival": "holiday arrival",
    "arrival_dow": "day of week", "month": "month", "is_peak_season": "peak season",
    "ocean_h": "route length", "buffer_h": "route buffer", "route_variability": "route volatility",
    "inland_h": "inland leg", "dep_missing": "departure time missing",
    "arr_missing": "arrival time missing", "cc_missing": "customs time missing",
}


def explain_factor(feature, value, row):
    amount = abs(value)
    direction = "increasing the expected delay" if value > 0 else "offsetting part of the expected delay"
    dep_delay = abs(row.dep_obs) if pd.notna(row.dep_obs) else None
    arrival_delay = abs(row.arr_obs) if pd.notna(row.arr_obs) else None
    customs_delay = abs(row.cc_obs) if pd.notna(row.cc_obs) else None
    messages = {
        "dep_delay": (
            f"The shipment left the origin {dep_delay:.1f} hours late."
            if dep_delay is not None else
            "The actual departure time is missing, so the departure delay was estimated."
        ),
        "arrival_delay": (
            f"The shipment reached the port {arrival_delay:.1f} hours late."
            if arrival_delay is not None else
            "The actual port-arrival time is missing, so the arrival delay was estimated."
        ),
        "customs_delay": (
            f"Customs clearance is {customs_delay:.1f} hours from schedule."
            if customs_delay is not None else
            "The actual customs time is missing, so the customs delay was estimated."
        ),
        "late_vs_buffer": f"Port arrival is {amount:.1f} hours beyond the route buffer.",
        "late_x_cong": f"Late arrival and port congestion are {direction}.",
        "arrival_beyond_buffer": "The shipment missed its planned port arrival window.",
        "port_congestion": f"Port congestion is {direction}.",
        "cong_sq": f"High port congestion is {direction}.",
        "cong_x_dayoff": f"Port congestion combined with a weekend or holiday is {direction}.",
        "weather_severity": f"Weather conditions are {direction}.",
        "is_weekend_arrival": "The planned port arrival falls on a weekend.",
        "is_holiday_arrival": "The planned port arrival falls on a holiday.",
        "arrival_dow": "The planned day of arrival is affecting the estimate.",
        "month": "The shipment is moving through a higher-risk seasonal period.",
        "is_peak_season": "The shipment is moving through the peak season.",
        "ocean_h": "The route has a longer ocean journey.",
        "buffer_h": "The route has a limited buffer for absorbing delays.",
        "route_variability": "This route has higher-than-average travel variability.",
        "inland_h": "The inland delivery leg is affecting the final ETA.",
        "dep_missing": "The actual departure time is missing, so the departure delay was estimated.",
        "arr_missing": "The actual port-arrival time is missing, so the current delay was estimated.",
        "cc_missing": "The actual customs time is missing, so the customs delay was estimated.",
    }
    return messages.get(feature, f"{NICE.get(feature, feature)} is affecting the estimate.")

def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def make_gbr(**kw):   # shallow trees + subsampling -> less overfitting on ~240 rows
    return GradientBoostingRegressor(n_estimators=300, learning_rate=0.03, max_depth=3,
                                     min_samples_leaf=5, subsample=0.8, random_state=SEED, **kw)


def make_gbc():
    return GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=2,
                                      min_samples_leaf=5, subsample=0.8, random_state=SEED)


def bucket(p):
    return "Low" if p < LOW_MAX else ("Medium" if p < MED_MAX else "High")


ytr, yte = train["final_delay_h"], test["final_delay_h"]
ctr, cte = train["is_delayed_6h"], test["is_delayed_6h"]
base_by_route = train.groupby("route")["final_delay_h"].mean()

results, models = [], {}
for cp in CHECKPOINTS:
    Xtr, ltr = build_features(train, cp)
    Xte, lte = build_features(test, cp)

    preds = {
        "Baseline: route mean": test.route.map(base_by_route).to_numpy(),
        "Baseline: persistence (current delay stays)": lte.to_numpy(),
    }
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=5.0)).fit(Xtr, ytr)
    preds["Ridge regression"] = ridge.predict(Xte)
    gbr = make_gbr().fit(Xtr, ytr)
    preds["Gradient boosting (main)"] = gbr.predict(Xte)

    cv_mae = -cross_val_score(make_gbr(), Xtr, ytr, cv=KFold(5, shuffle=True, random_state=SEED),
                              scoring="neg_mean_absolute_error").mean()
    for name, p in preds.items():
        results.append({"checkpoint": cp, "model": name,
                        "MAE_h": mean_absolute_error(yte, p), "RMSE_h": rmse(yte, p),
                        "CV_MAE_h": cv_mae if name.startswith("Gradient") else np.nan})

    clf = make_gbc().fit(Xtr, ctr)
    prob = clf.predict_proba(Xte)[:, 1]
    results.append({"checkpoint": cp, "model": "Classifier (delay>6h)",
                    "AUC": roc_auc_score(cte, prob), "Brier": brier_score_loss(cte, prob),
                    "F1": f1_score(cte, prob > 0.5)})
    models[cp] = dict(Xtr=Xtr, Xte=Xte, gbr=gbr, clf=clf, prob=prob, pred=preds["Gradient boosting (main)"],
                      latest_te=lte)

res = pd.DataFrame(results)
pd.set_option("display.width", 200)
print("\n=== REGRESSION: MAE (hours) on the 20% test set, by checkpoint ===")
reg = res[res.MAE_h.notna()].pivot(index="model", columns="checkpoint", values="MAE_h").round(2)
print(reg)
print("\nCV MAE of main model on training data (5-fold):")
print(res[res.CV_MAE_h.notna()][["checkpoint", "CV_MAE_h"]].round(2).to_string(index=False))
print("\n=== CLASSIFICATION (delay > 6h) by checkpoint ===")
print(res[res.AUC.notna()][["checkpoint", "AUC", "Brier", "F1"]].round(3).to_string(index=False))

M = models[MAIN]
pred, prob = M["pred"], M["prob"]
print(f"\n=== DETAILED VALIDATION @ checkpoint {MAIN} ===")

Xtr_m, Xte_m = M["Xtr"], M["Xte"]
yhat = (prob > 0.5).astype(int)
print(f"Precision {precision_score(cte, yhat):.2f} | Recall {recall_score(cte, yhat):.2f} | F1 {f1_score(cte, yhat):.2f}")
print("Confusion matrix [[TN FP],[FN TP]]:\n", confusion_matrix(cte, yhat))
buckets = pd.Series([bucket(p) for p in prob], index=test.index)
rel = pd.DataFrame({"bucket": buckets, "actually_delayed": cte}).groupby("bucket").agg(
    n=("actually_delayed", "size"), observed_delay_rate=("actually_delayed", "mean")).reindex(["Low", "Medium", "High"])
print("\nDo likelihood buckets mean what they say? (observed delay rate per bucket)")
print(rel.round(2))

slice_df = pd.DataFrame({"abs_err": np.abs(yte - pred), "route": test.route})
print("\nMAE by route:\n", slice_df.groupby("route").abs_err.mean().round(2))

explainer = shap.TreeExplainer(M["gbr"])
sv = pd.DataFrame(explainer.shap_values(Xte_m), index=Xte_m.index, columns=Xte_m.columns)
base_value = float(np.ravel(explainer.expected_value)[0])
assert np.allclose(sv.sum(axis=1) + base_value, pred, atol=1e-3), "SHAP must add up to prediction"

fam = pd.DataFrame({f: sv[[c for c in cols if c in sv.columns]].sum(axis=1)
                    for f, cols in FEATURE_FAMILY.items()})
print("\n=== GLOBAL DRIVERS: mean |SHAP| by factor family (hours) ===")
print(fam.abs().mean().sort_values(ascending=False).round(2))

print("\n=== EXPLANATION VALIDATION vs observable features (Spearman) ===")
checks = [
    ("congestion SHAP vs port_congestion value", sv["port_congestion"] + sv["cong_sq"] + sv["cong_x_dayoff"], Xte_m["port_congestion"]),
    ("lateness SHAP vs observed port-arrival delay", fam["lateness"], Xte_m["arrival_delay"]),
]
for name, a, b in checks:
    print(f"  {name:65s} rho = {spearmanr(a, b)[0]:+.2f}")
dayoff_mask = ((Xte_m.is_weekend_arrival == 1) | (Xte_m.is_holiday_arrival == 1))
print(f"  mean calendar SHAP on weekend/holiday arrivals: {fam.loc[dayoff_mask, 'calendar'].mean():+.2f}h "
      f"vs weekdays: {fam.loc[~dayoff_mask, 'calendar'].mean():+.2f}h")

def explain_shipment(idx, top_k=3):
    row = test.loc[idx]
    eta = row.sched_delivery + pd.to_timedelta(pred[test.index.get_loc(idx)], unit="h")
    i = test.index.get_loc(idx)
    p, d = prob[i], pred[i]
    s = sv.loc[idx].sort_values(key=np.abs, ascending=False).head(top_k)
    drivers = [explain_factor(f, v, row) for f, v in s.items()]
    x = Xte_m.loc[idx]
    dep, arr = x["dep_delay"], x["arrival_delay"]
    return (f"Shipment #{idx} | {row.route}\n"
            f"- ETA: {eta:%d %b %Y %H:%M}\n"
            f"- Expected delay: {d:+.1f} hours\n"
            f"- Delay likelihood: {bucket(p)} ({p:.0%})\n"
            f"- Current port delay: {arr:+.1f} hours\n"
            f"- Why:\n"
            + "".join(f"  - {factor}\n" for factor in drivers))


order = np.argsort(pred)
picks = [test.index[order[-1]], test.index[order[len(order) // 2]], test.index[order[0]]]
print("\n=== SAMPLE OUTPUTS (checkpoint B) ===")
samples = "\n\n".join(explain_shipment(i) for i in picks)
print(samples)
(RES / "result.txt").write_text(samples)

joblib.dump({"checkpoints": {cp: {"features": list(models[cp]["Xtr"].columns), "regressor": models[cp]["gbr"],
                                  "classifier": models[cp]["clf"]} for cp in CHECKPOINTS},
             "dep_median_by_route": dep_median_by_route, "dep_median_global": dep_median_global,
             "thresholds": {"low_max": LOW_MAX, "med_max": MED_MAX, "delay_h": DELAY_THRESHOLD_H}},
            RES / "eta_models.joblib")

out = test[["route", "sched_delivery"]].copy()
out["pred_delay_h"] = pred
out["prob_delay_gt6h"], out["likelihood"] = prob, buckets
out["pred_eta"] = out.sched_delivery + pd.to_timedelta(out.pred_delay_h, unit="h")
out["actual_delay_h"] = yte
out.round(2).to_csv(RES / "test_predictions.csv")
res.round(3).to_csv(RES / "metrics.csv", index=False)

fig, ax = plt.subplots(figsize=(7, 4))
reg.T.plot(kind="bar", ax=ax)
ax.set_ylabel("MAE (hours)")
ax.set_title("Test MAE by checkpoint: error falls as information arrives")
ax.legend(fontsize=7)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(RES / "mae_by_checkpoint.png", dpi=130)
plt.close()

fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(yte, pred, alpha=0.7)
mx = max(yte.max(), pred.max()) + 2
ax.plot([-5, mx], [-5, mx], "k--", lw=1)
ax.set_xlabel("Actual delay (h)")
ax.set_ylabel(f"Predicted delay (h), checkpoint {MAIN}")
ax.set_title("Predicted vs actual (test set)")
plt.tight_layout()
plt.savefig(RES / "pred_vs_actual.png", dpi=130)
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
fam.abs().mean().sort_values().plot(kind="barh", ax=axes[0])
axes[0].set_title("Mean |SHAP| by factor family (h)")
top = sv.abs().mean().sort_values().tail(10)
top.rename(index=NICE).plot(kind="barh", ax=axes[1])
axes[1].set_title("Top 10 individual features (h)")
plt.tight_layout()
plt.savefig(RES / "shap_importance.png", dpi=130)
plt.close()
print(f"\nSaved model, predictions, metrics and plots to {RES}")