"""Generate the synthetic shipment dataset."""
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
N = 300
ROOT = Path(__file__).parent
OUT = ROOT / "data"
OUT.mkdir(exist_ok=True)

routes = {
    "Nhava Sheva-Rotterdam":   {"ocean_h": 22*24, "buffer_h": 12, "variability": 1.0, "congestion_bias": 0.10, "inland_h": 36},
    "Nhava Sheva-Los Angeles": {"ocean_h": 30*24, "buffer_h": 18, "variability": 1.3, "congestion_bias": 0.15, "inland_h": 48},
    "Chennai-Singapore":       {"ocean_h": 5*24,  "buffer_h": 6,  "variability": 0.6, "congestion_bias": 0.05, "inland_h": 12},
    "Mundra-Jebel Ali":        {"ocean_h": 4*24,  "buffer_h": 6,  "variability": 0.5, "congestion_bias": 0.00, "inland_h": 18},
    "Chennai-Colombo":         {"ocean_h": 2*24,  "buffer_h": 4,  "variability": 0.4, "congestion_bias": -0.05, "inland_h": 12},
}
PLANNED_PORT_H = 18   # planned time at port: ~6h berth wait + ~12h customs

df = pd.DataFrame({"route": rng.choice(list(routes), size=N)})

start = pd.Timestamp("2025-10-01")
df["sched_departure"] = start + pd.to_timedelta(rng.integers(0, 365*24, size=N), unit="h")

for key in ["ocean_h", "buffer_h", "variability", "congestion_bias", "inland_h"]:
    df[key] = df["route"].map(lambda r: routes[r][key])

df["sched_port_arrival"]   = df["sched_departure"] + pd.to_timedelta(df["ocean_h"], unit="h")
df["sched_customs_clear"]  = df["sched_port_arrival"] + pd.to_timedelta(PLANNED_PORT_H, unit="h")
df["sched_delivery"]       = df["sched_customs_clear"] + pd.to_timedelta(df["inland_h"], unit="h")

holidays = pd.to_datetime([
    "2025-10-02", "2025-10-20", "2025-11-05", "2025-12-25", "2026-01-01",
    "2026-01-26", "2026-02-17", "2026-02-18", "2026-03-04", "2026-04-03",
    "2026-05-01", "2026-08-15", "2026-09-14",
])

def is_dayoff(ts):
    return ((ts.dt.dayofweek >= 5) | ts.dt.normalize().isin(holidays)).astype(int)

df["arrival_dow"]         = df["sched_port_arrival"].dt.dayofweek
df["is_weekend_arrival"]  = (df["arrival_dow"] >= 5).astype(int)
df["is_holiday_arrival"]  = df["sched_port_arrival"].dt.normalize().isin(holidays).astype(int)

month = df["sched_port_arrival"].dt.month

is_monsoon = month.isin([6, 7, 8, 9])
df["weather_severity"] = np.where(is_monsoon, rng.beta(3, 6, N), rng.beta(2, 8, N))

is_peak = month.isin([9, 10, 11])
base_cong = np.where(is_peak, rng.beta(4, 5, N), rng.beta(2, 6, N))
df["port_congestion"] = np.clip(base_cong + df["congestion_bias"], 0, 1)

ANOMALY_RATE = 0.06
is_anom = rng.random(N) < ANOMALY_RATE
anom_type = np.where(is_anom, rng.choice(["storm", "strike", "port_closure"], N), "none")
anom_extra = np.where(is_anom, rng.uniform(20, 60, N), 0.0)   # extra hours

storm_extra   = np.where(anom_type == "storm", anom_extra, 0)          # hits ocean leg
strike_extra  = np.where(anom_type == "strike", anom_extra, 0)         # hits port wait
closure_extra = np.where(anom_type == "port_closure", anom_extra, 0)   # hits customs

df["anomaly_type"] = anom_type
df["anomaly_extra_h"] = anom_extra.round(1)

cong = df["port_congestion"].to_numpy()
var  = df["variability"].to_numpy()

dep_delay = np.clip(rng.lognormal(np.log(2), 0.9, N) * var, 0, 48)

weather_delay = df["weather_severity"].to_numpy() * 30 * var * rng.uniform(0.3, 1.2, N) + storm_extra
late_so_far = dep_delay + weather_delay
recovery_cap = 0.05 * df["ocean_h"].to_numpy() * rng.uniform(0.2, 1.0, N)
ocean_recovery = np.minimum(late_so_far, recovery_cap)
arrival_delay = np.clip(late_so_far - ocean_recovery + rng.normal(0, 1.5, N), -6, None)

missed_window = (arrival_delay > df["buffer_h"].to_numpy()).astype(int)
berth_jump = missed_window * rng.uniform(8, 24, N)
wait_h = 4 + 18 * cong**2 * rng.uniform(0.5, 1.5, N) + berth_jump + strike_extra
wait_extra = wait_h - 6      # vs planned 6h (negative = faster than plan)

actual_port_arrival = df["sched_port_arrival"] + pd.to_timedelta(arrival_delay, unit="h")
customs_start = actual_port_arrival + pd.to_timedelta(wait_h, unit="h")

dayoff_customs = is_dayoff(customs_start).to_numpy()
customs_h = (10 * (1 + 0.8 * cong**2)               # congestion x processing interaction
             * (1 + 0.3 * dayoff_customs * (1 + cong))  # weekend/holiday x congestion interaction
             * rng.lognormal(0, 0.2, N)             # random noise (positive, skewed)
             + closure_extra)
customs_extra = customs_h - 12

actual_customs_clear = customs_start + pd.to_timedelta(customs_h, unit="h")
customs_clear_delay = (actual_customs_clear - df["sched_customs_clear"]).dt.total_seconds().to_numpy() / 3600

passthrough = 0.4 + 0.4 * cong
dayoff_delivery = is_dayoff(actual_customs_clear).to_numpy()
delivery_new = rng.lognormal(np.log(1), 0.7, N) - 1 + 4 * dayoff_delivery   # trucking/rail thinner on days off
final_delay = passthrough * customs_clear_delay + delivery_new

actual_delivery = df["sched_delivery"] + pd.to_timedelta(final_delay, unit="h")

truth = df.copy()
truth["dep_delay_h"]           = dep_delay.round(2)
truth["weather_delay_h"]       = weather_delay.round(2)
truth["ocean_recovery_h"]      = ocean_recovery.round(2)
truth["port_arrival_delay_h"]  = arrival_delay.round(2)
truth["missed_berth_window"]   = missed_window
truth["port_wait_extra_h"]     = wait_extra.round(2)
truth["customs_extra_h"]       = customs_extra.round(2)
truth["dayoff_customs"]        = dayoff_customs
truth["customs_clear_delay_h"] = customs_clear_delay.round(2)
truth["passthrough"]           = passthrough.round(3)
truth["dayoff_delivery"]       = dayoff_delivery
truth["delivery_new_h"]        = delivery_new.round(2)
truth["actual_departure"]      = df["sched_departure"] + pd.to_timedelta(dep_delay, unit="h")
truth["actual_port_arrival"]   = actual_port_arrival
truth["actual_customs_clear"]  = actual_customs_clear
truth["actual_delivery"]       = actual_delivery

truth["final_delay_h"]  = final_delay.round(2)
truth["is_delayed_6h"]  = (final_delay > 6).astype(int)

model = truth.copy()
for col in ["actual_departure", "actual_port_arrival", "actual_customs_clear"]:
    mask = rng.random(N) < 0.10
    model[col + "_was_missing"] = mask.astype(int)
    model.loc[mask, col] = pd.NaT

hidden = ["dep_delay_h", "weather_delay_h", "ocean_recovery_h", "port_arrival_delay_h",
          "missed_berth_window", "port_wait_extra_h", "customs_extra_h", "dayoff_customs",
          "customs_clear_delay_h", "passthrough", "dayoff_delivery", "delivery_new_h",
          "anomaly_type", "anomaly_extra_h"]
model = model.drop(columns=hidden)

model.to_csv(OUT / "shipments.model.csv", index=False)

print(f"Rows: {len(truth)}")
print("\nFinal delay (hours):")
print(truth["final_delay_h"].describe().round(2))
print(f"\nShare delayed > 6h: {truth['is_delayed_6h'].mean():.2%}")
print(f"Anomaly share:      {is_anom.mean():.2%}")
print(f"Missed berth window:{missed_window.mean():.2%}")
print(f"Weekend arrivals:   {truth['is_weekend_arrival'].mean():.2%}  (expect ~28%)")
print("\nMean congestion: peak vs non-peak")
print(truth.groupby(is_peak.values)["port_congestion"].mean().round(3))
print("\nMean final delay by route:")
print(truth.groupby("route")["final_delay_h"].mean().round(2))
print("\nMean final delay: anomaly vs normal")
print(truth.groupby(truth["anomaly_type"] != "none")["final_delay_h"].mean().round(2))
print("\nMissing share per timestamp column:")
print(model[["actual_departure", "actual_port_arrival", "actual_customs_clear"]].isna().mean().round(3))