# AIQON Shipment ETA - Technical Report

## Objective

The system estimates:

- final shipment ETA,
- final delay in hours,
- likelihood of a delay above six hours,
- factors behind the prediction.

ETA is updated at three points in the journey:

- **A - Departure:** departure information is available.
- **B - Port arrival:** port-arrival information is available.
- **C - Customs cleared:** customs information is available.

The delay target is:

`final delay = actual delivery time - scheduled delivery time`

The final ETA is:

`predicted ETA = scheduled delivery time + predicted delay`

I used six hours as the delayed/not-delayed threshold.

## Data

I used synthetic data because the assignment allows it and because it lets me include
the cases needed for this problem. The dataset contains 300 shipments over one year and
five routes:

- Nhava Sheva - Rotterdam
- Nhava Sheva - Los Angeles
- Chennai - Singapore
- Mundra - Jebel Ali
- Chennai - Colombo

The generator includes:

- scheduled and actual stage timestamps,
- route duration and route buffer,
- route variability,
- port congestion,
- weather severity,
- weekends and holidays,
- seasonal effects,
- missing intermediate timestamps,
- missed berth windows,
- storm, strike, and port closure events.

The journey is generated stage by stage. A late departure can affect port arrival.
Arrival beyond the buffer can add a berth delay. Congestion affects port waiting and
customs processing. Part of the delay is carried into inland delivery.

The model input is `data/shipments.model.csv`. About 10% of intermediate timestamps are
missing. The model keeps a missing flag and uses the previous known delay when it needs
to fill a missing stage.

## Model options

### Rule-based model

**Used for:** generating the shipment journey.

Rules are useful for creating delay propagation and anomalies. I did not use rules as
the final predictor because fixed rules do not handle noisy operational data as well as
a trained model.

### Ridge regression

**Used for:** baseline comparison.

Ridge is simple and works well for mostly linear relationships. It does not capture
the non-linear effect of congestion, buffer misses, and stage interactions as well as
the selected model.

### Gradient Boosting

**Used for:** final regression and classification models.

Gradient Boosting fits this small tabular dataset and can learn non-linear interactions.
I used it to predict delay hours and the probability of a delay above six hours.

### Neural network or recurrent model

**Not used.**

There are only 300 independent journeys. A neural model would add complexity and would
be more likely to overfit without more historical event data.

### Survival model

**Not used.**

Survival models are useful when final delivery is unknown for many active shipments.
Here, every generated shipment has a final outcome, so regression and classification
are easier to validate.

## Features

The model uses:

- route duration, buffer, variability, and inland duration,
- port congestion and weather severity,
- month, arrival day, weekend, holiday, and peak season,
- observed departure, port-arrival, and customs delays,
- lateness compared with the route buffer,
- congestion squared,
- congestion combined with days off,
- lateness combined with congestion,
- missing timestamp flags.

The feature set is different at each checkpoint. Future stage information is not used
to predict an earlier checkpoint.

## Training and validation

The records are sorted by scheduled departure and split by time:

- **Training:** 80% of data i.e, 240 shipments, 2 October 2025 to 31 July 2026.
- **Testing:** 20% of data i.e, 60 shipments, 1 August 2026 to 30 September 2026.

I used a time split instead of only a random split because it is closer to the real
case: use older shipments to predict later shipments.

Metrics:

- MAE for average delay error,
- RMSE for larger errors,
- AUC for delay ranking,
- Brier score for probability quality,
- F1 for classification balance.

I also compare against route-mean and persistence baselines.

## Results

| Checkpoint | Gradient Boosting MAE | Ridge MAE | Classifier AUC | Classifier F1 |
|---|---:|---:|---:|---:|
| A - Departure | 5.54 hours | 4.56 hours | 0.686 | 0.512 |
| B - Port arrival | 4.31 hours | 3.66 hours | 0.857 | 0.720 |
| C - Customs cleared | **1.53 hours** | 1.79 hours | **0.942** | **0.920** |

Checkpoint B is used for the detailed output:

- MAE: 4.31 hours
- RMSE: 6.23 hours
- precision: 0.78
- recall: 0.67
- F1: 0.72
The B checkpoint likelihood groups had these actual delay rates:

- Low: 18%
- Medium: 46%
- High: 84%

Checkpoint C performs better because it has more observed information from the journey.

## Explanations

The output includes the scheduled delivery, predicted ETA, delay hours, likelihood,
current port delay, and the top factors.

The largest average SHAP groups were:

1. lateness,
2. congestion,
3. calendar,
4. weather,
5. route.

The lateness SHAP group had a 0.73 Spearman correlation with observed port-arrival
delay. The congestion SHAP group had a 0.51 correlation with port congestion.

This lets the output say more than just the ETA. It can show that the main cause is
late arrival beyond the buffer, congestion, or a calendar effect.

## Deployment

Deployment is not part of this version. The trained models are saved in
`results/eta_models.joblib`.

A production version would add an API for shipment updates, the same feature-building
logic, an ETA response with explanations, a dashboard, scheduled retraining, and
monitoring for error, calibration, missing data, route drift, and anomalies.

## Limitations

The data is synthetic and the sample is small. It cannot cover every carrier, port,
customs, or weather pattern. There is no live event feed, API, or dashboard yet.
