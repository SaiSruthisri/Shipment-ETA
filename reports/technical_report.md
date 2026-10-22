# AIQON Shipment ETA - Technical Report

## 1. What I built

The aim of this project is to predict the final delivery time of a shipment while
the shipment is still moving through its journey.

The system gives three things:

1. the expected final ETA,
2. the expected delay in hours,
3. the likelihood that the delay will be more than six hours.

It also gives a short explanation of why the shipment is expected to be delayed.
The readable examples produced by the model are saved in `results/result.txt`.

I treated ETA as something that changes when new shipment information becomes
available. The model makes predictions at three checkpoints:

- **Departure:** the shipment has left the origin.
- **Port arrival:** the vessel has reached the port.
- **Customs cleared:** customs information is available.

## 2. How I calculate delay and ETA

The final delay is defined as:

`final delay = actual delivery time - scheduled delivery time`

The predicted ETA is:

`predicted ETA = scheduled delivery time + predicted final delay`

For the delay likelihood, I used a six-hour threshold:

- up to six hours late: not delayed,
- more than six hours late: delayed.

The classifier converts this result into Low, Medium, or High likelihood.

## 3. Data and assumptions

I used synthetic data because the assignment allows a constructed dataset and the
required size is 100-300 records. I generated 300 shipments over one year across
five routes:

- Nhava Sheva - Rotterdam
- Nhava Sheva - Los Angeles
- Chennai - Singapore
- Mundra - Jebel Ali
- Chennai - Colombo

The data includes the fields needed for an ETA problem:

- scheduled and actual timestamps,
- route duration and route buffer,
- weather severity,
- port congestion,
- weekends and holidays,
- departure, port, customs, and inland stages,
- missing intermediate timestamps.

I also included rare events such as storms, strikes, and port closures.

The main assumptions are:

- route duration and route variability are different for each route;
- bad weather can delay the ocean journey;
- congestion increases port waiting and customs processing;
- arriving beyond the route buffer can cause a missed berth window;
- weekends and holidays reduce available capacity;
- some delay is absorbed, but some delay continues into the inland leg;
- about 10% of intermediate timestamps may be missing.

These assumptions make the data behave more like a shipment process instead of
being a random table.

## 4. How delay propagation is represented

The generator creates the shipment in stages.

1. A departure delay can make the vessel late on the ocean leg.
2. Weather can add more delay, while a limited recovery is possible during sailing.
3. If port arrival is later than the route buffer, the shipment can miss its berth
   window and receive an additional delay.
4. Port congestion increases waiting time.
5. Customs takes longer when congestion and a weekend or holiday are involved.
6. Part of the accumulated delay is passed into inland delivery.

This means the model is not treating departure, port, customs, and final delivery
as unrelated events. It uses the current delay and the interaction between stages
to estimate the final delay.

When an intermediate timestamp is missing, the model carries forward the latest
known delay and adds a missing-data flag. This lets the model use incomplete
shipments without silently pretending that the information is complete.

## 5. Model approach

I considered a few approaches:

- **Rules only:** useful for simulating delay propagation, but too fixed to be the
  final predictor.
- **Ridge regression:** simple and useful as a baseline, but less suitable for
  non-linear effects such as congestion and missed buffers.
- **Gradient Boosting:** selected because it works well with small tabular data and
  can learn non-linear relationships between route, congestion, weather, and
  stage delays.
- **Neural network or time-series model:** not selected because 300 shipments is
  too small to justify the added complexity.

I selected Gradient Boosting for both tasks:

- a regressor predicts the final delay in hours;
- a classifier predicts whether the final delay will exceed six hours.

The model uses route, weather, congestion, calendar, observed stage delays,
buffer interactions, and missing-data flags.

## 6. Training and evaluation

The records are sorted by scheduled departure and split by time:

- **Training:** 240 shipments, 80% of the data.
- **Testing:** 60 shipments, 20% of the data.

The test records are later shipments than the training records. This is closer to
the real use case than randomly mixing older and newer shipments.

I used:

- **MAE:** average prediction error in hours;
- **RMSE:** gives more weight to large errors;
- **AUC:** measures how well the classifier ranks delayed shipments;
- **F1:** balances precision and recall.

The model was also compared with route-average and persistence baselines.

## 7. Results

| Checkpoint | Gradient Boosting MAE | Classifier AUC | Classifier F1 |
|---|---:|---:|---:|
| Departure | 5.54 hours | 0.686 | 0.512 |
| Port arrival | 4.31 hours | 0.857 | 0.720 |
| Customs cleared | **1.53 hours** | **0.942** | **0.920** |

The model becomes more accurate as more shipment information becomes available.
This is expected because the customs-cleared checkpoint has observed information
from more of the journey.

At the main port-arrival checkpoint:

- MAE was 4.31 hours;
- precision was 0.78;
- recall was 0.67;
- F1 was 0.72.

The Low, Medium, and High groups also followed the expected order. Their actual
delay rates were 18%, 46%, and 84% respectively.

## 8. Example output

This is one of the outputs generated by the trained model and saved in
`results/result.txt`:

```text
Shipment #276 | Chennai-Singapore
- ETA: 11 Sep 2026 09:30
- Expected delay: +29.5 hours
- Delay likelihood: High (92%)
- Current port delay: +33.8 hours
- Why:
  - Late arrival and port congestion are increasing the expected delay.
  - Port arrival is 7.0 hours beyond the route buffer.
  - The shipment reached the port 33.8 hours late.
```

This output is intended for a person reading the result. It gives the ETA first,
then the delay, likelihood, and a few reasons. The reasons connect the prediction
back to the shipment journey instead of only showing technical feature names.

## 9. Future enhancements

- **Customer alerts:** Send the updated ETA, expected delay, likelihood, and reason
  by email, SMS, or a customer portal when delivery is at risk.
- **Operations dashboard:** Show active shipments, risk levels, delayed routes,
  port issues, and shipments that need attention.
- **Customer and warehouse impact:** Identify which customer deliveries, warehouse
  replenishments, and onward transport bookings may be affected.
- **AI recommendations:** Suggest actions such as rescheduling inland transport,
  prioritising customs, contacting the carrier, or notifying the customer.
- **Early warnings:** Detect unusual increases in port congestion, carrier delays,
  customs time, or route delays before they affect more shipments.
- **Feedback loop:** Compare predicted and actual delivery times after completion
  and use the results to improve the model.
- **Real data and API:** Replace or supplement the synthetic data with historical
  shipment data and expose the trained model through an API.
