# AIQON Shipment ETA

This project predicts shipment ETA, final delay, and the chance of a delay above six hours.
It also shows which shipment factors affected the prediction.

The data is synthetic and contains 300 shipments across five routes. It includes route
duration, port congestion, weather, weekends, holidays, missing stage timestamps, and
rare events such as storms, strikes, and port closures.

## Project structure

```text
Shipment eta/
├── generate_data.py
├── train.py
├── requirements.txt
├── data/
│   └── shipments.model.csv
├── results/
│   ├── eta_models.joblib
│   ├── metrics.csv
│   ├── test_predictions.csv
│   ├── result.txt
│   ├── mae_by_checkpoint.png
│   ├── pred_vs_actual.png
│   └── shap_importance.png
└── reports/
    └── technical_report.md
```

`generate_data.py` creates the dataset. `train.py` trains and evaluates the models.
The `data` folder contains the model input. The `results` folder contains predictions,
metrics, the saved models, and plots. The report explains the approach and assumptions.

## Setup

Use Python 3.11 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run

From the project folder:

```powershell
python .\generate_data.py
python .\train.py
```

The first command creates `data/shipments.model.csv`. The second command uses a
time-based 80/20 split:

- 240 earlier shipments for training
- 60 later shipments for testing

The training script evaluates three checkpoints:

- A: departure
- B: port arrival
- C: customs cleared

## Results from the current run

| Checkpoint | Gradient Boosting MAE | Classifier AUC | Classifier F1 |
|---|---:|---:|---:|
| A | 5.54 hours | 0.686 | 0.512 |
| B | 4.31 hours | 0.857 | 0.720 |
| C | 1.53 hours | 0.942 | 0.920 |

The model predicts final delay in hours with Gradient Boosting Regression.
A Gradient Boosting Classifier predicts whether the delay will be more than six hours.
The result file shows the ETA, delay hours, likelihood, and main factors in plain
language. Ridge regression and simple baselines are included for comparison.
SHAP is used for the factor explanations.

## Output files

- `results/metrics.csv` - model metrics for every checkpoint.
- `results/test_predictions.csv` - test predictions and actual delays.
- `results/eta_models.joblib` - saved trained models.
- `results/result.txt` - readable prediction results from the latest run.
- `results/*.png` - evaluation and explanation plots.
