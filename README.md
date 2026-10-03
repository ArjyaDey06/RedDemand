# RedDemand

RedDemand is a predictive modeling project developed for a redBus hackathon hosted by Analytics Vidhya. The goal is to predict the number of passengers for a specific journey 15 days before the actual travel date and generate the results in a CSV file.

The project uses a stacked ensemble of LightGBM and XGBoost models, combined with feature engineering built around route, timing, and demand patterns.

## Author

Arjya Dey

## Problem Statement

Forecast the `final_seatcount` for each journey based on historical travel and demand signals such as:

- source and destination IDs
- date and day-of-week information
- route-level demand trends
- search and seat-count patterns
- holiday-related features
- rolling historical averages

## Project Highlights

- Uses a hybrid ensemble model with LightGBM and XGBoost
- Adds route-level and time-based feature engineering
- Builds lag-based demand features from transaction data
- Includes holiday importance as part of the model input
- Produces a submission-ready CSV file for prediction output

## Repository Contents

- `xgb_lgbm_stacking_final.py` — main training and prediction pipeline
- `xgb_lgbm_stacking_final.csv` — generated output predictions
- `README.md` — project overview and usage guide

## Tech Stack

- Python
- pandas
- numpy
- LightGBM
- XGBoost
- scikit-learn

## Dependencies

Install the required libraries using:

```bash
pip install pandas numpy lightgbm xgboost scikit-learn
```

## Data Requirements

The script expects the following CSV files in the project directory:

- `train.csv`
- `test.csv`
- `transactions.csv`
- `holidays.csv` (optional, but recommended)

## How It Works

1. Load journey, transaction, and holiday datasets.
2. Convert date columns to datetime format.
3. Engineer lag features and route-level aggregation metrics.
4. Create time-based features like weekday, weekend flag, month, and week-of-year.
5. Train LightGBM and XGBoost models on the training set.
6. Combine model predictions using a Ridge meta-model.
7. Generate final predictions for the test set.
8. Save output as `xgb_lgbm_stacking_final.csv`.

## Run the Model

```bash
python xgb_lgbm_stacking_final.py
```

This script will train the model and generate the output CSV file in the project directory.

## Output Format

The final output file contains:

- `route_key`
- `final_seatcount`

## Notes

This repository is a hackathon-style solution focused on forecasting demand and building a practical prediction pipeline. It is intended for experimentation, learning, and competitive modeling rather than production deployment.
