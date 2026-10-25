# Analysis Package — Predictive Analytics for Flight Delay Reduction

Code, data and outputs behind Chapter 4 of the thesis by Jawaher Alali
(MSc Professional Studies: Data Analytics, RIT Dubai, Cohort 15).

## Data

| File | Contents | Rows |
|---|---|---|
| `real_perflight_dataset.csv` | One row per flight departing Seattle (SEA) or Portland (PDX) in 2014: date, scheduled/actual times, departure and arrival delays, carrier, tail number, flight, origin, destination, air time, distance | 162,049 |
| `real_weather_dataset.csv` | Hourly weather at SEA and PDX in 2014: temperature, dew point, humidity, wind direction, wind speed, wind gust, precipitation, pressure, visibility | 17,426 |

Source: the pnwflights14 dataset (Pacific Northwest flights and weather, 2014), as distributed in
Zhang, Z. (2021). *Flight delay prediction* [Data set]. Kaggle. https://www.kaggle.com/competitions/flight-delay-prediction

The two files are joined on origin airport, month, day and scheduled departure hour (left join).

## Setup (one time)

```
pip install -r requirements.txt
```

## Run order

All scripts use relative paths. From inside this folder, run:

```
python3 run_perflight_analysis.py   # 1. clean data, join weather, engineer the 24 features
python3 train_models.py             # 2. logistic regression, Random Forest, XGBoost; bootstrap CIs; 5-fold CV
python3 shap_analysis.py            # 3. TreeSHAP for XGBoost and Random Forest (Figures 4.5-4.7)
python3 lstm_numpy.py               # 4. NumPy LSTM on five-flight tail-number sequences + permutation importance
python3 final_chart.py              # 5. four-model comparison chart (Figure 4.1)
python3 lr_and_roc_viz.py           # 6. LR coefficients, ROC curves, LR probability plot (Figures 4.2-4.4)
python3 extra_analysis.py           # 7. matched-configuration CV check and delay-duration regression (Table 4.5)
```

Every script sets `seed = 42`; the `.json`, `.csv` and `figures/*.png` files included here are the
outputs of the run reported in the thesis.

## Dashboard back-end (optional)

Run after stage 2 (it reads `models.pkl` and `X_train.pkl` from this folder):

```
cd dashboard
uvicorn api:app --host 127.0.0.1 --port 8123
```

Example request from a second terminal:

```
curl -X POST http://127.0.0.1:8123/predict -H "Content-Type: application/json" -d '{
  "month": 12, "day": 20, "hour": 17, "distance": 954, "prev_arr_delay": 45,
  "is_weekend": 0, "temp": 38, "wind_speed": 18, "wind_gust": 25, "precip": 0.1, "visib": 6,
  "carrier": "AS", "origin": "SEA"
}'
```

The response contains the delay probability, a risk tier (Green < 0.3, Amber 0.3–0.6, Red ≥ 0.6)
and the five features with the largest SHAP contributions.

## Files

| File | Produces / contains |
|---|---|
| `preprocessing_report.json` | Missing-value counts, rows removed, values capped, class balance (Section 4.3, Table 4.3) |
| `data_dictionary.json` | Feature definitions (Table 4.2) |
| `model_results.json` | Test-set metrics and F1-macro bootstrap CIs (Table 4.4) |
| `extra_analysis_results.json` | Matched-configuration CV (Section 4.4) and duration regression (Table 4.5) |
| `shap_importance_xgb.csv`, `shap_importance_rf.csv` | Mean absolute SHAP values (Section 4.6) |
| `lstm_results.json`, `lstm_permutation_importance.json` | LSTM metrics and permutation importance (Table 4.6) |
| `lr_coefficients.csv` | Standardized logistic-regression coefficients (Figure 4.3) |
| `dashboard/eval_results.json` | Scenario tests and 200-request load test (Table 4.7) |
| `figures/` | Figures 4.1–4.7 |

## Notes

- The LSTM is written directly in NumPy (single layer, 12 hidden units, 10 epochs); its 18,000
  sampled sequences are split 80/20 at random rather than chronologically.
- The winsorisation cut-off (99.5th percentile, 169 minutes) and the weather medians are computed on
  the full cleaned dataset before the train/test split (under 1% of records affected).
- Logistic regression and XGBoost reproduce exactly across the library versions tested; Random Forest
  can shift slightly with the scikit-learn version (e.g. F1-macro 0.621 vs 0.629), within the
  reported confidence interval.
- To apply the pipeline to another airline, replace the two CSV files with data in the same columns
  and re-run the stages above.
