import pandas as pd
import numpy as np
import json
np.random.seed(42)

# =====================================================================
# STAGE 0: LOAD PER-FLIGHT DATA (pnwflights14: SEA/PDX flights 2014 + hourly weather)
# =====================================================================
flights = pd.read_csv("./real_perflight_dataset.csv")
weather = pd.read_csv("./real_weather_dataset.csv")

raw_n = len(flights)

# =====================================================================
# STAGE 1: PRE-PROCESSING (report missing-value / outlier counts)
# =====================================================================
missing_report = flights.isna().sum().to_dict()

# Drop flights with no recorded arrival delay (cancelled/diverted-equivalent)
flights_clean = flights.dropna(subset=["arr_delay", "dep_delay"]).copy()
n_dropped_missing = raw_n - len(flights_clean)

# Outlier handling: winsorize arr_delay at 99.5th percentile, floor at -60
upper_cap = flights_clean["arr_delay"].quantile(0.995)
n_outliers_capped = int((flights_clean["arr_delay"] > upper_cap).sum() + (flights_clean["arr_delay"] < -60).sum())
flights_clean["arr_delay"] = flights_clean["arr_delay"].clip(lower=-60, upper=upper_cap)

# Target: binary delay classification (arrival delay > 15 minutes)
flights_clean["delayed"] = (flights_clean["arr_delay"] > 15).astype(int)

class_balance = flights_clean["delayed"].value_counts(normalize=True).to_dict()

# =====================================================================
# STAGE 2: FEATURE ENGINEERING (temporal, network, weather)
# =====================================================================
flights_clean = flights_clean.sort_values(["tailnum", "month", "day", "hour", "minute"]).reset_index(drop=True)

# Network-propagation feature: previous flight's arrival delay for same tail number (cascade signal)
flights_clean["prev_arr_delay"] = flights_clean.groupby("tailnum")["arr_delay"].shift(1)
flights_clean["prev_arr_delay"] = flights_clean["prev_arr_delay"].fillna(0)

# Temporal features
flights_clean["is_weekend"] = (pd.to_datetime(dict(year=flights_clean.year, month=flights_clean.month, day=flights_clean.day)).dt.dayofweek >= 5).astype(int)
flights_clean["hour"] = flights_clean["hour"].fillna(flights_clean["hour"].median())

# Merge weather at origin/month/day/hour (left join: flights without a matching weather hour are kept)
weather_small = weather[["origin","month","day","hour","temp","wind_speed","wind_gust","precip","visib"]].copy()
flights_clean = flights_clean.merge(weather_small, on=["origin","month","day","hour"], how="left")
weather_missing_before_fill = flights_clean[["temp","wind_speed","wind_gust","precip","visib"]].isna().sum().to_dict()
for col in ["temp","wind_speed","wind_gust","precip","visib"]:
    flights_clean[col] = flights_clean[col].fillna(flights_clean[col].median())

# Carrier / origin dummies
carrier_dummies = pd.get_dummies(flights_clean["carrier"], prefix="carrier")
origin_dummies = pd.get_dummies(flights_clean["origin"], prefix="origin")

feature_cols_numeric = ["month","day","hour","distance","prev_arr_delay","is_weekend",
                          "temp","wind_speed","wind_gust","precip","visib"]
X = pd.concat([flights_clean[feature_cols_numeric], carrier_dummies, origin_dummies], axis=1)
y = flights_clean["delayed"]

n_final = len(flights_clean)

data_dictionary = {
    "month": "Calendar month of scheduled departure (1-12)",
    "day": "Day of month of scheduled departure",
    "hour": "Scheduled departure hour (0-23, local)",
    "distance": "Great-circle distance, origin to destination (miles)",
    "prev_arr_delay": "Arrival delay (minutes) of this tail number's immediately preceding flight -- network/cascade feature",
    "is_weekend": "Binary flag, 1 if scheduled departure date falls on Saturday/Sunday",
    "temp": "Origin airport temperature at scheduled departure hour (degrees F)",
    "wind_speed": "Origin airport wind speed at scheduled departure hour (mph)",
    "wind_gust": "Origin airport wind gust speed at scheduled departure hour (mph)",
    "precip": "Origin airport precipitation at scheduled departure hour (inches)",
    "visib": "Origin airport visibility at scheduled departure hour (miles)",
    "carrier_*": "One-hot indicator for operating carrier (11 carriers)",
    "origin_*": "One-hot indicator for origin airport (SEA, PDX)",
    "delayed (target)": "Binary target: 1 if arrival delay > 15 minutes, else 0",
}

print(json.dumps({
    "raw_n": raw_n,
    "n_dropped_missing_delay_fields": n_dropped_missing,
    "n_outliers_capped": n_outliers_capped,
    "n_final": n_final,
    "class_balance": class_balance,
    "weather_missing_before_fill": weather_missing_before_fill,
}, indent=2, default=str))

with open("./preprocessing_report.json","w") as f:
    json.dump({
        "raw_n": raw_n,
        "n_dropped_missing_delay_fields": n_dropped_missing,
        "n_outliers_capped": n_outliers_capped,
        "n_final": n_final,
        "class_balance": {str(k):v for k,v in class_balance.items()},
        "weather_missing_before_fill": weather_missing_before_fill,
        "missing_report_raw": {k:int(v) for k,v in missing_report.items()},
    }, f, indent=2, default=str)

with open("./data_dictionary.json","w") as f:
    json.dump(data_dictionary, f, indent=2)

X.to_pickle("./X.pkl")
y.to_pickle("./y.pkl")
flights_clean.to_pickle("./flights_clean.pkl")
print("STAGE 0-2 DONE")
