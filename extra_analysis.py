"""
extra_analysis.py -- two additions requested in committee feedback
  (A) like-for-like cross-validation, to explain the CV (0.565) vs test (0.624) F1-macro gap
  (B) delay-DURATION regression (minutes), promised in Objective 3 / Chapter 3
Run AFTER run_perflight_analysis.py and train_models.py.  Seed = 42.
"""
import pandas as pd, numpy as np, json
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
np.random.seed(42)

X = pd.read_pickle("./X.pkl"); y = pd.read_pickle("./y.pkl")
fc = pd.read_pickle("./flights_clean.pkl")
idx = fc.sort_values(["month","day","hour"]).index
X = X.loc[idx].reset_index(drop=True); y = y.loc[idx].reset_index(drop=True)
arr = fc.loc[idx, "arr_delay"].reset_index(drop=True)          # winsorised minutes
n = len(X); tr = int(n*0.75); va = int(n*0.90)
Xtr, Xte = X.iloc[:tr], X.iloc[va:]
ytr, yte = y.iloc[:tr], y.iloc[va:]
out = {}

# ---------- (A) like-for-like CV ----------
spw = (ytr==0).sum()/(ytr==1).sum()
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
weighted = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, random_state=42, scale_pos_weight=spw, eval_metric="logloss")
cvw = cross_val_score(weighted, Xtr, ytr, cv=cv, scoring="f1_macro", n_jobs=-1)
unweighted = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, eval_metric="logloss")
unweighted.fit(Xtr, ytr)
f1_unw_test = f1_score(yte, unweighted.predict(Xte), average="macro")
out["cv_check"] = {
  "cv_unweighted_200trees_F1macro_mean": 0.5646, "cv_unweighted_note": "configuration used in train_models.py (no class weighting)",
  "cv_weighted_400trees_F1macro_mean": float(cvw.mean()), "cv_weighted_std": float(cvw.std()),
  "test_unweighted_200trees_F1macro": float(f1_unw_test),
}

# ---------- (B) delay-duration regression ----------
ytr_r, yte_r = arr.iloc[:tr], arr.iloc[va:]
def rep(name, pred):
    return {"RMSE": float(np.sqrt(mean_squared_error(yte_r, pred))), "MAE": float(mean_absolute_error(yte_r, pred)), "R2": float(r2_score(yte_r, pred))}
res = {}
res["Mean baseline"] = rep("mean", np.full(len(yte_r), ytr_r.mean()))
ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(Xtr, ytr_r); res["Ridge regression"] = rep("ridge", ridge.predict(Xte))
rf = RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=20, n_jobs=-1, random_state=42).fit(Xtr, ytr_r)
res["Random Forest"] = rep("rf", rf.predict(Xte))
xr = xgb.XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42).fit(Xtr, ytr_r)
res["XGBoost"] = rep("xgb", xr.predict(Xte))
out["regression"] = res
out["regression_target"] = "arr_delay in minutes (winsorised at 99.5th pct, floored at -60); same chronological 75/15/10 split"
print(json.dumps(out, indent=2))
json.dump(out, open("./extra_analysis_results.json","w"), indent=2)
