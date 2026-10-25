import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (f1_score, roc_auc_score, precision_score, recall_score,
                               brier_score_loss, accuracy_score)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)

X = pd.read_pickle("./X.pkl")
y = pd.read_pickle("./y.pkl")
flights_clean = pd.read_pickle("./flights_clean.pkl")

# Chronological split (no shuffling)
sort_idx = flights_clean.sort_values(["month","day","hour"]).index
X = X.loc[sort_idx].reset_index(drop=True)
y = y.loc[sort_idx].reset_index(drop=True)
flights_sorted = flights_clean.loc[sort_idx].reset_index(drop=True)

n = len(X)
train_end = int(n*0.75)
val_end = int(n*0.90)

X_train, X_val, X_test = X.iloc[:train_end], X.iloc[train_end:val_end], X.iloc[val_end:]
y_train, y_val, y_test = y.iloc[:train_end], y.iloc[train_end:val_end], y.iloc[val_end:]

print("Train/Val/Test sizes:", len(X_train), len(X_val), len(X_test))

results = {}

def bootstrap_ci(y_true, y_pred_proba, y_pred, metric_fn, n_boot=500, use_proba=False):
    scores = []
    rng = np.random.RandomState(42)
    n = len(y_true)
    y_true = np.array(y_true)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        try:
            if use_proba:
                scores.append(metric_fn(y_true[idx], np.array(y_pred_proba)[idx]))
            else:
                scores.append(metric_fn(y_true[idx], np.array(y_pred)[idx]))
        except Exception:
            continue
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))

# ---------------- Logistic Regression ----------------
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

lr = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
lr.fit(X_train_s, y_train)
lr_proba = lr.predict_proba(X_test_s)[:,1]
lr_pred = (lr_proba >= 0.5).astype(int)

results["LogisticRegression"] = {
    "F1_macro": float(f1_score(y_test, lr_pred, average="macro")),
    "AUC_ROC": float(roc_auc_score(y_test, lr_proba)),
    "Precision": float(precision_score(y_test, lr_pred)),
    "Recall": float(recall_score(y_test, lr_pred)),
    "Brier": float(brier_score_loss(y_test, lr_proba)),
    "Accuracy": float(accuracy_score(y_test, lr_pred)),
}
f1_ci = bootstrap_ci(y_test.values, lr_proba, lr_pred, lambda yt,yp: f1_score(yt,yp,average="macro"))
results["LogisticRegression"]["F1_macro_95CI"] = f1_ci

# ---------------- Random Forest ----------------
rf = RandomForestClassifier(n_estimators=500, max_depth=10, random_state=42, n_jobs=-1, class_weight="balanced")
rf.fit(X_train, y_train)
rf_proba = rf.predict_proba(X_test)[:,1]
rf_pred = (rf_proba >= 0.5).astype(int)

results["RandomForest"] = {
    "F1_macro": float(f1_score(y_test, rf_pred, average="macro")),
    "AUC_ROC": float(roc_auc_score(y_test, rf_proba)),
    "Precision": float(precision_score(y_test, rf_pred)),
    "Recall": float(recall_score(y_test, rf_pred)),
    "Brier": float(brier_score_loss(y_test, rf_proba)),
    "Accuracy": float(accuracy_score(y_test, rf_pred)),
}
f1_ci = bootstrap_ci(y_test.values, rf_proba, rf_pred, lambda yt,yp: f1_score(yt,yp,average="macro"))
results["RandomForest"]["F1_macro_95CI"] = f1_ci

# ---------------- XGBoost ----------------
scale_pos_weight = (y_train==0).sum() / (y_train==1).sum()
xgbc = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, random_state=42,
                          scale_pos_weight=scale_pos_weight, eval_metric="logloss")
xgbc.fit(X_train, y_train)
xgb_proba = xgbc.predict_proba(X_test)[:,1]
xgb_pred = (xgb_proba >= 0.5).astype(int)

results["XGBoost"] = {
    "F1_macro": float(f1_score(y_test, xgb_pred, average="macro")),
    "AUC_ROC": float(roc_auc_score(y_test, xgb_proba)),
    "Precision": float(precision_score(y_test, xgb_pred)),
    "Recall": float(recall_score(y_test, xgb_pred)),
    "Brier": float(brier_score_loss(y_test, xgb_proba)),
    "Accuracy": float(accuracy_score(y_test, xgb_pred)),
}
f1_ci = bootstrap_ci(y_test.values, xgb_proba, xgb_pred, lambda yt,yp: f1_score(yt,yp,average="macro"))
results["XGBoost"]["F1_macro_95CI"] = f1_ci

# 5-fold CV (reliability/validity check) for XGBoost on training data
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_f1 = cross_val_score(xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                                           random_state=42, eval_metric="logloss"),
                          X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
results["XGBoost_5fold_CV_F1macro"] = {"mean": float(cv_f1.mean()), "std": float(cv_f1.std()), "folds": cv_f1.tolist()}

print(json.dumps(results, indent=2))
with open("./model_results.json","w") as f:
    json.dump(results, f, indent=2)

# Save test set + predictions for SHAP stage
X_test.to_pickle("./X_test.pkl")
y_test.to_pickle("./y_test.pkl")
X_train.to_pickle("./X_train.pkl")
y_train.to_pickle("./y_train.pkl")
import pickle
with open("./models.pkl","wb") as f:
    pickle.dump({"lr":lr,"rf":rf,"xgb":xgbc,"scaler":scaler}, f)

print("MODEL TRAINING DONE")
