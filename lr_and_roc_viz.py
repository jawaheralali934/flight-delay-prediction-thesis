import pandas as pd
import numpy as np
import pickle
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

np.random.seed(42)

X_test = pd.read_pickle("./X_test.pkl")
y_test = pd.read_pickle("./y_test.pkl")
X_train = pd.read_pickle("./X_train.pkl")
y_train = pd.read_pickle("./y_train.pkl")

with open("./models.pkl", "rb") as f:
    models = pickle.load(f)
lr = models["lr"]
rf = models["rf"]
xgbc = models["xgb"]
scaler = models["scaler"]

with open("./lstm_results.json") as f:
    lstm_results = json.load(f)

# =====================================================================
# VISUALIZATION 1: Logistic Regression coefficient (feature importance) plot
# This is the LR-equivalent of the SHAP bar charts already made for
# XGBoost/Random Forest -- LR has no SHAP-style chart otherwise.
# =====================================================================
feature_names = X_train.columns.tolist()
coefs = lr.coef_[0]  # standardized coefficients (X was scaled before fitting)

coef_df = pd.DataFrame({"feature": feature_names, "coefficient": coefs})
coef_df["abs_coef"] = coef_df["coefficient"].abs()
coef_df = coef_df.sort_values("abs_coef", ascending=False).head(15)
coef_df.to_csv("./lr_coefficients.csv", index=False)

plt.figure(figsize=(7.5,6))
colors = ["#c53030" if c < 0 else "#2b6cb0" for c in coef_df["coefficient"][::-1]]
plt.barh(coef_df["feature"][::-1], coef_df["coefficient"][::-1], color=colors)
plt.axvline(0, color="black", linewidth=0.8)
plt.xlabel("Standardized coefficient (log-odds)")
plt.title("Logistic Regression Coefficients -- Per-Flight Delay Classification\n(blue = increases delay odds, red = decreases delay odds)")
plt.tight_layout()
plt.savefig("./figures/lr_coefficients.png", dpi=150)
plt.close()

print("Top LR coefficients:")
print(coef_df.head(10).to_string(index=False))

# =====================================================================
# VISUALIZATION 2: ROC curve comparison (LR, RF, XGBoost; the LSTM uses a separate test set)
# =====================================================================
X_test_s = scaler.transform(X_test)
lr_proba = lr.predict_proba(X_test_s)[:,1]
rf_proba = rf.predict_proba(X_test)[:,1]
xgb_proba = xgbc.predict_proba(X_test)[:,1]

fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_proba)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_proba)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_proba)

auc_lr = auc(fpr_lr, tpr_lr)
auc_rf = auc(fpr_rf, tpr_rf)
auc_xgb = auc(fpr_xgb, tpr_xgb)
auc_lstm = lstm_results["AUC_ROC"]

plt.figure(figsize=(6.5,6))
plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC={auc_lr:.3f})", color="#a0aec0", linewidth=2)
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC={auc_rf:.3f})", color="#4299e1", linewidth=2)
plt.plot(fpr_xgb, tpr_xgb, label=f"XGBoost (AUC={auc_xgb:.3f})", color="#2b6cb0", linewidth=2)
plt.plot([0,1],[0,1], linestyle="--", color="gray", linewidth=1, label="Chance (AUC=0.500)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison -- Per-Flight Delay Classification")
plt.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig("./figures/roc_comparison.png", dpi=150)
plt.close()

print(f"\nAUCs -> LR: {auc_lr:.3f}, RF: {auc_rf:.3f}, XGB: {auc_xgb:.3f}, LSTM (from Ch.4 Table 4.3): {auc_lstm:.3f}")

# =====================================================================
# VISUALIZATION 3: LR predicted-probability distribution plot
# (shows visually WHY LR's recall was near-zero: predicted probabilities
#  rarely cross the 0.5 threshold, clustering below it)
# =====================================================================
plt.figure(figsize=(7,4.5))
plt.hist(lr_proba[y_test==0], bins=40, alpha=0.6, label="Actually on-time", color="#a0aec0", density=True)
plt.hist(lr_proba[y_test==1], bins=40, alpha=0.6, label="Actually delayed", color="#c53030", density=True)
plt.axvline(0.5, color="black", linestyle="--", linewidth=1, label="Decision threshold (0.5)")
plt.xlabel("Logistic Regression predicted P(delayed)")
plt.ylabel("Density")
plt.title("Logistic Regression Predicted-Probability Distribution\nby True Class -- Per-Flight Test Set")
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig("./figures/lr_probability_distribution.png", dpi=150)
plt.close()

print("\nAll 3 visualizations saved to ./figures/")
