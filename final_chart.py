import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("./model_results.json") as f:
    results = json.load(f)
with open("./lstm_results.json") as f:
    lstm = json.load(f)

models = ["LogisticRegression", "RandomForest", "XGBoost", "LSTM (NumPy, seq)"]
f1s = [results["LogisticRegression"]["F1_macro"], results["RandomForest"]["F1_macro"],
       results["XGBoost"]["F1_macro"], lstm["F1_macro"]]
aucs = [results["LogisticRegression"]["AUC_ROC"], results["RandomForest"]["AUC_ROC"],
        results["XGBoost"]["AUC_ROC"], lstm["AUC_ROC"]]

fig, axes = plt.subplots(1, 2, figsize=(11,4.5))
colors = ["#a0aec0","#4299e1","#2b6cb0","#805ad5"]
axes[0].bar(models, f1s, color=colors)
axes[0].set_ylabel("F1-macro (test set)")
axes[0].set_title("Model Comparison: F1-macro")
axes[0].tick_params(axis='x', rotation=25)

axes[1].bar(models, aucs, color=colors)
axes[1].set_ylabel("AUC-ROC (test set)")
axes[1].set_title("Model Comparison: AUC-ROC")
axes[1].tick_params(axis='x', rotation=25)

plt.tight_layout()
plt.savefig("./figures/model_comparison_perflight.png", dpi=150)
plt.close()
print("chart saved")
