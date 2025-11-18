# %% [markdown]
# X-IDS: Random Forest Training, Tuning & SHAP Explainability
#
# Purpose: Train a RandomForestClassifier on processed dataset produced
# by notebook 01, optionally tune hyperparameters, compute SHAP explanations,
# generate sample case reports, and save model + artifacts to `results/`.

# %% [markdown]
# ## Install required packages (run once)
# If running locally and packages already installed, you can skip this cell.

# %% [markdown]
# ## Imports & Configuration

# %%
import os
import json
import time
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from sklearn.model_selection import RandomizedSearchCV
import shap
import joblib
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 200)
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

print("="*80)
print("X-IDS RANDOM FOREST TRAINING & SHAP")
print("="*80)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Random Seed: {RANDOM_SEED}")
print("="*80)

# %% [markdown]
# ## Paths

# %%
DATA_DIR = "../data"
RESULTS_DIR = "../results"
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"Data dir: {DATA_DIR}")
print(f"Results dir: {RESULTS_DIR}")

# %% [markdown]
# ## Load processed data (from notebook 01)

# %%
print("="*80)
print("LOADING PROCESSED DATA")
print("="*80)

X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train_binary.csv"))["is_attack"]
X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test_binary.csv"))["is_attack"]

with open(os.path.join(DATA_DIR, "feature_names.txt"), "r") as f:
    feature_names = [line.strip() for line in f]

with open(os.path.join(DATA_DIR, "label_mapping.json"), "r") as f:
    label_mapping = json.load(f)

print(f"Train shape: {X_train.shape}")
print(f"Test shape:  {X_test.shape}")
print(f"Features:    {len(feature_names)}")
print(f"Label types: {len(label_mapping)}")
print("Class distribution (train):")
print(y_train.value_counts(normalize=False))
print("="*80)

# %% [markdown]
# ## Baseline Random Forest (quick smoke test)
#
# Notes:
# - Uses `class_weight='balanced'` to help class imbalance.
# - This is a baseline run with `n_estimators=100`.

# %%
print("="*80)
print("TRAINING BASELINE RANDOM FOREST")
print("="*80)

rf_baseline = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=RANDOM_SEED,
    n_jobs=-1,
    class_weight="balanced"
)

start = time.time()
rf_baseline.fit(X_train, y_train)
train_time = time.time() - start

y_pred = rf_baseline.predict(X_test)
y_proba = rf_baseline.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label=1)
recall = recall_score(y_test, y_pred, pos_label=1)
f1 = f1_score(y_test, y_pred, pos_label=1)
roc_auc = roc_auc_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
fpr = fp / (fp + tn)

print(f"Training time: {train_time:.2f}s")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (Attack): {precision:.4f}")
print(f"Recall (Attack): {recall:.4f}")
print(f"F1 (Attack): {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"False Positive Rate: {fpr:.4f}")

# Save baseline model
joblib.dump(rf_baseline, os.path.join(RESULTS_DIR, "rf_model_baseline.pkl"))
print(f"Saved baseline RF to: {os.path.join(RESULTS_DIR, 'rf_model_baseline.pkl')}")

# %% [markdown]
# ## Optional: Randomized Hyperparameter Search (recommended)
# This is optional and can be time-consuming. Adjust `n_iter` and `cv` for speed.

# %%
run_hyperparam_search = False  # set to True to run tuning

if run_hyperparam_search:
    print("="*80)
    print("RUNNING RANDOMIZED SEARCH FOR RF HYPERPARAMS")
    print("="*80)

    param_dist = {
        "n_estimators": [100, 200, 400],
        "max_depth": [None, 10, 20, 40],
        "max_features": ["sqrt", "log2", 0.2, 0.5],
        "min_samples_leaf": [1, 2, 4, 8]
    }

    rf_base = RandomForestClassifier(random_state=RANDOM_SEED, class_weight="balanced", n_jobs=-1)
    rnd = RandomizedSearchCV(rf_base, param_distributions=param_dist,
                             n_iter=12, cv=3, scoring="roc_auc", verbose=2, random_state=RANDOM_SEED)
    start = time.time()
    rnd.fit(X_train, y_train)
    print(f"Randomized search complete in {(time.time()-start):.1f}s")
    print("Best params:", rnd.best_params_)
    best_rf = rnd.best_estimator_
    joblib.dump(best_rf, os.path.join(RESULTS_DIR, "rf_model_tuned.pkl"))
    print(f"Saved tuned RF to: {os.path.join(RESULTS_DIR, 'rf_model_tuned.pkl')}")
else:
    best_rf = rf_baseline

# %% [markdown]
# ## Final Evaluation (use tuned model if available)

# %%
model = best_rf

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label=1)
recall = recall_score(y_test, y_pred, pos_label=1)
f1 = f1_score(y_test, y_pred, pos_label=1)
roc_auc = roc_auc_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
fpr = fp / (fp + tn)

print("="*80)
print("FINAL MODEL PERFORMANCE")
print("="*80)
print(f"Accuracy:                {accuracy:.4f}")
print(f"ROC-AUC:                 {roc_auc:.4f}")
print(f"Precision (Attack):      {precision:.4f}")
print(f"Recall (Attack):         {recall:.4f}")
print(f"F1 (Attack):             {f1:.4f}")
print(f"False Positive Rate:     {fpr:.4f}")
print("="*80)

# Save final model
joblib.dump(model, os.path.join(RESULTS_DIR, "rf_model_final.pkl"))
print(f"Saved final model to: {os.path.join(RESULTS_DIR, 'rf_model_final.pkl')}")

# %% [markdown]
# ## SHAP Explainability
# TreeExplainer works with sklearn RandomForestClassifier.

# %%
print("="*80)
print("SHAP EXPLAINABILITY (TreeExplainer)")
print("="*80)

# Choose subset for SHAP for speed
sample_size = min(1000, len(X_test))
print(f"Computing SHAP on {sample_size} samples (adjust sample_size if needed)")
X_test_sample = X_test.head(sample_size)
y_test_sample = y_test.head(sample_size)

start = time.time()
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_sample)
shap_time = time.time() - start
print(f"SHAP computation time: {shap_time:.2f}s")

# Save SHAP values and sample
np.save(os.path.join(RESULTS_DIR, "rf_shap_values.npy"), shap_values)
X_test_sample.to_csv(os.path.join(RESULTS_DIR, "rf_shap_X_sample.csv"), index=False)
print("Saved SHAP artifacts")

# %% [markdown]
# ## SHAP Global Importance & Summary Plot

# %%
# Global importance bar plot
plt.figure(figsize=(10, 8))
try:
    shap.summary_plot(shap_values, X_test_sample, plot_type="bar", show=False)
    plt.title("Global Feature Importance (SHAP)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "rf_shap_global_importance.png"), dpi=300, bbox_inches="tight")
    plt.show()
    print("Saved rf_shap_global_importance.png")
except Exception as e:
    print("Could not create SHAP summary plot:", e)

# Print numeric top features
if isinstance(shap_values, list):
    # older shap returns list for classifiers: shap_values[1] corresponds to positive class
    arr = np.abs(shap_values[1]).mean(axis=0)
else:
    arr = np.abs(shap_values).mean(axis=0)

top_features = pd.DataFrame({
    "Feature": X_test_sample.columns,
    "Mean |SHAP|": arr
}).sort_values("Mean |SHAP|", ascending=False)

print("Top 10 features (SHAP mean |value|):")
print(top_features.head(10).to_string(index=False))

# %% [markdown]
# ## Individual prediction explanation & force plot (example)

# %%
# Find first attack in sample
attack_indices = [i for i, v in enumerate(y_test_sample) if v == 1]
if len(attack_indices) > 0:
    idx = attack_indices[0]
    print(f"Explaining sample index {idx} (actual attack). Model confidence: {model.predict_proba(X_test_sample.iloc[[idx]])[0][1]*100:.2f}%")
    try:
        shap.force_plot(explainer.expected_value, shap_values[idx], X_test_sample.iloc[idx], matplotlib=True, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "rf_shap_force_plot_attack.png"), dpi=300, bbox_inches="tight")
        plt.show()
        print("Saved rf_shap_force_plot_attack.png")
    except Exception as e:
        print("Could not generate force plot:", e)
else:
    print("No attack in SHAP sample subset to demonstrate force plot.")

# %% [markdown]
# ## Case report generator (similar style to XGBoost notebook)

# %%
def generate_case_report(sample_idx, X_sample, shap_values, model, y_true=None, alert_id=None):
    prediction = model.predict(X_sample.iloc[[sample_idx]])[0]
    probabilities = model.predict_proba(X_sample.iloc[[sample_idx]])[0]
    risk_score = probabilities[1] * 100
    # extract shap vector for positive class if shap_values is list
    if isinstance(shap_values, list):
        sv = shap_values[1][sample_idx]
    else:
        sv = shap_values[sample_idx]

    df = pd.DataFrame({
        "Feature": X_sample.columns,
        "Value": X_sample.iloc[sample_idx].values,
        "SHAP Impact": sv
    })
    df["Abs_Impact"] = df["SHAP Impact"].abs()
    df = df.sort_values("Abs_Impact", ascending=False)
    top = df.head(5)

    alert_id = alert_id if alert_id else f"RF-ALERT-{sample_idx:06d}"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = "="*80 + "\n"
    report += "AUTOMATED TIER 1 TRIAGE REPORT - X-IDS (Random Forest)\n"
    report += "="*80 + "\n"
    report += f"Alert ID: {alert_id}\n"
    report += f"Timestamp: {timestamp}\n"
    report += "="*80 + "\n"
    report += "PREDICTION:\n"
    report += f"   Classification: {' ATTACK DETECTED' if prediction == 1 else ' BENIGN TRAFFIC'}\n"
    report += f"   Risk Score: {risk_score:.2f}%\n"
    report += f"   Confidence: {probabilities[prediction]*100:.2f}%\n"
    if y_true is not None:
        actual = "ATTACK" if y_true == 1 else "BENIGN"
        correct = "" if y_true == prediction else ""
        report += f"   Ground Truth: {actual} {correct}\n"
    report += "="*80 + "\n"
    report += "TOP CONTRIBUTING FACTORS (SHAP):\n"
    for i, r in enumerate(top.itertuples(), 1):
        direction = " INCREASES" if r._3 > 0 else " DECREASES"
        report += f"  {i}. {r.Feature}\n"
        report += f"      Feature Value: {r.Value:.4f}\n"
        report += f"      SHAP Impact: {r._3:+.4f} ({direction} attack likelihood)\n"
    report += "="*80 + "\n"
    if risk_score > 80:
        report += "  HIGH RISK - ESCALATE TO TIER 2 IMMEDIATELY\n"
    elif risk_score > 50:
        report += "  MEDIUM RISK - MANUAL REVIEW\n"
    elif risk_score > 20:
        report += "ℹ  LOW RISK - MONITOR\n"
    else:
        report += " BENIGN - AUTO-CLOSE\n"
    report += "="*80 + "\n"
    report += "SYSTEM INFORMATION:\n"
    report += "   Model: RandomForestClassifier\n"
    report += "   Explainability: SHAP TreeExplainer\n"
    report += "="*80 + "\n"
    return report

# %% [markdown]
# ## Generate sample case reports and save them

# %%
reports = []
#find one attack and one benign in the SHAP sample
attack_indices = [i for i, v in enumerate(y_test_sample) if v == 1]
benign_indices = [i for i, v in enumerate(y_test_sample) if v == 0]

if len(attack_indices) > 0:
    r = generate_case_report(attack_indices[0], X_test_sample, shap_values, model, y_true=y_test_sample.iloc[attack_indices[0]], alert_id="DEMO-RF-ATTACK-001")
    reports.append(("attack_report_rf.txt", r))
    print(r)

if len(benign_indices) > 0:
    r = generate_case_report(benign_indices[0], X_test_sample, shap_values, model, y_true=y_test_sample.iloc[benign_indices[0]], alert_id="DEMO-RF-BENIGN-001")
    reports.append(("benign_report_rf.txt", r))
    print(r)

for fname, text in reports:
    with open(os.path.join(RESULTS_DIR, fname), "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved {fname}")

# %% [markdown]
# ## Triage timing simulation (model inference + SHAP per sample)

# %%
manual_triage_time = 5 * 60  # 5 minutes -> seconds
automated_review_time = 30   # seconds

num_test_alerts = 10
test_sample = X_test.head(num_test_alerts)
start = time.time()
_ = model.predict(test_sample)
_ = model.predict_proba(test_sample)
_ = explainer.shap_values(test_sample)
end = time.time()
per_alert = (end - start) / num_test_alerts

total_automated_time = per_alert + automated_review_time
time_saved = manual_triage_time - total_automated_time
reduction_pct = (time_saved / manual_triage_time) * 100

print(f"Per-alert model+SHAP: {per_alert:.2f}s")
print(f"Total automated (incl. review): {total_automated_time:.2f}s")
print(f"Time saved per alert: {time_saved/60:.2f} minutes ({reduction_pct:.1f}% reduction)")

# %% [markdown]
# ## Final summary and cleanup

# %%
summary = {
    "model": "RandomForestClassifier",
    "train_samples": len(X_train),
    "test_samples": len(X_test),
    "accuracy": float(accuracy),
    "precision_attack": float(precision),
    "recall_attack": float(recall),
    "f1_attack": float(f1),
    "roc_auc": float(roc_auc),
    "fpr": float(fpr),
    "training_time_seconds": float(train_time),
    "shap_samples": int(sample_size),
    "shap_time_seconds": float(shap_time)
}
with open(os.path.join(RESULTS_DIR, "rf_final_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("Saved rf_final_summary.json")
print("Finished.")
