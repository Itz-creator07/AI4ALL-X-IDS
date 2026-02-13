"""
X-IDS: Explainable Network Intrusion Detection System
Streamlit deployment app
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import shap
import os
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
MODELS = {
    "XGBoost (Optimized)": BASE_DIR / "results" / "xgboost_model_optimized.pkl",
    "Random Forest":        BASE_DIR / "results" / "rf_model_final.pkl",
}
FEATURE_NAMES_FILE = BASE_DIR / "data" / "feature_names.txt"
RESULTS_DIR = BASE_DIR / "results"

# ─── Feature defaults (median-like values for demo sliders) ───────────────────
FEATURE_DEFAULTS = {
    "Bwd Packet Length Std":  20.0,
    "Bwd Packet Length Max":  60.0,
    "Bwd Packet Length Mean": 30.0,
    "Avg Bwd Segment Size":   30.0,
    "Packet Length Std":      25.0,
    "Max Packet Length":      80.0,
    "Packet Length Variance": 625.0,
    "Fwd IAT Std":            50000.0,
    "Packet Length Mean":     40.0,
    "Average Packet Size":    45.0,
    "Idle Max":               0.0,
    "Idle Mean":              0.0,
    "Flow IAT Max":           100000.0,
    "Fwd IAT Max":            80000.0,
    "Idle Min":               0.0,
    "Flow IAT Std":           60000.0,
    "Min Packet Length":      0.0,
    "Bwd Packet Length Min":  0.0,
    "Fwd IAT Total":          200000.0,
    "FIN Flag Count":         0.0,
    "PSH Flag Count":         0.0,
    "Flow IAT Mean":          50000.0,
    "Bwd IAT Std":            40000.0,
    "Fwd IAT Mean":           60000.0,
    "Destination Port":       80.0,
    "URG Flag Count":         0.0,
    "Fwd Packet Length Min":  0.0,
    "ACK Flag Count":         0.0,
    "Bwd IAT Max":            90000.0,
    "Idle Std":               0.0,
}

# ─── Helpers ──────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_feature_names() -> list[str]:
    with open(FEATURE_NAMES_FILE) as f:
        return [line.strip() for line in f if line.strip()]

def predict_and_explain(model, features_df):
    """Return prediction, probability, and SHAP values."""
    pred = model.predict(features_df)[0]
    prob = model.predict_proba(features_df)[0]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features_df)
    # For binary classifiers shap_values may be list [class0, class1]
    if isinstance(shap_values, list):
        sv = shap_values[1]  # class 1 = Attack
    else:
        sv = shap_values
    return pred, prob, sv, explainer.expected_value

def risk_level(prob_attack: float) -> tuple[str, str]:
    if prob_attack >= 0.85:
        return "HIGH", "🔴"
    elif prob_attack >= 0.50:
        return "MEDIUM", "🟡"
    else:
        return "LOW", "🟢"

def generate_triage_report(pred, prob_attack, feature_names, shap_vals, input_vals):
    level, icon = risk_level(prob_attack)
    top_idx = np.argsort(np.abs(shap_vals[0]))[::-1][:5]
    top_features = [(feature_names[i], shap_vals[0][i], input_vals[i]) for i in top_idx]

    lines = [
        "─" * 52,
        "  X-IDS  |  SOC TIER 1 TRIAGE REPORT",
        "─" * 52,
        f"  Verdict       : {'ATTACK' if pred == 1 else 'BENIGN'}",
        f"  Risk Level    : {icon} {level}",
        f"  Attack Prob   : {prob_attack:.1%}",
        "",
        "  Top Contributing Features:",
    ]
    for rank, (fname, sv, val) in enumerate(top_features, 1):
        direction = "↑ increases risk" if sv > 0 else "↓ decreases risk"
        lines.append(f"    {rank}. {fname}")
        lines.append(f"       Value={val:.4g}  SHAP={sv:+.4f}  ({direction})")
    lines.append("")
    if pred == 1:
        lines.append("  Recommended Action: ESCALATE to Tier 2 analyst.")
        lines.append("  Review flagged features for IOC correlation.")
    else:
        lines.append("  Recommended Action: CLOSE alert — no threat detected.")
    lines.append("─" * 52)
    return "\n".join(lines)

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="X-IDS | Explainable IDS",
    page_icon="🛡️",
    layout="wide",
)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🛡️ X-IDS")
    st.caption("Explainable Network Intrusion Detection System")
    st.divider()
    model_name = st.selectbox(
        "Model",
        list(MODELS.keys()),
        index=0,
        help="XGBoost is recommended on free-tier hosting (lower memory usage).",
    )
    st.divider()
    page = st.radio(
        "Navigate",
        ["Home", "Single Prediction", "Batch Prediction", "Global Explainability"],
    )
    st.divider()
    st.caption("CICIDS2017 dataset · XGBoost + SHAP")

model = load_model(MODELS[model_name])
feature_names = load_feature_names()

# ─── HOME ─────────────────────────────────────────────────────────────────────
if page == "Home":
    st.title("X-IDS: Explainable Network Intrusion Detection System")
    st.markdown(
        """
        > **Research Question:** How can Explainable AI (XAI), specifically SHAP-enhanced XGBoost
        > models, reduce alert fatigue and improve efficiency in SOC Tier 1 network intrusion
        > detection triage through automated, transparent, and actionable threat explanations?
        """
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy",   "99.67%", "↑ vs 95% target")
    col2.metric("Recall",     "99.70%", "↑ vs 92% target")
    col3.metric("FPR",        "0.34%",  "↓ vs 5% target")
    col4.metric("ROC-AUC",    "0.9999", "↑ vs 0.97 target")

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Dataset")
        st.markdown(
            """
            - **CICIDS2017** — 2.83 M network flows (Mon–Fri)
            - **80.3%** benign · **19.7%** attack
            - **30** selected features from 79 raw
            - **15** attack types (DDoS, DoS, PortScan, …)
            """
        )
    with col_b:
        st.subheader("Operational Impact")
        st.markdown(
            """
            - **90%** triage time reduction (5 min → 30 sec/alert)
            - **80%** fewer analysts needed per 10 K alerts/day
            - **$800 K+** estimated annual savings per SOC
            - No single feature dominates (top: 16.28%)
            """
        )

    img_path = RESULTS_DIR / "shap_global_importance.png"
    if img_path.exists():
        st.divider()
        st.subheader("Global SHAP Feature Importance")
        st.image(str(img_path))

# ─── SINGLE PREDICTION ────────────────────────────────────────────────────────
elif page == "Single Prediction":
    st.title("Single Flow Prediction")
    st.caption("Enter 30 network flow features to classify one connection and get a SHAP explanation.")

    with st.form("prediction_form"):
        cols = st.columns(3)
        input_values = {}
        for idx, fname in enumerate(feature_names):
            default = FEATURE_DEFAULTS.get(fname, 0.0)
            input_values[fname] = cols[idx % 3].number_input(
                fname, value=float(default), format="%.4f", key=fname
            )
        submitted = st.form_submit_button("Predict & Explain", type="primary")

    if submitted:
        input_df = pd.DataFrame([input_values])[feature_names]
        pred, prob, shap_vals, base_val = predict_and_explain(model, input_df)
        prob_attack = prob[1]
        level, icon = risk_level(prob_attack)

        # ── Verdict banner
        if pred == 1:
            st.error(f"{icon} **ATTACK DETECTED** — Risk: {level} ({prob_attack:.1%})")
        else:
            st.success(f"{icon} **BENIGN** — Risk: {level} ({prob_attack:.1%})")

        col_r, col_p = st.columns([1, 1])

        # ── Triage report
        with col_r:
            report = generate_triage_report(
                pred, prob_attack, feature_names,
                shap_vals, list(input_values.values())
            )
            st.subheader("SOC Triage Report")
            st.code(report, language=None)

        # ── SHAP waterfall plot
        with col_p:
            st.subheader("SHAP Explanation")
            fig, ax = plt.subplots(figsize=(7, 5))
            top_n = 10
            top_idx = np.argsort(np.abs(shap_vals[0]))[::-1][:top_n]
            top_sv  = shap_vals[0][top_idx]
            top_fn  = [feature_names[i] for i in top_idx]
            colors  = ["#d9534f" if v > 0 else "#5bc0de" for v in top_sv]
            ax.barh(range(top_n), top_sv[::-1], color=colors[::-1])
            ax.set_yticks(range(top_n))
            ax.set_yticklabels(top_fn[::-1], fontsize=9)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("SHAP value (impact on model output)")
            ax.set_title(f"Top {top_n} Features — {'Attack' if pred==1 else 'Benign'}")
            ax.set_facecolor("#0e1117")
            fig.patch.set_facecolor("#0e1117")
            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("white")
            ax.title.set_color("white")
            st.pyplot(fig)
            plt.close(fig)

# ─── BATCH PREDICTION ─────────────────────────────────────────────────────────
elif page == "Batch Prediction":
    st.title("Batch Prediction")
    st.markdown(
        f"Upload a CSV with **{len(feature_names)} columns** matching the feature names below. "
        "The app will classify each row and return a downloadable results file."
    )
    with st.expander("Required column names"):
        st.code("\n".join(feature_names))

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        missing = [f for f in feature_names if f not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            with st.spinner("Running predictions…"):
                X = df[feature_names]
                preds = model.predict(X)
                probs = model.predict_proba(X)[:, 1]
                df["Prediction"] = ["ATTACK" if p == 1 else "BENIGN" for p in preds]
                df["Attack_Probability"] = probs.round(4)
                df["Risk_Level"] = [risk_level(p)[0] for p in probs]

            st.success(f"Processed {len(df):,} flows.")
            attack_count = (preds == 1).sum()
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Flows", f"{len(df):,}")
            col2.metric("Attacks Detected", f"{attack_count:,}")
            col3.metric("Attack Rate", f"{attack_count/len(df):.1%}")

            st.dataframe(
                df[["Prediction", "Attack_Probability", "Risk_Level"] + feature_names[:5]],
                use_container_width=True,
            )
            csv_out = df.to_csv(index=False).encode()
            st.download_button(
                "Download Full Results CSV",
                data=csv_out,
                file_name="xids_predictions.csv",
                mime="text/csv",
            )

# ─── GLOBAL EXPLAINABILITY ────────────────────────────────────────────────────
elif page == "Global Explainability":
    st.title("Global Explainability")
    st.caption("Pre-computed SHAP visualizations from the training run (1,000 test samples).")

    plots = {
        "SHAP Summary (Beeswarm)":    "shap_summary.png",
        "SHAP Global Importance":     "shap_global_importance.png",
        "Force Plot — Attack Sample": "shap_force_plot_attack.png",
        "Confusion Matrix":           "confusion_matrix.png",
        "ROC Curve":                  "roc_curve.png",
    }

    for title, fname in plots.items():
        path = RESULTS_DIR / fname
        if path.exists():
            st.subheader(title)
            st.image(str(path))
        else:
            st.info(f"{title}: file not found ({fname})")

    # Show pre-computed SHAP values heatmap for first 50 samples
    shap_file = RESULTS_DIR / "shap_values.npy"
    if shap_file.exists():
        st.subheader("SHAP Values Heatmap (first 50 samples)")
        sv = np.load(str(shap_file))
        if sv.ndim == 3:
            sv = sv[:, :, 1]  # attack class
        sv_df = pd.DataFrame(sv[:50], columns=feature_names)
        st.dataframe(sv_df.style.background_gradient(cmap="RdBu_r", axis=None), use_container_width=True)
