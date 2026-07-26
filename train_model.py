"""
train_model.py
--------------
Phase 3 - Machine Learning
Customer Churn Prediction & Business Intelligence System

Trains & compares multiple ML models on the preprocessed data produced by
preprocessing.py (data/train.csv, data/test.csv), evaluates them on
Accuracy / Precision / Recall / F1 / ROC-AUC / Confusion Matrix,
selects the best model, and persists it as model.joblib for Phase 5
(prediction system) and Phase 6 (dashboard).

Also generates:
    - reports/model_comparison.csv        (metrics table for all models)
    - reports/confusion_matrices.html     (Plotly confusion matrices)
    - reports/roc_curves.html             (Plotly ROC curves, all models)
    - reports/feature_importance.html     (Plotly feature importance - best model)
    - models/model.joblib                 (best trained model)
    - models/model_metadata.joblib        (best model name + feature list + metrics)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix,
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[Warning] xgboost not installed - skipping XGBoost model. "
          "Install with: pip install xgboost")

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
TARGET_COL = "Churn"

MODEL_PATH = "models/model.joblib"          # final best model (matches project structure: model.joblib)
METADATA_PATH = "models/model_metadata.joblib"
REPORTS_DIR = "reports"
RANDOM_STATE = 42

# Primary metric used to pick the "best" model.
# ROC-AUC is preferred for churn (imbalanced target, ranking-based business use).
SELECTION_METRIC = "roc_auc"


# --------------------------------------------------------------------------
# 1. LOAD TRAIN/TEST DATA
# --------------------------------------------------------------------------
def load_train_test(train_path: str, test_path: str, target_col: str):
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        raise FileNotFoundError(
            "train.csv / test.csv not found. Run preprocessing.py first."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    # Safety net: if the target is still Yes/No strings (e.g. train.csv was
    # generated before the preprocessing.py fix), encode it here instead of
    # crashing. Ideally this branch should never trigger - re-run
    # preprocessing.py to regenerate clean train/test CSVs.
    if y_train.dtype == object or str(y_train.dtype) == "string":
        print("[Warning] Target column is non-numeric - encoding Yes/No -> 1/0. "
              "Consider re-running preprocessing.py to avoid this in future.")
        y_train = y_train.astype(str).str.strip().map({"Yes": 1, "No": 0}).astype(int)
        y_test = y_test.astype(str).str.strip().map({"Yes": 1, "No": 0}).astype(int)

    print(f"[Data] X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"[Data] Churn rate - train: {y_train.mean():.2%}, test: {y_test.mean():.2%}")

    return X_train, X_test, y_train, y_test


# --------------------------------------------------------------------------
# 2. DEFINE MODELS
# --------------------------------------------------------------------------
def get_models():
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=10, random_state=RANDOM_STATE,
            class_weight="balanced", n_jobs=-1
        ),
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            random_state=RANDOM_STATE, eval_metric="logloss",
            use_label_encoder=False,
        )

    return models


# --------------------------------------------------------------------------
# 3. TRAIN + EVALUATE ALL MODELS
# --------------------------------------------------------------------------
def train_and_evaluate(models: dict, X_train, y_train, X_test, y_test):
    results = {}
    trained_models = {}

    for name, model in models.items():
        print(f"\n[Training] {name} ...")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {"metrics": metrics, "confusion_matrix": cm, "y_proba": y_proba}
        trained_models[name] = model

        print(f"[{name}] "
              f"Acc: {metrics['accuracy']:.4f} | "
              f"Prec: {metrics['precision']:.4f} | "
              f"Recall: {metrics['recall']:.4f} | "
              f"F1: {metrics['f1_score']:.4f} | "
              f"ROC-AUC: {metrics['roc_auc']:.4f}")

    return results, trained_models


# --------------------------------------------------------------------------
# 4. SELECT BEST MODEL
# --------------------------------------------------------------------------
def select_best_model(results: dict, metric: str = SELECTION_METRIC):
    best_name = max(results, key=lambda name: results[name]["metrics"][metric])
    best_score = results[best_name]["metrics"][metric]
    print(f"\n[Selection] Best model -> {best_name} ({metric} = {best_score:.4f})")

    print("[Justification] Selected based on highest ROC-AUC, which best reflects "
          "the model's ability to rank customers by churn risk - the core need "
          "for prioritizing retention efforts on an imbalanced churn dataset.")

    return best_name


# --------------------------------------------------------------------------
# 5. SAVE COMPARISON TABLE
# --------------------------------------------------------------------------
def save_comparison_table(results: dict, out_dir: str):
    rows = []
    for name, res in results.items():
        row = {"Model": name}
        row.update(res["metrics"])
        rows.append(row)

    comparison_df = pd.DataFrame(rows).sort_values(by=SELECTION_METRIC, ascending=False)
    path = os.path.join(out_dir, "model_comparison.csv")
    comparison_df.to_csv(path, index=False)
    print(f"[Saved] {path}")
    print("\n", comparison_df.to_string(index=False))
    return comparison_df


# --------------------------------------------------------------------------
# 6. VISUALIZATIONS (Plotly)
# --------------------------------------------------------------------------
def plot_confusion_matrices(results: dict, out_dir: str):
    names = list(results.keys())
    fig = make_subplots(rows=1, cols=len(names), subplot_titles=names)

    for i, name in enumerate(names, start=1):
        cm = results[name]["confusion_matrix"]
        fig.add_trace(
            go.Heatmap(
                z=cm, x=["Pred: No Churn", "Pred: Churn"],
                y=["Actual: No Churn", "Actual: Churn"],
                colorscale="Blues", showscale=(i == len(names)),
                text=cm, texttemplate="%{text}",
            ),
            row=1, col=i,
        )

    fig.update_layout(title="Confusion Matrices - All Models", height=400, width=350 * len(names))
    path = os.path.join(out_dir, "confusion_matrices.html")
    fig.write_html(path)
    print(f"[Saved] {path}")


def plot_roc_curves(results: dict, y_test, out_dir: str):
    fig = go.Figure()
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
        auc = res["metrics"]["roc_auc"]
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={auc:.3f})"))

    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                              line=dict(dash="dash", color="gray"), name="Random Guess"))

    fig.update_layout(
        title="ROC Curves - Model Comparison",
        xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
        height=500, width=700,
    )
    path = os.path.join(out_dir, "roc_curves.html")
    fig.write_html(path)
    print(f"[Saved] {path}")


def plot_feature_importance(model, feature_names, best_name: str, out_dir: str):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        print(f"[Feature Importance] {best_name} does not expose importances; skipping.")
        return None

    imp_df = pd.DataFrame({
        "Feature": feature_names, "Importance": importances
    }).sort_values(by="Importance", ascending=True).tail(15)

    fig = go.Figure(go.Bar(
        x=imp_df["Importance"], y=imp_df["Feature"], orientation="h",
        marker_color="teal",
    ))
    fig.update_layout(
        title=f"Top 15 Feature Importances - {best_name}",
        xaxis_title="Importance", height=600, width=800,
    )
    path = os.path.join(out_dir, "feature_importance.html")
    fig.write_html(path)
    print(f"[Saved] {path}")

    return imp_df.sort_values(by="Importance", ascending=False)


# --------------------------------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------------------------------
def run_training_pipeline():
    os.makedirs("models", exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # 1. Load data
    X_train, X_test, y_train, y_test = load_train_test(TRAIN_PATH, TEST_PATH, TARGET_COL)

    # 2. Get models
    models = get_models()

    # 3. Train + evaluate
    results, trained_models = train_and_evaluate(models, X_train, y_train, X_test, y_test)

    # 4. Comparison table
    comparison_df = save_comparison_table(results, REPORTS_DIR)

    # 5. Select best model
    best_name = select_best_model(results, SELECTION_METRIC)
    best_model = trained_models[best_name]

    # 6. Visualizations
    plot_confusion_matrices(results, REPORTS_DIR)
    plot_roc_curves(results, y_test, REPORTS_DIR)
    top_features = plot_feature_importance(best_model, X_train.columns.tolist(), best_name, REPORTS_DIR)

    # 7. Persist best model + metadata
    joblib.dump(best_model, MODEL_PATH)

    metadata = {
        "best_model_name": best_name,
        "metrics": results[best_name]["metrics"],
        "feature_names": X_train.columns.tolist(),
        "selection_metric": SELECTION_METRIC,
        "top_features": (
            top_features.to_dict(orient="records") if top_features is not None else None
        ),
    }
    joblib.dump(metadata, METADATA_PATH)

    # Human-readable summary for the business insights phase
    with open(os.path.join(REPORTS_DIR, "model_summary.json"), "w") as f:
        json.dump(
            {k: (v if not isinstance(v, np.floating) else float(v))
             for k, v in metadata.items() if k != "top_features"},
            f, indent=2, default=str
        )

    print(f"\n[Saved] {MODEL_PATH}")
    print(f"[Saved] {METADATA_PATH}")
    print(f"\n✅ Phase 3 (Model Training & Evaluation) complete. Best model: {best_name}")

    return {
        "results": results,
        "trained_models": trained_models,
        "best_model_name": best_name,
        "best_model": best_model,
        "comparison_df": comparison_df,
    }


if __name__ == "__main__":
    run_training_pipeline()