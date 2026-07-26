"""
predict.py
----------
Phase 4 - Model Explainability
Phase 5 - Prediction System
Customer Churn Prediction & Business Intelligence System

Loads the trained model + saved encoders/scaler (from preprocessing.py /
train_model.py) and exposes a clean prediction API used by:
    - dashboard.py / app.py (Streamlit interactive prediction form)
    - this file's own CLI mode (for quick manual testing)

Given raw customer input (same fields as the original dataset), this module:
    1. Re-applies the exact same feature engineering / encoding / scaling
       used during training (so train-serve skew is avoided).
    2. Returns: Churn Prediction, Churn Probability, Risk Level,
       Prediction Confidence, and Key Contributing Factors.
    3. Uses SHAP for per-customer explanations when available,
       falling back to global feature importance otherwise.
"""

import os
import joblib
import numpy as np
import pandas as pd

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[Warning] shap not installed - falling back to global feature "
          "importance for explanations. Install with: pip install shap")

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
MODEL_PATH = "models/model.joblib"
METADATA_PATH = "models/model_metadata.joblib"
ENCODERS_PATH = "models/encoders.joblib"
SCALER_PATH = "models/scaler.joblib"

RISK_THRESHOLDS = {"low": 0.30, "medium": 0.60}  # < low -> Low, < medium -> Medium, else High

# Columns engineered exactly like in preprocessing.py -> engineer_features()
SERVICE_COLS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]


# --------------------------------------------------------------------------
# 1. LOAD ARTIFACTS
# --------------------------------------------------------------------------
def load_artifacts():
    """Load trained model, metadata, encoders, and scaler from disk."""
    for path in (MODEL_PATH, METADATA_PATH, ENCODERS_PATH, SCALER_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"'{path}' not found. Run preprocessing.py then train_model.py first."
            )

    model = joblib.load(MODEL_PATH)
    metadata = joblib.load(METADATA_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    scaler = joblib.load(SCALER_PATH)

    return model, metadata, encoders, scaler


# --------------------------------------------------------------------------
# 2. FEATURE ENGINEERING (mirrors preprocessing.py exactly)
# --------------------------------------------------------------------------
def engineer_features_single(customer: dict) -> dict:
    """Apply the same derived features used in preprocessing.py to one record."""
    row = dict(customer)  # copy

    tenure = float(row.get("tenure", 0))
    total_charges = float(row.get("TotalCharges", 0))

    # tenure_group
    bins = [0, 12, 24, 48, 60, np.inf]
    labels = ["0-12", "13-24", "25-48", "49-60", "60+"]
    for lower, upper, label in zip(bins[:-1], bins[1:], labels):
        if lower < tenure <= upper or (tenure == 0 and lower == 0):
            row["tenure_group"] = label
            break
    else:
        row["tenure_group"] = labels[-1]

    # avg_monthly_spend
    row["avg_monthly_spend"] = total_charges / (tenure if tenure != 0 else 1)

    # num_services
    row["num_services"] = sum(
        1 for col in SERVICE_COLS if str(row.get(col, "No")) == "Yes"
    )

    # has_internet
    row["has_internet"] = int(row.get("InternetService", "No") != "No")

    row.pop("customerID", None)
    return row


# --------------------------------------------------------------------------
# 3. ENCODE + SCALE A SINGLE RECORD (using saved encoders/scaler)
# --------------------------------------------------------------------------
def preprocess_input(customer: dict, metadata: dict, encoders: dict, scaler) -> pd.DataFrame:
    """
    Turn a raw customer dict into the exact feature-encoded, scaled row
    the model expects, in the exact column order used at training time.
    """
    engineered = engineer_features_single(customer)
    df = pd.DataFrame([engineered])

    # Encode categorical columns with the SAME fitted encoders from training.
    # Unseen categories fall back to the encoder's most frequent known class
    # instead of crashing, so the demo/dashboard never breaks on edge cases.
    for col, encoder in encoders.items():
        if col not in df.columns:
            continue
        value = str(df.at[0, col])
        if value in encoder.classes_:
            df[col] = encoder.transform([value])
        else:
            fallback = encoder.classes_[0]
            df[col] = encoder.transform([fallback])

    # Ensure column order matches training feature order exactly
    feature_names = metadata["feature_names"]
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0  # safe default for any missing engineered/encoded column
    df = df[feature_names]

    # Scale numerical columns with the SAME fitted scaler from training
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[num_cols] = scaler.transform(df[num_cols])

    return df


# --------------------------------------------------------------------------
# 4. RISK LEVEL + CONFIDENCE
# --------------------------------------------------------------------------
def get_risk_level(probability: float) -> str:
    if probability < RISK_THRESHOLDS["low"]:
        return "Low"
    elif probability < RISK_THRESHOLDS["medium"]:
        return "Medium"
    return "High"


def get_confidence(probability: float) -> float:
    """
    Confidence = how far the probability is from the 50/50 decision boundary,
    rescaled to 0-100%. A probability of 0.95 or 0.05 -> high confidence;
    a probability of 0.50 -> lowest confidence.
    """
    return round(abs(probability - 0.5) * 2 * 100, 2)


# --------------------------------------------------------------------------
# 5. EXPLAINABILITY - KEY CONTRIBUTING FACTORS
# --------------------------------------------------------------------------
def explain_with_shap(model, X_row: pd.DataFrame, top_n: int = 5):
    """Local, per-customer explanation using SHAP values."""
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_row)
        values = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]

        contrib = pd.DataFrame({
            "Feature": X_row.columns,
            "Impact": values,
        })
        contrib["AbsImpact"] = contrib["Impact"].abs()
        contrib = contrib.sort_values("AbsImpact", ascending=False).head(top_n)
        contrib["Direction"] = contrib["Impact"].apply(
            lambda v: "Increases churn risk" if v > 0 else "Decreases churn risk"
        )
        return contrib[["Feature", "Direction", "Impact"]].to_dict(orient="records")
    except Exception as e:
        print(f"[SHAP] Falling back to global importance - {e}")
        return None


def explain_with_global_importance(metadata: dict, top_n: int = 5):
    """Fallback: use the model's global feature importance (from train_model.py)."""
    top_features = metadata.get("top_features")
    if not top_features:
        return []
    return [
        {"Feature": f["Feature"], "Direction": "Key global churn driver", "Impact": f["Importance"]}
        for f in top_features[:top_n]
    ]


def get_key_factors(model, X_row: pd.DataFrame, metadata: dict, top_n: int = 5):
    if SHAP_AVAILABLE:
        result = explain_with_shap(model, X_row, top_n)
        if result is not None:
            return result
    return explain_with_global_importance(metadata, top_n)


# --------------------------------------------------------------------------
# 6. MAIN PREDICTION FUNCTION (used by dashboard.py / app.py)
# --------------------------------------------------------------------------
def predict_churn(customer: dict, model=None, metadata=None, encoders=None, scaler=None) -> dict:
    """
    Full prediction pipeline for ONE customer record.

    Parameters
    ----------
    customer : dict
        Raw customer fields, same names/values as the original dataset
        (e.g. {"gender": "Female", "tenure": 5, "MonthlyCharges": 70.5, ...})
    model, metadata, encoders, scaler : optional
        Pass these in if already loaded (e.g. cached in Streamlit) to avoid
        reloading from disk on every prediction.

    Returns
    -------
    dict with keys:
        churn_prediction, churn_probability, risk_level,
        prediction_confidence, key_factors
    """
    if model is None or metadata is None or encoders is None or scaler is None:
        model, metadata, encoders, scaler = load_artifacts()

    X_row = preprocess_input(customer, metadata, encoders, scaler)

    probability = float(model.predict_proba(X_row)[0][1])
    prediction = int(probability >= 0.5)
    risk_level = get_risk_level(probability)
    confidence = get_confidence(probability)
    key_factors = get_key_factors(model, X_row, metadata)

    return {
        "churn_prediction": "Yes" if prediction == 1 else "No",
        "churn_probability": round(probability * 100, 2),
        "risk_level": risk_level,
        "prediction_confidence": confidence,
        "key_factors": key_factors,
    }


# --------------------------------------------------------------------------
# 7. SIMPLE CLI FOR MANUAL TESTING
# --------------------------------------------------------------------------
def _prompt_customer_input() -> dict:
    """Interactive terminal prompts to build a sample customer record."""
    print("\nEnter customer details (press Enter to accept the [default]):\n")

    def ask(field, default):
        val = input(f"{field} [{default}]: ").strip()
        return val if val else default

    customer = {
        "gender": ask("Gender (Male/Female)", "Female"),
        "SeniorCitizen": int(ask("Senior Citizen (0/1)", "0")),
        "Partner": ask("Partner (Yes/No)", "No"),
        "Dependents": ask("Dependents (Yes/No)", "No"),
        "tenure": float(ask("Tenure (months)", "5")),
        "PhoneService": ask("Phone Service (Yes/No)", "Yes"),
        "MultipleLines": ask("Multiple Lines (Yes/No/No phone service)", "No"),
        "InternetService": ask("Internet Service (DSL/Fiber optic/No)", "Fiber optic"),
        "OnlineSecurity": ask("Online Security (Yes/No/No internet service)", "No"),
        "OnlineBackup": ask("Online Backup (Yes/No/No internet service)", "No"),
        "DeviceProtection": ask("Device Protection (Yes/No/No internet service)", "No"),
        "TechSupport": ask("Tech Support (Yes/No/No internet service)", "No"),
        "StreamingTV": ask("Streaming TV (Yes/No/No internet service)", "Yes"),
        "StreamingMovies": ask("Streaming Movies (Yes/No/No internet service)", "Yes"),
        "Contract": ask("Contract (Month-to-month/One year/Two year)", "Month-to-month"),
        "PaperlessBilling": ask("Paperless Billing (Yes/No)", "Yes"),
        "PaymentMethod": ask(
            "Payment Method (Electronic check/Mailed check/Bank transfer (automatic)/Credit card (automatic))",
            "Electronic check",
        ),
        "MonthlyCharges": float(ask("Monthly Charges", "85.0")),
        "TotalCharges": float(ask("Total Charges", "425.0")),
    }
    return customer


def _print_result(result: dict):
    print("\n" + "=" * 50)
    print("PREDICTION RESULT")
    print("=" * 50)
    print(f"Churn Prediction     : {result['churn_prediction']}")
    print(f"Churn Probability    : {result['churn_probability']}%")
    print(f"Risk Level           : {result['risk_level']}")
    print(f"Prediction Confidence: {result['prediction_confidence']}%")
    print("\nKey Contributing Factors:")
    for factor in result["key_factors"]:
        print(f"  - {factor['Feature']}: {factor['Direction']} (impact: {factor['Impact']:.4f})")
    print("=" * 50)


if __name__ == "__main__":
    print("[Loading] model, encoders, scaler ...")
    model, metadata, encoders, scaler = load_artifacts()
    print(f"[Loaded] Best model: {metadata['best_model_name']}")

    customer = _prompt_customer_input()
    result = predict_churn(customer, model, metadata, encoders, scaler)
    _print_result(result)