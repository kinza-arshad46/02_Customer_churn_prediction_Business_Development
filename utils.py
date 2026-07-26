"""
utils.py
--------
Shared Utilities
Customer Churn Prediction & Business Intelligence System

Central place for logic that is reused across preprocessing.py,
train_model.py, predict.py, dashboard.py, and app.py, so it isn't
duplicated in every file. Also implements Phase 8 (downloadable reports):

    - Prediction Report (PDF)
    - Analytics Report (PDF)
    - Processed Dataset (CSV)
"""

import os
import logging
from datetime import datetime

import pandas as pd

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer,
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


# --------------------------------------------------------------------------
# CENTRALIZED CONFIG / PATHS
# --------------------------------------------------------------------------
PATHS = {
    "raw_data": "data/customer_churn.csv",
    "processed_data": "data/telco_churn_processed.csv",
    "train_data": "data/train.csv",
    "test_data": "data/test.csv",
    "model": "models/model.joblib",
    "metadata": "models/model_metadata.joblib",
    "encoders": "models/encoders.joblib",
    "scaler": "models/scaler.joblib",
    "reports_dir": "reports",
}

RISK_THRESHOLDS = {"low": 0.30, "medium": 0.60}
TARGET_COL = "Churn"


# --------------------------------------------------------------------------
# LOGGING
# --------------------------------------------------------------------------
def setup_logger(name: str = "churn_app", log_file: str = "reports/app.log") -> logging.Logger:
    """Create a logger that writes to both console and a log file."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:  # avoid duplicate handlers on Streamlit re-runs
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


# --------------------------------------------------------------------------
# VALIDATION HELPERS
# --------------------------------------------------------------------------
def ensure_file_exists(path: str, hint: str = ""):
    """Raise a clear, actionable error if a required artifact is missing."""
    if not os.path.exists(path):
        message = f"Required file not found: '{path}'."
        if hint:
            message += f" {hint}"
        raise FileNotFoundError(message)


def ensure_dirs(*dirs):
    """Create one or more directories if they don't already exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)


# --------------------------------------------------------------------------
# FORMATTING HELPERS
# --------------------------------------------------------------------------
def format_currency(value: float) -> str:
    return f"${value:,.2f}"


def format_percentage(value: float, decimals: int = 1) -> str:
    return f"{value:.{decimals}f}%"


# --------------------------------------------------------------------------
# RISK LEVEL / CONFIDENCE (single source of truth)
# --------------------------------------------------------------------------
def get_risk_level(probability: float) -> str:
    if probability < RISK_THRESHOLDS["low"]:
        return "Low"
    elif probability < RISK_THRESHOLDS["medium"]:
        return "Medium"
    return "High"


def get_confidence(probability: float) -> float:
    return round(abs(probability - 0.5) * 2 * 100, 2)


# --------------------------------------------------------------------------
# BUSINESS INSIGHTS (Phase 7 - single source of truth)
# --------------------------------------------------------------------------
def generate_business_insights(df: pd.DataFrame, target_col: str = TARGET_COL) -> dict:
    """
    Compute high-risk segments, revenue at risk, and summary stats from the
    (unencoded, human-readable) processed dataset. Used by dashboard.py and
    the analytics PDF report.
    """
    insights = {}
    if target_col not in df.columns or df.empty:
        return insights

    churned = df[df[target_col] == "Yes"]
    total = len(df)
    churn_rate = len(churned) / total * 100 if total else 0

    if "Contract" in df.columns:
        insights["high_risk_contract"] = (
            df.groupby("Contract")[target_col]
            .apply(lambda s: (s == "Yes").mean() * 100)
            .sort_values(ascending=False)
        )

    if "PaymentMethod" in df.columns:
        insights["high_risk_payment"] = (
            df.groupby("PaymentMethod")[target_col]
            .apply(lambda s: (s == "Yes").mean() * 100)
            .sort_values(ascending=False)
        )

    if "MonthlyCharges" in df.columns:
        insights["revenue_at_risk_monthly"] = churned["MonthlyCharges"].sum()
        insights["revenue_at_risk_annual"] = churned["MonthlyCharges"].sum() * 12

    insights["overall_churn_rate"] = churn_rate
    insights["total_customers"] = total
    insights["churned_customers"] = len(churned)

    return insights


RETENTION_RECOMMENDATIONS = [
    "Prioritize retention outreach for month-to-month contract customers - "
    "offer incentives to upgrade to annual plans.",
    "Review pricing/experience for customers paying via electronic check, "
    "which typically shows elevated churn.",
    "Bundle tech support / online security add-ons for fiber-optic customers, "
    "since low service engagement correlates with higher churn.",
    "Set up an early-warning trigger for customers in their first 12 months "
    "of tenure, the highest-risk window.",
    "Use the churn prediction tool to proactively flag high-risk accounts "
    "for the retention team before they cancel.",
]


# --------------------------------------------------------------------------
# PHASE 8 - REPORT DOWNLOADS
# --------------------------------------------------------------------------
def export_processed_dataset_csv(df: pd.DataFrame, out_path: str = None) -> str:
    """Save the processed dataset to CSV for download. Returns the file path."""
    out_path = out_path or os.path.join(PATHS["reports_dir"], "processed_dataset.csv")
    ensure_dirs(os.path.dirname(out_path))
    df.to_csv(out_path, index=False)
    return out_path


def _pdf_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle", fontSize=18, leading=22,
        spaceAfter=14, textColor=colors.HexColor("#1F2937"),
    ))
    styles.add(ParagraphStyle(
        name="SectionHeading", fontSize=13, leading=16,
        spaceBefore=12, spaceAfter=6, textColor=colors.HexColor("#0F766E"),
    ))
    return styles


def generate_prediction_report_pdf(customer: dict, result: dict, out_path: str = None) -> str:
    """
    Build a one-page PDF summarizing a single customer's churn prediction.
    Requires: pip install reportlab
    """
    if not REPORTLAB_AVAILABLE:
        raise ImportError("reportlab is required for PDF reports. Install with: pip install reportlab")

    out_path = out_path or os.path.join(
        PATHS["reports_dir"], f"prediction_report_{datetime.now():%Y%m%d_%H%M%S}.pdf"
    )
    ensure_dirs(os.path.dirname(out_path))
    styles = _pdf_styles()

    story = [
        Paragraph("Customer Churn - Prediction Report", styles["ReportTitle"]),
        Paragraph(f"Generated: {datetime.now():%Y-%m-%d %H:%M}", styles["Normal"]),
        Spacer(1, 0.5 * cm),
        Paragraph("Prediction Summary", styles["SectionHeading"]),
    ]

    summary_data = [
        ["Churn Prediction", result.get("churn_prediction", "-")],
        ["Churn Probability", f"{result.get('churn_probability', 0)}%"],
        ["Risk Level", result.get("risk_level", "-")],
        ["Prediction Confidence", f"{result.get('prediction_confidence', 0)}%"],
    ]
    summary_table = Table(summary_data, colWidths=[7 * cm, 7 * cm])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F3F4F6")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(summary_table)

    story.append(Paragraph("Key Contributing Factors", styles["SectionHeading"]))
    factors = result.get("key_factors", [])
    if factors:
        factor_data = [["Feature", "Direction", "Impact"]] + [
            [f["Feature"], f["Direction"], f"{f['Impact']:.4f}"] for f in factors
        ]
        factor_table = Table(factor_data, colWidths=[5 * cm, 6 * cm, 3 * cm])
        factor_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F766E")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
        ]))
        story.append(factor_table)

    story.append(Paragraph("Customer Details", styles["SectionHeading"]))
    detail_data = [[k, str(v)] for k, v in customer.items()]
    detail_table = Table(detail_data, colWidths=[7 * cm, 7 * cm])
    detail_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
    ]))
    story.append(detail_table)

    doc = SimpleDocTemplate(out_path, pagesize=A4)
    doc.build(story)
    return out_path


def generate_analytics_report_pdf(insights: dict, model_metadata: dict = None, out_path: str = None) -> str:
    """
    Build a business analytics / executive summary PDF from
    generate_business_insights() output. Requires: pip install reportlab
    """
    if not REPORTLAB_AVAILABLE:
        raise ImportError("reportlab is required for PDF reports. Install with: pip install reportlab")

    out_path = out_path or os.path.join(
        PATHS["reports_dir"], f"analytics_report_{datetime.now():%Y%m%d_%H%M%S}.pdf"
    )
    ensure_dirs(os.path.dirname(out_path))
    styles = _pdf_styles()

    story = [
        Paragraph("Customer Churn - Analytics Report", styles["ReportTitle"]),
        Paragraph(f"Generated: {datetime.now():%Y-%m-%d %H:%M}", styles["Normal"]),
        Spacer(1, 0.5 * cm),
        Paragraph("Executive Summary", styles["SectionHeading"]),
        Paragraph(
            f"Out of {insights.get('total_customers', 0):,} customers, "
            f"{insights.get('churned_customers', 0):,} have churned "
            f"({insights.get('overall_churn_rate', 0):.1f}% churn rate), "
            f"representing an estimated {format_currency(insights.get('revenue_at_risk_monthly', 0))}/month "
            f"({format_currency(insights.get('revenue_at_risk_annual', 0))}/year) in revenue at risk.",
            styles["Normal"],
        ),
    ]

    if "high_risk_contract" in insights:
        story.append(Paragraph("Highest-Risk Contract Types", styles["SectionHeading"]))
        data = [["Contract Type", "Churn Rate"]] + [
            [name, f"{rate:.1f}%"] for name, rate in insights["high_risk_contract"].items()
        ]
        table = Table(data, colWidths=[8 * cm, 4 * cm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F766E")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
        ]))
        story.append(table)

    if model_metadata and model_metadata.get("top_features"):
        story.append(Paragraph("Major Churn Drivers", styles["SectionHeading"]))
        data = [["Feature", "Importance"]] + [
            [f["Feature"], f"{f['Importance']:.4f}"] for f in model_metadata["top_features"][:5]
        ]
        table = Table(data, colWidths=[8 * cm, 4 * cm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F766E")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
        ]))
        story.append(table)

    story.append(Paragraph("Retention Recommendations", styles["SectionHeading"]))
    for rec in RETENTION_RECOMMENDATIONS:
        story.append(Paragraph(f"• {rec}", styles["Normal"]))
        story.append(Spacer(1, 0.15 * cm))

    doc = SimpleDocTemplate(out_path, pagesize=A4)
    doc.build(story)
    return out_path