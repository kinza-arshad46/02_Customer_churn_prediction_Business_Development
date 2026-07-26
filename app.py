"""
app.py
------
MAIN APPLICATION ENTRY POINT
Customer Churn Prediction & Business Intelligence System

This is the single deployable Streamlit app (streamlit run app.py) that ties
together every phase of the project into one navigable experience:

    Phase 2 - Exploratory Data Analysis
    Phase 3 - Model Performance & Comparison
    Phase 4 - Model Explainability
    Phase 5 - Prediction System
    Phase 6 - Business Intelligence Dashboard
    Phase 7 - Business Insights
    Phase 8 - Downloadable Reports (PDF / CSV)

Prerequisites (run once, from the terminal, before launching this app):
    python preprocessing.py
    python train_model.py

Then launch with:
    streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from predict import predict_churn, load_artifacts
from utils import (
    PATHS,
    generate_business_insights,
    RETENTION_RECOMMENDATIONS,
    export_processed_dataset_csv,
    generate_prediction_report_pdf,
    generate_analytics_report_pdf,
    format_currency,
    setup_logger,
    REPORTLAB_AVAILABLE,
)

logger = setup_logger()

st.set_page_config(
    page_title="Customer Churn Prediction System",
    layout="wide",
    page_icon="📉",
)

CHURN_COLORS = {"Yes": "#EF553B", "No": "#00CC96"}


# ==========================================================================
# CACHED LOADERS
# ==========================================================================
@st.cache_data
def load_processed_data():
    path = PATHS["processed_data"]
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


@st.cache_data
def load_model_comparison():
    path = os.path.join(PATHS["reports_dir"], "model_comparison.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


@st.cache_resource
def load_model_bundle():
    try:
        return load_artifacts()
    except FileNotFoundError:
        return None, None, None, None


@st.cache_data
def load_html_report(filename: str):
    path = os.path.join(PATHS["reports_dir"], filename)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


df = load_processed_data()
model, metadata, encoders, scaler = load_model_bundle()


# ==========================================================================
# SIDEBAR NAVIGATION
# ==========================================================================
st.sidebar.title("📉 Churn Prediction System")
page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Home",
        "📊 Exploratory Data Analysis",
        "🤖 Model Performance",
        "🔍 Model Explainability",
        "🔮 Predict Churn",
        "📈 BI Dashboard",
        "💡 Business Insights",
        "📥 Reports & Downloads",
    ],
)

if df is None:
    st.sidebar.error("⚠️ Processed dataset missing. Run `preprocessing.py` first.")
if model is None:
    st.sidebar.error("⚠️ Trained model missing. Run `train_model.py` first.")


# ==========================================================================
# PAGE: HOME
# ==========================================================================
if page == "🏠 Home":
    st.title("📉 Customer Churn Prediction & Business Intelligence System")
    st.markdown(
        """
        A production-style ML application that predicts customer churn and
        surfaces actionable business insights, built across the full
        data-science lifecycle - preprocessing, EDA, modeling,
        explainability, prediction, and BI reporting.

        **Dataset:** IBM Telco Customer Churn (or equivalent)
        """
    )

    if df is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Records", f"{len(df):,}")
        c2.metric("Features", f"{df.shape[1] - 1}")
        if "Churn" in df.columns:
            c3.metric("Churn Rate", f"{(df['Churn'] == 'Yes').mean() * 100:.1f}%")
        if model is not None and metadata is not None:
            c4.metric("Best Model", metadata.get("best_model_name", "-"))

        st.markdown("---")
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)
    else:
        st.warning("Run `python preprocessing.py` to generate the dataset used by this app.")

    st.markdown("---")
    st.markdown(
        """
        **Use the sidebar to navigate:**
        - **Exploratory Data Analysis** - understand the data (Phase 2)
        - **Model Performance** - compare Logistic Regression, Decision Tree,
          Random Forest, and XGBoost (Phase 3)
        - **Model Explainability** - see what drives churn predictions (Phase 4)
        - **Predict Churn** - score a single customer live (Phase 5)
        - **BI Dashboard** - KPIs, segmentation, filters (Phase 6)
        - **Business Insights** - auto-generated recommendations (Phase 7)
        - **Reports & Downloads** - export PDF/CSV reports (Phase 8)
        """
    )


# ==========================================================================
# PAGE: EDA
# ==========================================================================
elif page == "📊 Exploratory Data Analysis":
    st.title("📊 Exploratory Data Analysis")

    if df is None:
        st.error("Processed dataset not found. Run `preprocessing.py` first.")
        st.stop()

    st.subheader("Customer & Churn Distribution")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.pie(df, names="Churn", hole=0.4, color="Churn", color_discrete_map=CHURN_COLORS,
                     title="Overall Churn Distribution")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        if "gender" in df.columns:
            fig = px.histogram(df, x="gender", color="Churn", barmode="group",
                                color_discrete_map=CHURN_COLORS, title="Customer Distribution by Gender")
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Tenure Distribution")
    if "tenure" in df.columns:
        fig = px.histogram(df, x="tenure", color="Churn", nbins=30,
                            color_discrete_map=CHURN_COLORS, opacity=0.7,
                            title="Tenure Distribution by Churn")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Monthly Charges Analysis")
    if "MonthlyCharges" in df.columns:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(df, x="MonthlyCharges", color="Churn", nbins=40,
                                color_discrete_map=CHURN_COLORS, opacity=0.7)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.box(df, x="Churn", y="MonthlyCharges", color="Churn",
                         color_discrete_map=CHURN_COLORS)
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Contract Type Analysis")
    if "Contract" in df.columns:
        fig = px.histogram(df, x="Contract", color="Churn", barmode="group",
                            color_discrete_map=CHURN_COLORS)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Payment Method Analysis")
    if "PaymentMethod" in df.columns:
        fig = px.histogram(df, x="PaymentMethod", color="Churn", barmode="group",
                            color_discrete_map=CHURN_COLORS)
        fig.update_xaxes(tickangle=25)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Heatmap")
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        corr = numeric_df.corr()
        fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns,
                                         colorscale="RdBu", zmid=0))
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature Relationships")
    if {"tenure", "MonthlyCharges"}.issubset(df.columns):
        fig = px.scatter(df, x="tenure", y="MonthlyCharges", color="Churn",
                          color_discrete_map=CHURN_COLORS, opacity=0.6,
                          title="Tenure vs Monthly Charges")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📌 Business Insights from EDA")
    if "Churn" in df.columns:
        churn_rate = (df["Churn"] == "Yes").mean() * 100
        st.markdown(
            f"""
            - Overall churn rate in this dataset is **{churn_rate:.1f}%**.
            - Customers on **month-to-month contracts** and paying via
              **electronic check** tend to show visibly higher churn in the
              charts above.
            - Churn is concentrated in the **early tenure months**, suggesting
              onboarding/retention efforts should focus on new customers.
            """
        )


# ==========================================================================
# PAGE: MODEL PERFORMANCE
# ==========================================================================
elif page == "🤖 Model Performance":
    st.title("🤖 Model Performance & Comparison")

    comparison_df = load_model_comparison()
    if comparison_df is None:
        st.error("No model comparison report found. Run `train_model.py` first.")
        st.stop()

    st.subheader("Metrics Comparison")
    st.dataframe(
        comparison_df.style.highlight_max(
            subset=["accuracy", "precision", "recall", "f1_score", "roc_auc"],
            color="#d1fae5",
        ),
        use_container_width=True,
    )

    fig = px.bar(
        comparison_df.melt(id_vars="Model", var_name="Metric", value_name="Score"),
        x="Model", y="Score", color="Metric", barmode="group",
        title="Model Metrics Comparison",
    )
    st.plotly_chart(fig, use_container_width=True)

    if metadata:
        st.success(
            f"**Best Model Selected: {metadata['best_model_name']}** "
            f"(ROC-AUC: {metadata['metrics']['roc_auc']:.4f}) - chosen for its "
            f"strongest ability to rank customers by churn risk."
        )

    st.subheader("Confusion Matrices")
    cm_html = load_html_report("confusion_matrices.html")
    if cm_html:
        components.html(cm_html, height=450, scrolling=True)

    st.subheader("ROC Curves")
    roc_html = load_html_report("roc_curves.html")
    if roc_html:
        components.html(roc_html, height=550, scrolling=True)


# ==========================================================================
# PAGE: MODEL EXPLAINABILITY
# ==========================================================================
elif page == "🔍 Model Explainability":
    st.title("🔍 Model Explainability")

    if metadata is None:
        st.error("No trained model metadata found. Run `train_model.py` first.")
        st.stop()

    st.subheader(f"Feature Importance - {metadata['best_model_name']}")
    fi_html = load_html_report("feature_importance.html")
    if fi_html:
        components.html(fi_html, height=650, scrolling=True)

    st.subheader("Top Factors Affecting Churn")
    top_features = metadata.get("top_features") or []
    for i, f in enumerate(top_features[:10], start=1):
        st.write(f"{i}. **{f['Feature']}** — importance score: {f['Importance']:.4f}")

    st.subheader("📌 Business Interpretation")
    st.markdown(
        """
        - Features related to **contract type and tenure** typically dominate
          churn risk - customers with short tenure and flexible (month-to-month)
          contracts are the most likely to leave.
        - **Billing and payment behavior** (payment method, paperless billing,
          monthly charges) often ranks highly - friction in the billing
          experience is a common churn trigger.
        - **Service engagement** features (number of add-on services, internet
          service type) reflect how "locked in" a customer is - customers with
          more bundled services tend to churn less.
        - These patterns should guide retention strategy: target short-tenure,
          month-to-month, low-engagement customers first.
        """
    )


# ==========================================================================
# PAGE: PREDICT CHURN
# ==========================================================================
elif page == "🔮 Predict Churn":
    st.title("🔮 Predict Churn for a Customer")

    if model is None:
        st.error("No trained model found. Run `preprocessing.py` then `train_model.py` first.")
        st.stop()

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        with c2:
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        with c3:
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox(
                "Payment Method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            )
            monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)

        total_charges = st.number_input("Total Charges", min_value=0.0, value=float(tenure * monthly_charges))
        submitted = st.form_submit_button("Predict Churn")

    if submitted:
        customer = {
            "gender": gender, "SeniorCitizen": senior_citizen, "Partner": partner,
            "Dependents": dependents, "tenure": tenure, "PhoneService": phone_service,
            "MultipleLines": multiple_lines, "InternetService": internet_service,
            "OnlineSecurity": online_security, "OnlineBackup": online_backup,
            "DeviceProtection": device_protection, "TechSupport": tech_support,
            "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
            "Contract": contract, "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method, "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }
        result = predict_churn(customer, model, metadata, encoders, scaler)

        # Stash in session state so the Reports page can offer a PDF download
        st.session_state["last_customer"] = customer
        st.session_state["last_result"] = result

        st.markdown("---")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Churn Prediction", result["churn_prediction"])
        r2.metric("Churn Probability", f"{result['churn_probability']}%")
        risk_color = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}[result["risk_level"]]
        r3.metric("Risk Level", f"{risk_color} {result['risk_level']}")
        r4.metric("Confidence", f"{result['prediction_confidence']}%")

        st.subheader("Key Contributing Factors")
        st.dataframe(pd.DataFrame(result["key_factors"]), use_container_width=True)

        st.info("Go to **📥 Reports & Downloads** to export this prediction as a PDF.")


# ==========================================================================
# PAGE: BI DASHBOARD
# ==========================================================================
elif page == "📈 BI Dashboard":
    st.title("📈 Business Intelligence Dashboard")

    if df is None:
        st.error("Processed dataset not found. Run `preprocessing.py` first.")
        st.stop()

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Dashboard Filters")
    filtered = df.copy()

    if "gender" in df.columns:
        sel = st.sidebar.multiselect("Gender", sorted(df["gender"].unique()), default=list(df["gender"].unique()))
        filtered = filtered[filtered["gender"].isin(sel)]
    if "Contract" in df.columns:
        sel = st.sidebar.multiselect("Contract Type", sorted(df["Contract"].unique()), default=list(df["Contract"].unique()))
        filtered = filtered[filtered["Contract"].isin(sel)]
    if "PaymentMethod" in df.columns:
        sel = st.sidebar.multiselect("Payment Method", sorted(df["PaymentMethod"].unique()), default=list(df["PaymentMethod"].unique()))
        filtered = filtered[filtered["PaymentMethod"].isin(sel)]
    if "InternetService" in df.columns:
        sel = st.sidebar.multiselect("Internet Service", sorted(df["InternetService"].unique()), default=list(df["InternetService"].unique()))
        filtered = filtered[filtered["InternetService"].isin(sel)]
    if "SeniorCitizen" in df.columns:
        opts = {0: "No", 1: "Yes"}
        sel = st.sidebar.multiselect("Senior Citizen", list(opts.values()), default=list(opts.values()))
        codes = [k for k, v in opts.items() if v in sel]
        filtered = filtered[filtered["SeniorCitizen"].isin(codes)]
    if "tenure" in df.columns:
        min_t, max_t = int(df["tenure"].min()), int(df["tenure"].max())
        rng = st.sidebar.slider("Tenure Range", min_t, max_t, (min_t, max_t))
        filtered = filtered[filtered["tenure"].between(*rng)]

    total = len(filtered)
    churned_mask = filtered["Churn"] == "Yes"
    churned = int(churned_mask.sum())
    active = total - churned
    churn_rate = (churned / total * 100) if total else 0
    avg_rev = filtered["MonthlyCharges"].mean() if "MonthlyCharges" in filtered.columns else 0
    rev_loss = filtered.loc[churned_mask, "MonthlyCharges"].sum() if "MonthlyCharges" in filtered.columns else 0

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Customers", f"{total:,}")
    c2.metric("Active Customers", f"{active:,}")
    c3.metric("Churned Customers", f"{churned:,}")
    c4.metric("Churn Rate", f"{churn_rate:.1f}%")
    c5.metric("Avg Monthly Revenue", format_currency(avg_rev))
    c6.metric("Est. Revenue Loss", format_currency(rev_loss))

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.pie(filtered, names="Churn", hole=0.4, color="Churn", color_discrete_map=CHURN_COLORS,
                     title="Churn Distribution")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        seg = filtered.groupby(["Contract", "Churn"]).size().reset_index(name="Count")
        fig = px.bar(seg, x="Contract", y="Count", color="Churn", barmode="group",
                     color_discrete_map=CHURN_COLORS, title="Segmentation by Contract")
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig = px.box(filtered, x="Churn", y="TotalCharges", color="Churn",
                     color_discrete_map=CHURN_COLORS, title="Revenue Analysis")
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        if metadata and metadata.get("top_features"):
            imp_df = pd.DataFrame(metadata["top_features"])
            fig = px.bar(imp_df.sort_values("Importance"), x="Importance", y="Feature",
                         orientation="h", title="Feature Importance")
            st.plotly_chart(fig, use_container_width=True)


# ==========================================================================
# PAGE: BUSINESS INSIGHTS
# ==========================================================================
elif page == "💡 Business Insights":
    st.title("💡 Business Insights")

    if df is None:
        st.error("Processed dataset not found. Run `preprocessing.py` first.")
        st.stop()

    insights = generate_business_insights(df)
    st.session_state["last_insights"] = insights  # for the Reports page

    st.subheader("📌 Executive Summary")
    st.markdown(
        f"""
        Out of **{insights['total_customers']:,}** customers,
        **{insights['churned_customers']:,}** have churned
        (**{insights['overall_churn_rate']:.1f}%** churn rate), representing an
        estimated **{format_currency(insights.get('revenue_at_risk_monthly', 0))}/month**
        (**{format_currency(insights.get('revenue_at_risk_annual', 0))}/year**) in revenue at risk.
        """
    )

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🚩 Highest-Risk Contract Types")
        if "high_risk_contract" in insights:
            for name, rate in insights["high_risk_contract"].items():
                st.write(f"- **{name}**: {rate:.1f}% churn rate")
    with c2:
        st.subheader("🚩 Highest-Risk Payment Methods")
        if "high_risk_payment" in insights:
            for name, rate in insights["high_risk_payment"].items():
                st.write(f"- **{name}**: {rate:.1f}% churn rate")

    st.markdown("---")
    st.subheader("🎯 Major Churn Drivers")
    if metadata and metadata.get("top_features"):
        for f in metadata["top_features"][:5]:
            st.write(f"- **{f['Feature']}** (importance score: {f['Importance']:.4f})")

    st.markdown("---")
    st.subheader("✅ Retention Recommendations")
    for rec in RETENTION_RECOMMENDATIONS:
        st.write(f"- {rec}")


# ==========================================================================
# PAGE: REPORTS & DOWNLOADS (Phase 8)
# ==========================================================================
elif page == "📥 Reports & Downloads":
    st.title("📥 Reports & Downloads")

    if not REPORTLAB_AVAILABLE:
        st.warning(
            "`reportlab` is not installed, so PDF export is disabled. "
            "Install it with: `pip install reportlab`"
        )

    st.subheader("1. Processed Dataset (CSV)")
    if df is not None:
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Processed Dataset (CSV)", data=csv_bytes,
            file_name="processed_dataset.csv", mime="text/csv",
        )
    else:
        st.info("Run `preprocessing.py` first to generate the processed dataset.")

    st.markdown("---")
    st.subheader("2. Analytics Report (PDF)")
    st.caption("Executive summary, high-risk segments, churn drivers, and recommendations.")
    if df is not None and REPORTLAB_AVAILABLE:
        if st.button("Generate Analytics Report"):
            insights = st.session_state.get("last_insights") or generate_business_insights(df)
            path = generate_analytics_report_pdf(insights, metadata)
            with open(path, "rb") as f:
                st.download_button(
                    "⬇️ Download Analytics Report (PDF)", data=f.read(),
                    file_name=os.path.basename(path), mime="application/pdf",
                )
    elif df is None:
        st.info("Run `preprocessing.py` first.")

    st.markdown("---")
    st.subheader("3. Prediction Report (PDF)")
    st.caption("Generate a prediction first on the **🔮 Predict Churn** page.")
    if REPORTLAB_AVAILABLE and st.session_state.get("last_result"):
        if st.button("Generate Prediction Report"):
            path = generate_prediction_report_pdf(
                st.session_state["last_customer"], st.session_state["last_result"]
            )
            with open(path, "rb") as f:
                st.download_button(
                    "⬇️ Download Prediction Report (PDF)", data=f.read(),
                    file_name=os.path.basename(path), mime="application/pdf",
                )
    elif not st.session_state.get("last_result"):
        st.info("No prediction made yet this session.")