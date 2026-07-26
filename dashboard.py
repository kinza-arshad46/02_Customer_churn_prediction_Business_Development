"""
dashboard.py
------------
Phase 6 - Business Intelligence Dashboard
Phase 7 - Business Insights
Customer Churn Prediction & Business Intelligence System

Run with:
    streamlit run dashboard.py

Reads the human-readable processed dataset produced by preprocessing.py
(data/telco_churn_processed.csv) - i.e. BEFORE label-encoding/scaling -
so filters and charts show real category names (e.g. "Month-to-month",
not an integer code).

Sections:
    1. Sidebar Filters   - Gender, Contract, Payment Method, Internet
                            Service, Senior Citizen, Tenure Range
    2. KPI Cards         - Total/Active/Churned Customers, Churn Rate,
                            Avg Monthly Revenue, Estimated Revenue Loss
    3. Interactive Charts- Segmentation, Churn by Contract/Payment,
                            Monthly Charges dist., Tenure analysis,
                            Revenue analysis, Feature Importance,
                            Correlation Heatmap
    4. Business Insights - Auto-generated high-risk segments, churn
                            drivers, revenue at risk, recommendations,
                            executive summary
    5. Live Prediction   - Reuses predict.py so there is a single source
                            of truth for the prediction logic
"""

import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from predict import predict_churn, load_artifacts

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
PROCESSED_DATA_PATH = "data/telco_churn_processed.csv"

st.set_page_config(
    page_title="Customer Churn - BI Dashboard",
    layout="wide",
    page_icon="📊",
)


# --------------------------------------------------------------------------
# DATA LOADING (cached)
# --------------------------------------------------------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(
            f"Processed dataset not found at '{path}'. "
            "Run preprocessing.py first to generate it."
        )
        st.stop()
    return pd.read_csv(path)


@st.cache_resource
def load_model_artifacts():
    try:
        return load_artifacts()
    except FileNotFoundError:
        return None, None, None, None


df = load_data(PROCESSED_DATA_PATH)
model, metadata, encoders, scaler = load_model_artifacts()


# --------------------------------------------------------------------------
# SIDEBAR FILTERS
# --------------------------------------------------------------------------
def build_sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("🔍 Filters")

    filtered = df.copy()

    if "gender" in df.columns:
        genders = st.sidebar.multiselect(
            "Gender", options=sorted(df["gender"].unique()), default=list(df["gender"].unique())
        )
        filtered = filtered[filtered["gender"].isin(genders)]

    if "Contract" in df.columns:
        contracts = st.sidebar.multiselect(
            "Contract Type", options=sorted(df["Contract"].unique()), default=list(df["Contract"].unique())
        )
        filtered = filtered[filtered["Contract"].isin(contracts)]

    if "PaymentMethod" in df.columns:
        payments = st.sidebar.multiselect(
            "Payment Method", options=sorted(df["PaymentMethod"].unique()),
            default=list(df["PaymentMethod"].unique())
        )
        filtered = filtered[filtered["PaymentMethod"].isin(payments)]

    if "InternetService" in df.columns:
        internet = st.sidebar.multiselect(
            "Internet Service", options=sorted(df["InternetService"].unique()),
            default=list(df["InternetService"].unique())
        )
        filtered = filtered[filtered["InternetService"].isin(internet)]

    if "SeniorCitizen" in df.columns:
        senior_options = {0: "No", 1: "Yes"}
        selected_senior = st.sidebar.multiselect(
            "Senior Citizen", options=list(senior_options.values()), default=list(senior_options.values())
        )
        selected_codes = [k for k, v in senior_options.items() if v in selected_senior]
        filtered = filtered[filtered["SeniorCitizen"].isin(selected_codes)]

    if "tenure" in df.columns:
        min_t, max_t = int(df["tenure"].min()), int(df["tenure"].max())
        tenure_range = st.sidebar.slider("Tenure Range (months)", min_t, max_t, (min_t, max_t))
        filtered = filtered[filtered["tenure"].between(*tenure_range)]

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Showing **{len(filtered):,}** of **{len(df):,}** customers")
    return filtered


filtered_df = build_sidebar_filters(df)


# --------------------------------------------------------------------------
# HEADER
# --------------------------------------------------------------------------
st.title("📊 Customer Churn - Business Intelligence Dashboard")
st.caption("Data source: IBM Telco Customer Churn dataset")

tab_overview, tab_charts, tab_insights, tab_predict = st.tabs(
    ["📈 KPIs & Overview", "📊 Analytics", "💡 Business Insights", "🔮 Predict Churn"]
)


# --------------------------------------------------------------------------
# TAB 1 - KPI CARDS
# --------------------------------------------------------------------------
with tab_overview:
    total_customers = len(filtered_df)
    churned_mask = filtered_df["Churn"] == "Yes" if "Churn" in filtered_df.columns else pd.Series([False])
    churned_customers = int(churned_mask.sum())
    active_customers = total_customers - churned_customers
    churn_rate = (churned_customers / total_customers * 100) if total_customers else 0
    avg_monthly_revenue = filtered_df["MonthlyCharges"].mean() if "MonthlyCharges" in filtered_df.columns else 0
    revenue_at_risk = (
        filtered_df.loc[churned_mask, "MonthlyCharges"].sum()
        if "MonthlyCharges" in filtered_df.columns else 0
    )

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Active Customers", f"{active_customers:,}")
    col3.metric("Churned Customers", f"{churned_customers:,}")
    col4.metric("Churn Rate", f"{churn_rate:.1f}%")
    col5.metric("Avg Monthly Revenue", f"${avg_monthly_revenue:,.2f}")
    col6.metric("Est. Revenue Loss (mo.)", f"${revenue_at_risk:,.2f}")

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Churn Distribution")
        churn_counts = filtered_df["Churn"].value_counts().reset_index()
        churn_counts.columns = ["Churn", "Count"]
        fig = px.pie(churn_counts, names="Churn", values="Count", hole=0.4,
                     color="Churn", color_discrete_map={"Yes": "#EF553B", "No": "#00CC96"})
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Customer Segmentation (by Contract)")
        seg = filtered_df.groupby(["Contract", "Churn"]).size().reset_index(name="Count")
        fig = px.bar(seg, x="Contract", y="Count", color="Churn", barmode="group",
                     color_discrete_map={"Yes": "#EF553B", "No": "#00CC96"})
        st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------------------------
# TAB 2 - ANALYTICS / CHARTS
# --------------------------------------------------------------------------
with tab_charts:
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Churn by Contract Type")
        fig = px.histogram(filtered_df, x="Contract", color="Churn", barmode="group",
                            color_discrete_map={"Yes": "#EF553B", "No": "#00CC96"})
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Churn by Payment Method")
        fig = px.histogram(filtered_df, x="PaymentMethod", color="Churn", barmode="group",
                            color_discrete_map={"Yes": "#EF553B", "No": "#00CC96"})
        fig.update_xaxes(tickangle=25)
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Monthly Charges Distribution")
        fig = px.histogram(filtered_df, x="MonthlyCharges", color="Churn", nbins=40,
                            color_discrete_map={"Yes": "#EF553B", "No": "#00CC96"}, opacity=0.7)
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        st.subheader("Tenure Analysis")
        fig = px.histogram(filtered_df, x="tenure", color="Churn", nbins=30,
                            color_discrete_map={"Yes": "#EF553B", "No": "#00CC96"}, opacity=0.7)
        st.plotly_chart(fig, use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        st.subheader("Revenue Analysis (Total Charges by Churn)")
        fig = px.box(filtered_df, x="Churn", y="TotalCharges", color="Churn",
                     color_discrete_map={"Yes": "#EF553B", "No": "#00CC96"})
        st.plotly_chart(fig, use_container_width=True)

    with c6:
        st.subheader("Feature Importance (Best Model)")
        if metadata and metadata.get("top_features"):
            imp_df = pd.DataFrame(metadata["top_features"])
            fig = px.bar(imp_df.sort_values("Importance"), x="Importance", y="Feature",
                         orientation="h", color="Importance", color_continuous_scale="Teal")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Train the model first (train_model.py) to see feature importance here.")

    st.subheader("Correlation Heatmap")
    numeric_df = filtered_df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        corr = numeric_df.corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale="RdBu", zmid=0,
        ))
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------------------------
# TAB 3 - BUSINESS INSIGHTS (Phase 7 - auto-generated)
# --------------------------------------------------------------------------
def generate_business_insights(df: pd.DataFrame) -> dict:
    insights = {}

    if "Churn" not in df.columns or df.empty:
        return insights

    churned = df[df["Churn"] == "Yes"]
    total = len(df)
    churn_rate = len(churned) / total * 100 if total else 0

    if "Contract" in df.columns:
        contract_risk = (
            df.groupby("Contract")["Churn"].apply(lambda s: (s == "Yes").mean() * 100)
            .sort_values(ascending=False)
        )
        insights["high_risk_contract"] = contract_risk

    if "PaymentMethod" in df.columns:
        payment_risk = (
            df.groupby("PaymentMethod")["Churn"].apply(lambda s: (s == "Yes").mean() * 100)
            .sort_values(ascending=False)
        )
        insights["high_risk_payment"] = payment_risk

    if "MonthlyCharges" in df.columns:
        insights["revenue_at_risk_monthly"] = churned["MonthlyCharges"].sum()
        insights["revenue_at_risk_annual"] = churned["MonthlyCharges"].sum() * 12

    insights["overall_churn_rate"] = churn_rate
    insights["total_customers"] = total
    insights["churned_customers"] = len(churned)

    return insights


with tab_insights:
    insights = generate_business_insights(filtered_df)

    if not insights:
        st.info("No data available for the current filter selection.")
    else:
        st.subheader("📌 Executive Summary")
        st.markdown(
            f"""
            Out of **{insights['total_customers']:,}** customers in the current view,
            **{insights['churned_customers']:,}** have churned
            (**{insights['overall_churn_rate']:.1f}%** churn rate),
            representing an estimated **${insights.get('revenue_at_risk_monthly', 0):,.2f}/month**
            (**${insights.get('revenue_at_risk_annual', 0):,.2f}/year**) in revenue at risk.
            """
        )

        st.markdown("---")
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("🚩 Highest-Risk Contract Types")
            if "high_risk_contract" in insights:
                for contract, rate in insights["high_risk_contract"].items():
                    st.write(f"- **{contract}**: {rate:.1f}% churn rate")

        with c2:
            st.subheader("🚩 Highest-Risk Payment Methods")
            if "high_risk_payment" in insights:
                for method, rate in insights["high_risk_payment"].items():
                    st.write(f"- **{method}**: {rate:.1f}% churn rate")

        st.markdown("---")
        st.subheader("🎯 Major Churn Drivers")
        if metadata and metadata.get("top_features"):
            for f in metadata["top_features"][:5]:
                st.write(f"- **{f['Feature']}** (importance score: {f['Importance']:.4f})")
        else:
            st.info("Train the model first to populate churn drivers here.")

        st.markdown("---")
        st.subheader("✅ Retention Recommendations")
        st.markdown(
            """
            - Prioritize retention outreach for **month-to-month contract** customers -
              offer incentives to upgrade to annual plans.
            - Review pricing/experience for customers paying via **electronic check**,
              which typically shows elevated churn.
            - Bundle **tech support / online security add-ons** for fiber-optic customers,
              since low service engagement correlates with higher churn.
            - Set up an early-warning trigger for customers in their **first 12 months**
              of tenure, the highest-risk window.
            - Use the **Predict Churn** tab to proactively flag high-risk accounts for
              the retention team before they cancel.
            """
        )


# --------------------------------------------------------------------------
# TAB 4 - LIVE PREDICTION (Phase 5, reusing predict.py)
# --------------------------------------------------------------------------
with tab_predict:
    st.subheader("🔮 Predict Churn for a Customer")

    if model is None:
        st.warning("No trained model found. Run preprocessing.py then train_model.py first.")
    else:
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

            st.markdown("---")
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Churn Prediction", result["churn_prediction"])
            r2.metric("Churn Probability", f"{result['churn_probability']}%")
            risk_color = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}[result["risk_level"]]
            r3.metric("Risk Level", f"{risk_color} {result['risk_level']}")
            r4.metric("Confidence", f"{result['prediction_confidence']}%")

            st.subheader("Key Contributing Factors")
            factors_df = pd.DataFrame(result["key_factors"])
            st.dataframe(factors_df, use_container_width=True)