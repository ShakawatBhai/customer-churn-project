import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Syne:wght@600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'Syne', sans-serif;
    }

    .main {
        padding-top: 1rem;
    }

    .metric-card {
        background: #1a1d24;
        border: 1px solid #2d323d;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        margin-bottom: 10px;
    }

    .metric-value {
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .metric-label {
        font-size: 13px;
        color: #9ca3af;
    }

    .risk-high {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.5);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }

    .risk-medium {
        background: rgba(245, 158, 11, 0.15);
        border: 1px solid rgba(245, 158, 11, 0.5);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }

    .risk-low {
        background: rgba(34, 197, 94, 0.15);
        border: 1px solid rgba(34, 197, 94, 0.5);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }

    .section-title {
        font-size: 22px;
        font-weight: 700;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    .info-box {
        background: #1a1d24;
        border: 1px solid #2d323d;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# MODEL LOADING
# =========================================================

@st.cache_resource
def load_model():
    try:
        model_path = Path(__file__).resolve().parent / "churn_model_pipeline.pkl"
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Unable to load model: {e}")
        return None


pipeline = load_model()


# =========================================================
# TRAINING FEATURES
# =========================================================

TRAINING_COLS = [
    "age",
    "gender",
    "region_category",
    "membership_category",
    "joined_through_referral",
    "preferred_offer_types",
    "medium_of_operation",
    "internet_option",
    "days_since_last_login",
    "avg_time_spent",
    "avg_transaction_value",
    "avg_frequency_login_days",
    "points_in_wallet",
    "used_special_discount",
    "offer_application_preference",
    "past_complaint",
    "complaint_status",
    "feedback"
]

CANONICAL_CATEGORICAL_VALUES = {
    "gender": {"Male": "M", "Female": "F", "M": "M", "F": "F", "Unknown": "Unknown"},
    "region_category": {"Town": "Town", "City": "City", "Village": "Village", "Unknown": np.nan},
    "membership_category": {
        "Basic Membership": "Basic Membership",
        "No Membership": "No Membership",
        "Gold Membership": "Gold Membership",
        "Silver Membership": "Silver Membership",
        "Premium Membership": "Premium Membership",
        "Platinum Membership": "Platinum Membership",
    },
    "joined_through_referral": {"Yes": "Yes", "No": "No", "Unknown": np.nan},
    "preferred_offer_types": {
        "Gift Vouchers/Coupons": "Gift Vouchers/Coupons",
        "Credit/Debit Card Offers": "Credit/Debit Card Offers",
        "Without Offers": "Without Offers",
        "Gift Vouchers": "Gift Vouchers/Coupons",
        "Credit Card Offers": "Credit/Debit Card Offers",
    },
    "medium_of_operation": {"Desktop": "Desktop", "Smartphone": "Smartphone", "Both": "Both", "Unknown": np.nan},
    "internet_option": {
        "Wi-Fi": "Wi-Fi",
        "Mobile_Data": "Mobile_Data",
        "Fiber_Optic": "Fiber_Optic",
        "WiFi": "Wi-Fi",
        "Mobile Data": "Mobile_Data",
        "Fiber Optic": "Fiber_Optic",
    },
    "used_special_discount": {"Yes": "Yes", "No": "No", "Unknown": np.nan},
    "offer_application_preference": {"Yes": "Yes", "No": "No", "Unknown": np.nan},
    "past_complaint": {"Yes": "Yes", "No": "No", "Unknown": np.nan},
    "complaint_status": {
        "No Information Available": "No Information Available",
        "Not Applicable": "Not Applicable",
        "Solved": "Solved",
        "Solved in Follow-up": "Solved in Follow-up",
        "Unsolved": "Unsolved",
        "Unknown": np.nan,
    },
    "feedback": {
        "Positive": "Products always in Stock",
        "Negative": "Poor Customer Service",
        "Neutral": "Reasonable Price",
        "No reason specified": "No reason specified",
        "Poor Customer Service": "Poor Customer Service",
        "Poor Product Quality": "Poor Product Quality",
        "Poor Website": "Poor Website",
        "Products always in Stock": "Products always in Stock",
        "Quality Customer Care": "Quality Customer Care",
        "Reasonable Price": "Reasonable Price",
        "Too many ads": "Too many ads",
        "User Friendly Website": "User Friendly Website",
        "Unknown": np.nan,
    },
}


def normalize_input_value(column_name, value):
    if value is None:
        return np.nan

    if isinstance(value, (np.generic,)):
        return value.item()

    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned in {"", "?", "nan", "NaN", "None", "null"}:
            return np.nan
        lookup = CANONICAL_CATEGORICAL_VALUES.get(column_name, {})
        if cleaned in lookup:
            return lookup[cleaned]
        if cleaned.lower() in {k.lower(): v for k, v in lookup.items()}:
            return {k.lower(): v for k, v in lookup.items()}[cleaned.lower()]
        return cleaned

    return value


def normalize_dataframe(df):
    normalized = df.copy()
    for col in TRAINING_COLS:
        if col in normalized.columns:
            normalized[col] = normalized[col].map(lambda value: normalize_input_value(col, value))
    for col in ["age", "days_since_last_login", "avg_time_spent", "avg_transaction_value", "avg_frequency_login_days", "points_in_wallet"]:
        if col in normalized.columns:
            normalized[col] = pd.to_numeric(normalized[col], errors="coerce").astype(np.float64)
    return normalized[TRAINING_COLS]


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Customer Profile")

st.sidebar.markdown("### Demographics")

age = st.sidebar.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female", "Unknown"]
)

region = st.sidebar.selectbox(
    "Region Category",
    ["Town", "City", "Village"]
)


st.sidebar.markdown("### Membership")

membership = st.sidebar.selectbox(
    "Membership Category",
    [
        "Basic Membership",
        "No Membership",
        "Gold Membership",
        "Silver Membership",
        "Premium Membership",
        "Platinum Membership"
    ]
)

referral = st.sidebar.selectbox(
    "Joined Through Referral",
    ["Yes", "No"]
)

preferred_offer = st.sidebar.selectbox(
    "Preferred Offer Types",
    [
        "Gift Vouchers/Coupons",
        "Credit/Debit Card Offers",
        "Without Offers"
    ]
)


st.sidebar.markdown("### Engagement")

medium = st.sidebar.selectbox(
    "Medium of Operation",
    [
        "Desktop",
        "Smartphone",
        "Both"
    ]
)

internet = st.sidebar.selectbox(
    "Internet Option",
    [
        "Wi-Fi",
        "Mobile_Data",
        "Fiber_Optic"
    ]
)

days_since_login = st.sidebar.number_input(
    "Days Since Last Login",
    min_value=0.0,
    max_value=365.0,
    value=5.0
)

avg_frequency = st.sidebar.number_input(
    "Average Login Frequency (days)",
    min_value=0.0,
    max_value=100.0,
    value=5.0
)

avg_time = st.sidebar.number_input(
    "Average Time Spent",
    min_value=0.0,
    max_value=10000.0,
    value=300.0
)


st.sidebar.markdown("### Financial")

avg_transaction = st.sidebar.number_input(
    "Average Transaction Value",
    min_value=0.0,
    max_value=1000000.0,
    value=5000.0
)

points = st.sidebar.number_input(
    "Points in Wallet",
    min_value=0.0,
    max_value=100000.0,
    value=700.0
)

discount = st.sidebar.selectbox(
    "Used Special Discount",
    ["Yes", "No"]
)

offer_application = st.sidebar.selectbox(
    "Offer Application Preference",
    ["Yes", "No"]
)


st.sidebar.markdown("### Complaints")

past_complaint = st.sidebar.selectbox(
    "Past Complaint",
    ["Yes", "No"]
)

complaint_status = st.sidebar.selectbox(
    "Complaint Status",
    [
        "Not Applicable",
        "Solved",
        "Unsolved"
    ]
)

feedback = st.sidebar.selectbox(
    "Customer Feedback",
    [
        "Products always in Stock",
        "Quality Customer Care",
        "Reasonable Price",
        "No reason specified",
        "Poor Customer Service",
        "Poor Product Quality",
        "Poor Website",
        "Too many ads",
        "User Friendly Website"
    ]
)


# =========================================================
# HEADER
# =========================================================

st.title("Customer Churn Intelligence")

st.markdown(
    "Predict customer churn risk using a machine-learning classification pipeline."
)

st.caption(
    "XGBoost model • 36,992 customer records • 18 predictive features"
)


# =========================================================
# MODEL STATUS
# =========================================================

if pipeline is None:

    st.error(
        "Model file not found. Please make sure "
        "`churn_model_pipeline.pkl` is present in the project folder."
    )

    st.stop()


# =========================================================
# MODEL PERFORMANCE
# =========================================================

st.markdown("### Model Performance")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">94.01%</div>
        <div class="metric-label">F1 Score</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">97.60%</div>
        <div class="metric-label">ROC-AUC</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">93.47%</div>
        <div class="metric-label">Accuracy</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">36,992</div>
        <div class="metric-label">Customer Records</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">18</div>
        <div class="metric-label">Features</div>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# PREDICTION BUTTON
# =========================================================

st.markdown("### Individual Prediction")

predict_button = st.button(
    "🔮 Predict Churn Risk",
    type="primary",
    use_container_width=True
)


# =========================================================
# SINGLE CUSTOMER PREDICTION
# =========================================================

if predict_button:

    input_data = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "region_category": region,
        "membership_category": membership,
        "joined_through_referral": referral,
        "preferred_offer_types": preferred_offer,
        "medium_of_operation": medium,
        "internet_option": internet,
        "days_since_last_login": days_since_login,
        "avg_time_spent": avg_time,
        "avg_transaction_value": avg_transaction,
        "avg_frequency_login_days": avg_frequency,
        "points_in_wallet": points,
        "used_special_discount": discount,
        "offer_application_preference": offer_application,
        "past_complaint": past_complaint,
        "complaint_status": complaint_status,
        "feedback": feedback
    }])

    input_data = normalize_dataframe(input_data)

    try:

        prediction = pipeline.predict(input_data)[0]

        probability = float(pipeline.predict_proba(input_data)[0][1])

        probability_percent = probability * 100

        # Risk classification
        if probability_percent >= 70:
            risk_level = "High"
            risk_class = "risk-high"
        elif probability_percent >= 40:
            risk_level = "Medium"
            risk_class = "risk-medium"
        else:
            risk_level = "Low"
            risk_class = "risk-low"

        # =================================================
        # RESULT
        # =================================================

        st.markdown("### Prediction Result")

        result_col1, result_col2 = st.columns([1, 1])

        with result_col1:

            st.markdown(
                f"""
                <div class="{risk_class}">
                    <h2>{risk_level} Risk</h2>
                    <h1>{probability_percent:.1f}%</h1>
                    <p>Estimated Churn Probability</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        with result_col2:

            st.metric(
                "Prediction",
                "Likely to Churn" if prediction == 1 else "Likely to Stay"
            )

            st.progress(
                min(max(probability, 0.0), 1.0)
            )

            if risk_level == "High":
                st.warning(
                    "This customer shows a high estimated probability of churn. "
                    "Consider proactive retention actions."
                )

            elif risk_level == "Medium":
                st.info(
                    "This customer shows a moderate estimated probability of churn. "
                    "Monitoring and targeted engagement may be useful."
                )

            else:
                st.success(
                    "This customer shows a relatively low estimated probability of churn."
                )


        # =================================================
        # GLOBAL CHURN DRIVERS
        # =================================================

        st.markdown("### Key Churn Drivers")

        driver_data = pd.DataFrame({
            "Feature": [
                "Points in Wallet",
                "Membership Category",
                "Customer Feedback",
                "Average Transaction Value",
                "Medium of Operation"
            ],
            "Importance": [
                0.4607,
                0.3552,
                0.0401,
                0.0301,
                0.0091
            ]
        })

        driver_data = driver_data.sort_values(
            "Importance",
            ascending=True
        )

        st.bar_chart(
            driver_data.set_index("Feature")
        )

        st.caption(
            "These are global model-level churn drivers, not individual "
            "SHAP values for this specific customer."
        )


        # =================================================
        # BUSINESS RECOMMENDATION
        # =================================================

        st.markdown("### Recommended Action")

        if risk_level == "High":

            st.error(
                """
                **Priority retention recommended**

                • Contact the customer proactively  
                • Consider a personalized retention offer  
                • Review recent complaints and feedback  
                • Encourage platform engagement  
                • Monitor the customer closely
                """
            )

        elif risk_level == "Medium":

            st.warning(
                """
                **Targeted engagement recommended**

                • Monitor recent activity  
                • Send personalized offers  
                • Encourage regular platform usage  
                • Address unresolved complaints
                """
            )

        else:

            st.success(
                """
                **Maintain engagement**

                • Continue personalized communication  
                • Reward regular activity  
                • Maintain customer satisfaction  
                • Monitor for future behavioral changes
                """
            )


        # =================================================
        # CUSTOMER PROFILE
        # =================================================

        st.markdown("### Submitted Customer Profile")

        profile_display = input_data.T.reset_index()

        profile_display.columns = ["Feature", "Value"]

        st.dataframe(
            profile_display,
            use_container_width=True,
            hide_index=True
        )

    except Exception as e:

        st.error(
            f"Prediction failed: {e}"
        )

        st.info(
            "Please verify that the input categories and package versions "
            "match those used when the model pipeline was trained."
        )


# =========================================================
# BATCH PREDICTION
# =========================================================

st.markdown("---")

st.markdown("### Batch Prediction")

st.write(
    "Upload a CSV file containing the required customer features "
    "to generate churn predictions for multiple customers."
)

uploaded_file = st.file_uploader(
    "Upload Customer CSV",
    type=["csv"]
)


if uploaded_file is not None:

    try:

        batch_df = pd.read_csv(uploaded_file)

        missing_cols = [
            col for col in TRAINING_COLS
            if col not in batch_df.columns
        ]

        if missing_cols:

            st.error(
                "The uploaded CSV is missing the following required columns:"
            )

            st.write(missing_cols)

        else:

            prediction_input = normalize_dataframe(batch_df)

            batch_predictions = pipeline.predict(
                prediction_input
            )

            batch_probabilities = pipeline.predict_proba(
                prediction_input
            )[:, 1].astype(float)

            batch_probability_percent = (
                batch_probabilities * 100
            )

            risk_levels = np.where(
                batch_probability_percent >= 70,
                "High",
                np.where(
                    batch_probability_percent >= 40,
                    "Medium",
                    "Low"
                )
            )

            result_df = batch_df.copy()

            result_df["Churn Probability (%)"] = (
                batch_probability_percent.round(2)
            )

            result_df["Prediction"] = np.where(
                batch_predictions == 1,
                "Likely to Churn",
                "Likely to Stay"
            )

            result_df["Risk Level"] = risk_levels


            # =================================================
            # BATCH SUMMARY
            # =================================================

            st.markdown("### Batch Summary")

            batch_col1, batch_col2, batch_col3, batch_col4 = st.columns(4)

            with batch_col1:
                st.metric(
                    "Total Customers",
                    len(result_df)
                )

            with batch_col2:
                st.metric(
                    "Predicted Churn",
                    int((batch_predictions == 1).sum())
                )

            with batch_col3:
                st.metric(
                    "High Risk",
                    int((risk_levels == "High").sum())
                )

            with batch_col4:
                st.metric(
                    "Average Churn Probability",
                    f"{batch_probability_percent.mean():.1f}%"
                )


            # =================================================
            # RESULT TABLE
            # =================================================

            st.markdown("### Prediction Results")

            st.dataframe(
                result_df,
                use_container_width=True,
                hide_index=True
            )


            # =================================================
            # DOWNLOAD
            # =================================================

            csv_data = result_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="⬇️ Download Predictions CSV",
                data=csv_data,
                file_name="customer_churn_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )

    except Exception as e:

        st.error(
            f"Unable to process the uploaded CSV: {e}"
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "Customer Churn Prediction • Machine Learning & Streamlit"
)