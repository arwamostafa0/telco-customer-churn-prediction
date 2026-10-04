import streamlit as st
import pandas as pd
import joblib

# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Telco Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# Load Model & Preprocessor
# ============================================================
@st.cache_resource
def load_data_and_models():
    model = joblib.load("final_model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    return model, preprocessor
model, preprocessor = load_data_and_models()

# Threshold selected using validation data
THRESHOLD = 0.40




# ============================================================
# Custom Styling
# Works with both Light and Dark Mode
# ============================================================

st.markdown("""
<style>

    /* Main content */
    .main {
        padding-top: 2rem;
    }

    /* Main title */
    h1 {
        font-size: 2.5rem;
        font-weight: 700;
    }

    /* Section headers */
    h2 {
        margin-top: 1.5rem;
    }

    /* Prediction button */
    .stButton > button {
        width: 100%;
        height: 3rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 10px;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background-color: var(--secondary-background-color);
        padding: 15px;
        border-radius: 12px;
        border: 1px solid var(--border-color);
    }

    /* Metric labels */
    div[data-testid="stMetric"] label {
        color: var(--text-color) !important;
    }

    /* Metric values */
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--text-color) !important;
    }

    /* Metric delta */
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: var(--text-color) !important;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# Header
# ============================================================

st.title("📊 Telco Customer Churn Prediction")

st.markdown(
    """
    **Machine Learning-powered customer churn prediction**

    Enter the customer's information below to estimate their
    churn risk and probability.
    """
)

st.divider()


# ============================================================
# Customer Information
# ============================================================

st.header("👤 Customer Information")

col1, col2 = st.columns(2)


with col1:

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )


with col2:

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=72,
        value=1
    )


# ============================================================
# Services
# ============================================================

st.divider()

st.header("📱 Services")

col1, col2 = st.columns(2)


with col1:

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )


with col2:

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly_charges = st.number_input(
        "Monthly Charges ($)",
        min_value=0.0,
        max_value=200.0,
        value=50.0
    )

    total_charges = st.number_input(
        "Total Charges ($)",
        min_value=0.0,
        value=50.0
    )


# ============================================================
# Prediction Button
# ============================================================

st.divider()

if st.button(
    "🔍 Predict Churn",
    use_container_width=True
):

    # ========================================================
    # Create Customer DataFrame
    # ========================================================

    customer_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })


    # ========================================================
    # Preprocessing
    # ========================================================

    customer_processed = preprocessor.transform(
        customer_data
    )


    # ========================================================
    # Churn Probability
    # ========================================================

    churn_probability = model.predict_proba(
        customer_processed
    )[0, 1]


    # ========================================================
    # Apply Decision Threshold
    # ========================================================

    prediction = int(
        churn_probability >= THRESHOLD
    )


    # ========================================================
    # Customer Summary
    # ========================================================

    st.divider()

    st.subheader("👤 Customer Summary")

    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)


    with summary_col1:

        st.metric(
            "Tenure",
            f"{tenure} months"
        )


    with summary_col2:

        st.metric(
            "Monthly Charges",
            f"${monthly_charges:.2f}"
        )


    with summary_col3:

        st.metric(
            "Contract",
            contract
        )


    with summary_col4:

        st.metric(
            "Internet Service",
            internet_service
        )


    # ========================================================
    # Prediction Result
    # ========================================================

    st.divider()

    st.subheader("🎯 Prediction Result")


    # Probability progress bar

    st.progress(
        churn_probability
    )


    result_col1, result_col2 = st.columns(2)


    with result_col1:

        st.metric(
            "Churn Probability",
            f"{churn_probability:.1%}"
        )


    with result_col2:

        st.metric(
            "Decision Threshold",
            f"{THRESHOLD:.0%}"
        )


    # ========================================================
    # Result Message
    # ========================================================

    if prediction == 1:

        st.error(
            "🔴 **High Churn Risk**\n\n"
            "This customer is likely to churn "
            "based on the model prediction."
        )

    else:

        st.success(
            "🟢 **Low Churn Risk**\n\n"
            "This customer is unlikely to churn "
            "based on the model prediction."
        )


    # ========================================================
    # Customer Risk Profile
    # ========================================================

    st.divider()

    st.subheader("🔎 Customer Risk Profile")

    profile_col1, profile_col2 = st.columns(2)


    with profile_col1:

        st.markdown("### Contract & Tenure")

        st.write(f"**Contract:** {contract}")

        st.write(
            f"**Tenure:** {tenure} months"
        )

        st.write(
            f"**Monthly Charges:** ${monthly_charges:.2f}"
        )


    with profile_col2:

        st.markdown("### Services & Billing")

        st.write(
            f"**Internet Service:** {internet_service}"
        )

        st.write(
            f"**Payment Method:** {payment_method}"
        )

        st.write(
            f"**Paperless Billing:** {paperless_billing}"
        )
