import streamlit as st
import joblib
import pandas as pd

@st.cache_resource
def load_models():
    churn_model = joblib.load("churn_model.pkl")
    arr_model = joblib.load("future_arr_model.pkl")
    return churn_model, arr_model

churn_model, arr_model = load_models()

st.markdown("""
<style>
    /* Corporate Analytics Theme - adapts to light and dark mode */
    :root {
        --card-border: #4A90D9;
        --card-shadow: rgba(74, 144, 217, 0.1);
    }

    /* Header styling */
    h1 {
        color: var(--text-color, inherit) !important;
        border-bottom: 3px solid #4A90D9;
        padding-bottom: 0.5rem;
    }

    h2, h3 {
        color: var(--text-color, inherit) !important;
    }

    /* Slider label visibility fix for dark/light mode */
    .stSlider [data-testid="stTickBarMin"],
    .stSlider [data-testid="stTickBarMax"] {
        color: var(--text-color, inherit) !important;
    }

    .stSlider p, .stSlider label {
        color: var(--text-color, inherit) !important;
    }

    /* General text elements */
    .stSelectbox label, .stNumberInput label, .stCheckbox label,
    .stRadio label, .stSlider label {
        color: var(--text-color, inherit) !important;
    }

    /* Card-style containers */
    [data-testid="stVerticalBlock"] > div[data-testid="stExpander"] {
        border: 1px solid var(--card-border, #4A90D9);
        border-radius: 8px;
        padding: 0.5rem;
    }

    /* Metric styling */
    [data-testid="stMetricValue"] {
        color: var(--text-color, inherit) !important;
    }

    /* Button styling */
    .stButton > button {
        background-color: #1B2A4A;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 2rem;
        font-weight: 600;
    }

    .stButton > button:hover {
        background-color: #4A90D9;
        color: white;
    }

    /* Divider accent */
    hr {
        border-color: #4A90D9;
    }
</style>
""", unsafe_allow_html=True)

st.title("OptimaLife Predictive Insights Dashboard")
st.subheader("Use this tool to estimate customer churn probability or forecast future ARR.")

st.divider()

mode = st.radio("Select Prediction Mode", ["Predict Churn", "Predict Future ARR"])

st.divider()

# --- Customer Profile Section ---
with st.expander("Customer Profile", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", min_value=18, max_value=100, value=35)
        income_level = st.selectbox("Income Level", ["Low", "Medium", "High", "Very High"])
        education = st.selectbox("Education", ["High School", "Graduate", "Post-Graduate", "Other"])
    with col2:
        device_type = st.selectbox("Device Type", ["Mobile", "Desktop", "Tablet", "Other"])
        tech_comfort_score = st.slider("Tech Comfort Score", min_value=1, max_value=10, value=5)
        arr_2023 = st.number_input("ARR 2023", min_value=0.0, value=500.0)

# --- Products Section ---
with st.expander("Product Subscriptions", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        has_daily_fitness = st.checkbox("Daily Fitness")
        has_healthy_meals = st.checkbox("Healthy Meals")
    with col2:
        has_mindful_living = st.checkbox("Mindful Living")
        has_premium_health = st.checkbox("Premium Health")
    with col3:
        has_wellness_tracker = st.checkbox("Wellness Tracker")

# --- Engagement Section ---
with st.expander("Engagement Metrics", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        total_sessions_90d = st.number_input("Total Sessions (last 90 days)", min_value=0, value=10)
        total_minutes_90d = st.number_input("Total Minutes (last 90 days)", min_value=0, value=300)
    with col2:
        days_since_last_active = st.number_input("Days Since Last Active", min_value=0, value=10)
        recency_bucket = st.selectbox("Recency Bucket", ["0-7 days", "8-30 days", "31-60 days", "61-90 days", "90+ days"])

# --- Tenure Section ---
with st.expander("Tenure", expanded=True):
    tenure_days = st.number_input("Tenure (days)", min_value=0, value=365)

# Derived features
num_products = int(has_daily_fitness) + int(has_healthy_meals) + int(has_mindful_living) + int(has_premium_health) + int(has_wellness_tracker)
avg_sessions_per_day = total_sessions_90d / 90.0
minutes_per_session = total_minutes_90d / total_sessions_90d if total_sessions_90d > 0 else 0
sessions_per_product = total_sessions_90d / num_products if num_products > 0 else 0

# Build input row
input_data = pd.DataFrame([{
    "ARR_2023": arr_2023,
    "AGE": age,
    "TECH_COMFORT_SCORE": tech_comfort_score,
    "NUM_PRODUCTS": num_products,
    "HAS_DAILY_FITNESS": int(has_daily_fitness),
    "HAS_HEALTHY_MEALS": int(has_healthy_meals),
    "HAS_MINDFUL_LIVING": int(has_mindful_living),
    "HAS_PREMIUM_HEALTH": int(has_premium_health),
    "HAS_WELLNESS_TRACKER": int(has_wellness_tracker),
    "TOTAL_SESSIONS_90D": total_sessions_90d,
    "TOTAL_MINUTES_90D": total_minutes_90d,
    "AVG_SESSIONS_PER_DAY": avg_sessions_per_day,
    "DAYS_SINCE_LAST_ACTIVE": days_since_last_active,
    "TENURE_DAYS": tenure_days,
    "MINUTES_PER_SESSION": minutes_per_session,
    "SESSIONS_PER_PRODUCT": sessions_per_product,
    "INCOME_LEVEL": income_level,
    "EDUCATION": education,
    "DEVICE_TYPE": device_type,
    "RECENCY_BUCKET": recency_bucket
}])

st.divider()

# Prediction
if st.button("Run Prediction"):
    if mode == "Predict Churn":
        prob = churn_model.predict_proba(input_data)[0, 1]
        st.subheader("Churn Prediction Result")
        st.metric("Estimated Churn Probability", f"{prob:.2%}")
    elif mode == "Predict Future ARR":
        predicted_arr = arr_model.predict(input_data)[0]
        st.subheader("Future ARR Prediction Result")
        if predicted_arr > arr_2023:
            color = "#28a745"
            arrow = "&#9650;"
            label = "Growth"
        elif predicted_arr < arr_2023:
            color = "#dc3545"
            arrow = "&#9660;"
            label = "Decline"
        else:
            color = "var(--text-color, inherit)"
            arrow = "&#9644;"
            label = "No Change"
        st.markdown(
            f'<p style="font-size: 2rem; font-weight: bold; color: {color};">'
            f'{arrow} ${predicted_arr:,.2f} <span style="font-size: 1rem;">({label})</span></p>',
            unsafe_allow_html=True
        )
        st.caption(f"Compared to ARR 2023: ${arr_2023:,.2f}")
