"""Streamlit interface for prioritizing wellness-package outreach."""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_model.joblib"

st.set_page_config(page_title="Wellness Package Interest", page_icon="✈️", layout="centered")
st.title("Wellness Package Interest Predictor")
st.write(
    "Enter a customer profile to estimate the likelihood of purchasing the Wellness Tourism Package. "
    "Use the estimate to help prioritize outreach; a person should review any final decision."
)

@st.cache_resource
def load_model():
    """Load the committed preprocessing-and-model pipeline once per app session."""
    return joblib.load(MODEL_PATH)

if not MODEL_PATH.exists():
    st.error("The trained model is not available yet. Run the GitHub Actions pipeline and commit its model artifact.")
    st.stop()

with st.form("customer_profile"):
    left, right = st.columns(2)
    with left:
        age = st.number_input("Age", min_value=18, max_value=100, value=37)
        city_tier = st.selectbox("City tier", [1, 2, 3])
        occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
        gender = st.selectbox("Gender", ["Female", "Male"])
        marital_status = st.selectbox("Marital status", ["Married", "Divorced", "Single", "Unmarried"])
        monthly_income = st.number_input("Monthly income", min_value=0, max_value=200000, value=23000, step=500)
        number_of_trips = st.number_input("Trips per year", min_value=0, max_value=30, value=3)
    with right:
        people = st.selectbox("People travelling", [1, 2, 3, 4, 5], index=2)
        children = st.selectbox("Children visiting", [0, 1, 2, 3])
        property_star = st.selectbox("Preferred property rating", [3, 4, 5], index=0)
        passport = st.selectbox("Has a passport", ["No", "Yes"])
        own_car = st.selectbox("Owns a car", ["No", "Yes"])
        designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    submitted = st.form_submit_button("Estimate purchase likelihood", type="primary")

if submitted:
    # Match the category normalization used during training.
    occupation_value = "Freelancer" if occupation == "Free Lancer" else occupation
    profile = pd.DataFrame([{
        "Age": age,
        "CityTier": city_tier,
        "Occupation": occupation_value,
        "Gender": gender.lower(),
        "NumberOfPersonVisiting": people,
        "PreferredPropertyStar": property_star,
        "MaritalStatus": marital_status,
        "NumberOfTrips": number_of_trips,
        "Passport": int(passport == "Yes"),
        "OwnCar": int(own_car == "Yes"),
        "NumberOfChildrenVisiting": children,
        "Designation": designation,
        "MonthlyIncome": monthly_income,
    }])
    model = load_model()
    probability = float(model.predict_proba(profile)[0, 1])
    prediction = int(probability >= 0.5)
    st.metric("Estimated purchase likelihood", f"{probability:.1%}")
    if prediction:
        st.success("This profile is above the model’s current 0.5 classification threshold.")
    else:
        st.info("This profile is below the model’s current 0.5 classification threshold.")
    st.caption(
        "The estimate is based on historical data and is not a guarantee of customer behavior. "
        "The threshold is a starting point and should be reviewed against campaign capacity and business costs."
    )

st.divider()
st.caption("Model inputs are limited to customer profile fields intended to be available before a sales contact.")
