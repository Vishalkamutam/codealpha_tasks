import os
import joblib
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

MODEL_PATH = "models/credit_model.joblib"
DATA_PATH = "data/raw/german_credit_data.csv"

def train_model():
    df = pd.read_csv(DATA_PATH)
    X = df.drop("credit_risk", axis=1)
    y = df["credit_risk"]

    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return model

def load_model():
    if not os.path.exists(MODEL_PATH):
        return train_model()
    return joblib.load(MODEL_PATH)

st.set_page_config(page_title="Credit Scoring System", page_icon="💳", layout="wide")

st.title("Credit Scoring & Risk Prediction System")
st.write("Predict applicant credit score and loan eligibility using Machine Learning.")

model = load_model()

col1, col2 = st.columns([1.1, 0.9])

with col1:
    st.subheader("Applicant Details")
    with st.form("loan_form"):
        f1, f2 = st.columns(2)

        with f1:
            status_opts = {
                "Healthy (₹50,000+ balance)": "... >= 200 DM / salary for at least 1 year",
                "Moderate (₹10,000 - ₹50,000)": "0 <= ... < 200 DM",
                "Low (< ₹10,000)": "... < 100 DM",
                "No Checking Account": "no checking account"
            }
            status_sel = st.selectbox("Account Balance", list(status_opts.keys()))

            amount_inr = st.number_input("Loan Amount (₹)", min_value=25000, max_value=2000000, value=300000, step=25000)
            duration_val = st.number_input("Loan Duration (Months)", min_value=4, max_value=72, value=18, step=1)
            
            emi = int(amount_inr / max(1, duration_val))
            st.info(f"Estimated Monthly EMI: ₹{emi:,}/month")

            history_opts = {
                "Good (Loans paid on time)": "existing credits paid back duly till now",
                "Excellent (All past loans cleared)": "all credits at this bank paid back duly",
                "First-time Borrower": "no credits taken/all credits paid back duly",
                "Past Delays": "delay in paying off in the past",
                "Critical (Past Defaults)": "critical account/other credits existing"
            }
            hist_sel = st.selectbox("Credit History", list(history_opts.keys()))

            purpose_val = st.selectbox("Loan Purpose", [
                "car (new)", "car (used)", "furniture/equipment", "radio/television",
                "education", "business", "repairs", "others"
            ])

            savings_opts = {
                "High (₹1,00,000+)": "... >= 1000 DM",
                "Substantial (₹50,000 - ₹1,00,000)": "500 <= ... < 1000 DM",
                "Moderate (₹10,000 - ₹50,000)": "100 <= ... < 500 DM",
                "Low (< ₹10,000)": "... < 100 DM",
                "No Savings Account": "unknown/no savings account"
            }
            sav_sel = st.selectbox("Savings Reserve", list(savings_opts.keys()))

        with f2:
            inst_opts = {"Low (< 20%)": 1, "Moderate (20% - 35%)": 2, "High (35% - 50%)": 3, "Very High (> 50%)": 4}
            inst_sel = st.selectbox("Monthly Installment Rate (% of Income)", list(inst_opts.keys()), index=1)

            emp_opts = {
                "7+ Years": "... >= 7 years",
                "4 to 7 Years": "4 <= ... < 7 years",
                "1 to 4 Years": "1 <= ... < 4 years",
                "< 1 Year": "... < 1 year",
                "Unemployed": "unemployed"
            }
            emp_sel = st.selectbox("Job Experience", list(emp_opts.keys()), index=0)

            age_val = st.number_input("Applicant Age", min_value=18, max_value=85, value=30, step=1)
            housing_val = st.selectbox("Housing Type", ["own", "rent", "for free"])
            prop_val = st.selectbox("Collateral / Property", [
                "real estate", "building society savings agreement/life insurance", "car or other", "unknown/no property"
            ])
            credits_val = st.number_input("Existing Bank Credits", min_value=1, max_value=10, value=1)

        submitted = st.form_submit_button("Predict Credit Score", use_container_width=True)

with col2:
    st.subheader("Prediction Result")
    
    input_data = {
        "status": status_opts[status_sel],
        "duration": duration_val,
        "credit_history": history_opts[hist_sel],
        "purpose": purpose_val,
        "amount": float(amount_inr) / 100.0,
        "savings": savings_opts[sav_sel],
        "employment_duration": emp_opts[emp_sel],
        "installment_rate": inst_opts[inst_sel],
        "personal_status_sex": "male : single",
        "other_debtors": "none",
        "present_residence": 3,
        "property": prop_val,
        "age": age_val,
        "other_installment_plans": "none",
        "housing": housing_val,
        "number_credits": credits_val,
        "job": "skilled employee/official",
        "people_liable": 1,
        "telephone": "yes",
        "foreign_worker": "yes"
    }

    input_df = pd.DataFrame([input_data])
    prob_good = float(model.predict_proba(input_df)[0, 1])
    score = int(round(300 + (prob_good * 550)))
    score = max(300, min(850, score))

    if score >= 750:
        tier = "Prime"
        rec = "Approved (Low Risk)"
    elif score >= 670:
        tier = "Near Prime"
        rec = "Approved (Moderate Risk)"
    elif score >= 580:
        tier = "Subprime"
        rec = "Conditional Approval"
    else:
        tier = "Deep Subprime"
        rec = "Declined (High Risk)"

    st.metric(label="Credit Score (300 - 850)", value=score, delta=tier)
    st.write(f"**Decision:** {rec}")
    st.write(f"**Repayment Probability:** {round(prob_good * 100, 1)}%")
    st.progress(prob_good)

    if score >= 670:
        st.success("Good credit profile and stable income history.")
    elif score >= 580:
        st.warning("Moderate risk profile. Additional guarantor may be required.")
    else:
        st.error("High risk of default based on financial history.")

st.write("---")
with st.expander("Key Mathematical & Machine Learning Formulas"):
    st.markdown(r"""
    * **1. Monthly Installment (EMI)**:
      $$\text{EMI (₹)} = \frac{\text{Loan Amount}}{\text{Duration in Months}}$$
    * **2. Credit Score Calculation**:
      $$\text{Credit Score} = 300 + (\text{Repayment Probability} \times 550)$$
    * **3. Probability Calculation**:
      $$P(\text{Good Credit}) = \text{Random Forest Probability Output}$$
    """)
