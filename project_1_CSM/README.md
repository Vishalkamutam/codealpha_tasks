# Credit Scoring & Risk Prediction System

A Machine Learning web application built using Python, Scikit-Learn, and Streamlit to evaluate loan applicant creditworthiness and predict credit scores.

## Project Overview
This project automates the credit risk assessment process by training a Random Forest Classifier on historical applicant financial records (credit history, loan amount, duration, savings, employment, and income details). 

The application calculates the estimated monthly EMI and predicts a credit score between **300 and 850** along with a loan approval decision (Approved, Conditional Approval, or Declined).

## Technologies Used
- Python
- Pandas & NumPy
- Scikit-Learn (Random Forest, StandardScaler, OneHotEncoder)
- Streamlit (Web UI)
- Joblib (Model persistence)

## Project Structure
```
project_1_CSM/
├── app.py                # Main ML pipeline and Streamlit dashboard
├── requirements.txt      # Project dependencies
├── data/
│   └── raw/              # German Credit Dataset (CSV)
├── models/
│   └── credit_model.joblib # Saved trained model pipeline
└── README.md             # Project documentation
```

## How to Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

3. Open your browser at `http://localhost:8501`.

## How the ML Model Works
1. **Dataset**: German Credit Dataset (1,000 records).
2. **Preprocessing**: 
   - `StandardScaler` for numeric values (loan amount, duration, age, etc.).
   - `OneHotEncoder` for categorical values (credit history, purpose, housing, etc.).
3. **Model**: `RandomForestClassifier` trained with an 80/20 train-test split.
4. **Credit Scoring**: Model probability is converted to a standard credit score scale:
   - Score = $300 + (\text{Repayment Probability} \times 550)$
   - **Prime (750+)**: Low Risk - Approved
   - **Near Prime (670–749)**: Moderate Risk - Approved
   - **Subprime (580–669)**: High Risk - Conditional Approval
   - **Deep Subprime (< 580)**: Very High Risk - Declined
