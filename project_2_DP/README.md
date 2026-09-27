# Multiple Disease Prediction System using Machine Learning

A Machine Learning project developed for predicting three major health conditions: **Heart Disease**, **Diabetes**, and **Breast Cancer**. The system uses supervised machine learning classification algorithms to predict disease risk based on patient clinical parameters.

---

## 📌 Project Overview

Early detection of chronic diseases is vital in healthcare. This project implements a machine learning system that takes patient medical parameters and predicts whether a patient is at risk of:
1. **Heart Disease** (Cardiovascular assessment)
2. **Diabetes** (Metabolic and glycemic screening)
3. **Breast Cancer** (Cellular morphology and tumor classification)

The project includes an interactive web interface where users can select any of the 3 diseases, choose an ML model, test patient inputs (or click pre-loaded healthy/at-risk sample buttons), and compare model accuracies.

---

## 🎯 Key Features

- **3 Disease Modules**: Heart Disease (13 features), Diabetes (8 features), Breast Cancer (10 features).
- **5 Machine Learning Algorithms**:
  - Random Forest Classifier
  - Logistic Regression
  - Decision Tree Classifier
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
- **Standard Evaluation Metrics**: Accuracy, Precision, Recall, and F1-Score.
- **Interactive Web Interface**: Single-click testing with pre-loaded sample patient data.
- **Model Accuracy Comparison**: Side-by-side performance table and bar chart for each disease.

---

## 📂 Project Structure

```text
project_2_DP/
│
├── data/
│   ├── heart_disease.csv      # Heart disease dataset (13 features + target)
│   ├── diabetes.csv           # Diabetes dataset (8 features + outcome)
│   └── breast_cancer.csv      # Breast cancer dataset (10 features + diagnosis)
│
├── main.py                    # Main script: ML training, evaluation, and Web UI
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation & viva guide
```

---

## ⚙️ Technologies & Libraries

- **Language**: Python 3.x
- **Data Analysis**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn (`sklearn`)
  - `train_test_split`: Splitting data (80% training, 20% testing)
  - `StandardScaler`: Normalizing feature scales
  - `RandomForestClassifier`, `LogisticRegression`, `DecisionTreeClassifier`, `SVC`, `KNeighborsClassifier`
  - `accuracy_score`, `precision_score`, `recall_score`, `f1_score`
- **Frontend**: HTML5, CSS3, JavaScript, Chart.js (served via Python's built-in `http.server`)

---

## 🚀 How to Run the Project

### 1. Install Dependencies
Open your terminal or command prompt inside the project folder and run:
```bash
pip install -r requirements.txt
```

### 2. Start the Application
Run the main script:
```bash
python main.py
```

### 3. Open in Browser
The application will automatically open in your default browser at:
```text
http://127.0.0.1:5000
```
*(If it doesn't open automatically, just paste the URL in Chrome, Edge, or Firefox).*

---

## 📊 Workflow of the System

1. **Dataset Loading**: Reads CSV files using `pandas.read_csv()`.
2. **Feature & Target Separation**: Separates independent features ($X$) and target column ($y$).
3. **Data Splitting**: Uses `train_test_split()` with 80% data for training and 20% for testing.
4. **Feature Scaling**: Applies `StandardScaler()` so all features have equal weight regardless of their units (e.g., Age vs Cholesterol).
5. **Model Training**: Fits the selected classifier on the scaled training dataset.
6. **Prediction**: Takes new input values, normalizes them, and outputs:
   - Positive (Disease Detected / High Risk) OR Negative (Healthy / Normal)
   - Confidence percentage and risk probability
7. **Performance Comparison**: Evaluates and compares all 5 algorithms on the test set.

---

## 🎓 Viva & Defense Q&A (For Professor / Reviewer)

### Q1: What is this project about?
> **Answer**: It is a Multiple Disease Prediction System built in Python using Machine Learning. It predicts whether a patient is at risk of Heart Disease, Diabetes, or Breast Cancer using supervised classification models trained on medical datasets.

### Q2: Which Machine Learning algorithms did you use and why?
> **Answer**: We used 5 standard classification algorithms:
> - **Random Forest**: An ensemble of multiple decision trees that provides high accuracy and prevents overfitting.
> - **Logistic Regression**: A statistical model used for binary classification that estimates the probability of disease.
> - **Decision Tree**: A flowchart-like tree structure that splits data based on feature thresholds.
> - **SVM (Support Vector Machine)**: Finds the optimal decision boundary (hyperplane) that separates classes.
> - **KNN (K-Nearest Neighbors)**: Classifies patient data based on similarity to the nearest $k$ neighboring data points.

### Q3: Why is `StandardScaler` used?
> **Answer**: Different medical parameters have vastly different scales (e.g., Age ranges from 20-80, while Cholesterol is 150-400 mg/dL). `StandardScaler` standardizes features to have a mean of 0 and variance of 1, preventing high-magnitude features from dominating the model.

### Q4: What is the 80/20 train-test split?
> **Answer**: 80% of the dataset is used to train the machine learning models, and 20% is held out as unseen test data to evaluate the model's true accuracy.

### Q5: What is the difference between Accuracy, Precision, and Recall in healthcare?
> **Answer**:
> - **Accuracy**: The total percentage of correct predictions (both healthy and diseased).
> - **Precision**: Out of all patients predicted as having the disease, how many actually had it.
> - **Recall (Sensitivity)**: Out of all patients who actually have the disease, how many did the model correctly identify. In medical diagnostics, high Recall is critical to avoid false negatives (missing a sick patient).
> - **F1-Score**: The harmonic mean of Precision and Recall.
