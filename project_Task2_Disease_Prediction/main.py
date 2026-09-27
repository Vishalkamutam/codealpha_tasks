import os
import json
import webbrowser
import urllib.parse
import warnings
from http.server import HTTPServer, BaseHTTPRequestHandler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings("ignore")

DEFAULT_PORT = 5000

DISEASES = {
    "Heart Disease": {
        "file": "heart_disease.csv",
        "features": [
            {"name": "Age", "unit": "years", "desc": "Patient age in years"},
            {"name": "Gender", "unit": "1=Male, 0=Female", "desc": "Biological sex"},
            {"name": "Chest Pain Type", "unit": "0 to 3", "desc": "0: Typical, 1: Atypical, 2: Non-anginal, 3: Asymptomatic"},
            {"name": "Resting Blood Pressure", "unit": "mm Hg", "desc": "Resting systolic blood pressure"},
            {"name": "Serum Cholesterol", "unit": "mg/dL", "desc": "Total cholesterol level"},
            {"name": "Fasting Blood Sugar > 120", "unit": "1=Yes, 0=No", "desc": "Fasting sugar higher than 120 mg/dl"},
            {"name": "Resting ECG Result", "unit": "0 to 2", "desc": "0: Normal, 1: ST-T wave issue, 2: Hypertrophy"},
            {"name": "Max Heart Rate", "unit": "bpm", "desc": "Maximum heart rate achieved"},
            {"name": "Exercise Induced Angina", "unit": "1=Yes, 0=No", "desc": "Angina induced by physical exercise"},
            {"name": "ST Depression (Oldpeak)", "unit": "mm", "desc": "ST depression during exercise"},
            {"name": "Slope of Peak ST", "unit": "0 to 2", "desc": "0: Upsloping, 1: Flat, 2: Downsloping"},
            {"name": "Major Vessels Colored", "unit": "0 to 3", "desc": "Number of major vessels (0-3)"},
            {"name": "Thalassemia", "unit": "1 to 3", "desc": "1: Normal, 2: Fixed defect, 3: Reversible defect"}
        ],
        "healthy": [45, 0, 1, 118, 210, 0, 1, 175, 0, 0.4, 2, 0, 2],
        "at_risk": [61, 1, 0, 155, 290, 1, 1, 110, 1, 3.2, 1, 2, 3],
        "pos_label": "High Risk: Signs of Heart Disease Detected",
        "neg_label": "Low Risk: Cardiovascular Profile is Normal"
    },
    "Diabetes": {
        "file": "diabetes.csv",
        "features": [
            {"name": "Pregnancies", "unit": "count", "desc": "Number of times pregnant"},
            {"name": "Glucose Level", "unit": "mg/dL", "desc": "2-hour plasma glucose concentration"},
            {"name": "Blood Pressure", "unit": "mm Hg", "desc": "Diastolic blood pressure"},
            {"name": "Skin Thickness", "unit": "mm", "desc": "Triceps skin fold thickness"},
            {"name": "Serum Insulin", "unit": "uU/mL", "desc": "2-hour serum insulin level"},
            {"name": "Body Mass Index (BMI)", "unit": "kg/m²", "desc": "Weight in kg / (height in m)^2"},
            {"name": "Diabetes Pedigree Function", "unit": "score", "desc": "Genetic family history score"},
            {"name": "Age", "unit": "years", "desc": "Patient age in years"}
        ],
        "healthy": [1, 88, 66, 22, 75, 23.4, 0.28, 25],
        "at_risk": [6, 168, 86, 38, 210, 36.5, 0.72, 48],
        "pos_label": "High Risk: Indicators of Diabetes Detected",
        "neg_label": "Low Risk: Blood Sugar & Metabolic Profile Normal"
    },
    "Breast Cancer": {
        "file": "breast_cancer.csv",
        "features": [
            {"name": "Mean Radius", "unit": "mm", "desc": "Mean distance from center to perimeter"},
            {"name": "Mean Texture", "unit": "std dev", "desc": "Standard deviation of gray-scale values"},
            {"name": "Mean Perimeter", "unit": "mm", "desc": "Core cell perimeter size"},
            {"name": "Mean Area", "unit": "mm²", "desc": "Nuclear area of cell sample"},
            {"name": "Mean Smoothness", "unit": "ratio", "desc": "Local variation in radius lengths"},
            {"name": "Mean Compactness", "unit": "ratio", "desc": "Perimeter^2 / Area - 1.0"},
            {"name": "Mean Concavity", "unit": "ratio", "desc": "Severity of concave portions"},
            {"name": "Mean Concave Points", "unit": "count", "desc": "Number of concave contour points"},
            {"name": "Mean Symmetry", "unit": "ratio", "desc": "Nuclear symmetry coefficient"},
            {"name": "Mean Fractal Dimension", "unit": "ratio", "desc": "Coastline approximation measure"}
        ],
        "healthy": [12.4, 16.5, 79.5, 470.0, 0.088, 0.065, 0.030, 0.022, 0.178, 0.061],
        "at_risk": [19.8, 24.5, 130.0, 1220.0, 0.115, 0.195, 0.210, 0.120, 0.220, 0.075],
        "pos_label": "High Risk: Morphological Signs of Malignancy Detected",
        "neg_label": "Low Risk: Tissue Sample Appears Benign / Healthy"
    }
}

MODEL_LIST = [
    "Random Forest",
    "Logistic Regression",
    "Decision Tree",
    "Support Vector Machine (SVM)",
    "K-Nearest Neighbors (KNN)"
]

def load_data(disease_name):
    cfg = DISEASES[disease_name]
    path = os.path.join("data", cfg["file"]) if os.path.exists(os.path.join("data", cfg["file"])) else cfg["file"]
    df = pd.read_csv(path)
    X = df.iloc[:, :-1].values.astype(float)
    y = df.iloc[:, -1].values.astype(int)
    return X, y, df

def get_classifier(model_name):
    if model_name == "Random Forest":
        return RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_name == "Logistic Regression":
        return LogisticRegression(max_iter=1000, random_state=42)
    elif model_name == "Decision Tree":
        return DecisionTreeClassifier(random_state=42)
    elif model_name == "Support Vector Machine (SVM)":
        return SVC(probability=True, random_state=42)
    elif model_name == "K-Nearest Neighbors (KNN)":
        return KNeighborsClassifier(n_neighbors=5)
    return RandomForestClassifier(n_estimators=100, random_state=42)

def train_and_evaluate(disease_name, model_name):
    X, y, _ = load_data(disease_name)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = get_classifier(model_name)
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred) * 100.0
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100.0
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0) * 100.0
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0) * 100.0

    return {
        "model": model_name,
        "disease": disease_name,
        "accuracy": round(acc, 1),
        "precision": round(prec, 1),
        "recall": round(rec, 1),
        "f1": round(f1, 1),
        "test_samples": len(y_test)
    }

def compare_all_models(disease_name):
    results = [train_and_evaluate(disease_name, m) for m in MODEL_LIST]
    best_model = max(results, key=lambda r: r["accuracy"])
    return {
        "disease": disease_name,
        "models": results,
        "best_model": best_model["model"],
        "best_accuracy": best_model["accuracy"]
    }

def predict_single(disease_name, model_name, values):
    cfg = DISEASES[disease_name]
    X, y, _ = load_data(disease_name)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = get_classifier(model_name)
    clf.fit(X_scaled, y)

    input_arr = np.array(values, dtype=float).reshape(1, -1)
    input_scaled = scaler.transform(input_arr)

    pred = int(clf.predict(input_scaled)[0])
    
    if hasattr(clf, "predict_proba"):
        probas = clf.predict_proba(input_scaled)[0]
        confidence = float(probas[pred]) * 100.0
        risk_score = float(probas[1]) * 100.0 if len(probas) > 1 else (100.0 if pred == 1 else 0.0)
    else:
        confidence = 90.0
        risk_score = 100.0 if pred == 1 else 0.0

    return {
        "disease": disease_name,
        "model": model_name,
        "is_positive": (pred == 1),
        "result_text": cfg["pos_label"] if pred == 1 else cfg["neg_label"],
        "confidence": round(confidence, 1),
        "risk_score": round(risk_score, 1)
    }

class RequestHandler(BaseHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path in ["/", "/index.html"]:
            html_content = get_ui_html()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=UTF-8')
            self.end_headers()
            self.wfile.write(html_content.encode('utf-8'))

        elif path == "/api/config":
            d = query.get('disease', ["Heart Disease"])[0]
            if d not in DISEASES:
                d = "Heart Disease"
            cfg = DISEASES[d]
            self.send_json({
                "disease": d,
                "features": cfg["features"],
                "healthy": cfg["healthy"],
                "at_risk": cfg["at_risk"],
                "models": MODEL_LIST
            })

        elif path == "/api/compare":
            d = query.get('disease', ["Heart Disease"])[0]
            if d not in DISEASES:
                d = "Heart Disease"
            self.send_json(compare_all_models(d))

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        raw_data = self.rfile.read(length).decode('utf-8')
        body = json.loads(raw_data) if raw_data else {}

        if self.path == "/api/predict":
            d = body.get("disease", "Heart Disease")
            m = body.get("model", "Random Forest")
            vals = body.get("values", [])
            self.send_json(predict_single(d, m, vals))
        else:
            self.send_response(404)
            self.end_headers()

    def send_json(self, data):
        resp = json.dumps(data).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=UTF-8')
        self.end_headers()
        self.wfile.write(resp)

    def log_message(self, format, *args):
        pass

def get_ui_html():
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Multiple Disease Prediction System</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet"/>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    :root {
      --bg: #f8fafc;
      --card-bg: #ffffff;
      --border: #e2e8f0;
      --text: #1e293b;
      --text-muted: #64748b;
      --primary: #2563eb;
      --primary-hover: #1d4ed8;
      --primary-light: #eff6ff;
      --success-bg: #dcfce7;
      --success-text: #15803d;
      --danger-bg: #fee2e2;
      --danger-text: #b91c1c;
      --shadow: 0 4px 6px -1px rgba(0,0,0,0.07);
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); padding-bottom: 40px; }
    .container { max-width: 1050px; margin: 0 auto; padding: 20px; }
    header { background: #1e3a8a; color: white; padding: 20px 24px; border-radius: 12px; margin-bottom: 20px; box-shadow: var(--shadow); display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; }
    header h1 { font-size: 20px; font-weight: 700; }
    header p { font-size: 13px; opacity: 0.85; margin-top: 2px; }
    .nav-tabs { display: flex; gap: 10px; margin-bottom: 20px; border-bottom: 2px solid var(--border); padding-bottom: 8px; }
    .tab-btn { background: none; border: none; padding: 8px 16px; font-size: 14px; font-weight: 600; color: var(--text-muted); cursor: pointer; border-radius: 6px; }
    .tab-btn.active { background: var(--primary); color: white; }
    .config-box { background: var(--card-bg); border: 1px solid var(--border); border-radius: 10px; padding: 16px; margin-bottom: 20px; display: grid; grid-template-columns: 1fr 1fr; gap: 16px; box-shadow: var(--shadow); }
    @media(max-width: 700px) { .config-box { grid-template-columns: 1fr; } }
    label { font-size: 13px; font-weight: 600; color: var(--text); margin-bottom: 6px; display: block; }
    select, input { width: 100%; padding: 10px; border-radius: 6px; border: 1px solid var(--border); font-size: 14px; outline: none; }
    select:focus, input:focus { border-color: var(--primary); }
    .grid { display: grid; grid-template-columns: 1.2fr 1fr; gap: 20px; }
    @media(max-width: 850px) { .grid { grid-template-columns: 1fr; } }
    .card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 10px; padding: 20px; box-shadow: var(--shadow); }
    .card-title { font-size: 16px; font-weight: 700; margin-bottom: 14px; padding-bottom: 8px; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; }
    .sample-btns { display: flex; gap: 8px; margin-bottom: 14px; }
    .btn-sm { padding: 6px 12px; font-size: 12px; font-weight: 600; border: 1px solid var(--border); background: var(--bg); border-radius: 6px; cursor: pointer; }
    .btn-sm:hover { background: var(--primary-light); color: var(--primary); border-color: var(--primary); }
    .form-inputs { max-height: 400px; overflow-y: auto; padding-right: 8px; }
    .input-row { margin-bottom: 12px; }
    .input-desc { font-size: 11px; color: var(--text-muted); margin-top: 2px; }
    .btn-submit { width: 100%; background: var(--primary); color: white; border: none; padding: 12px; font-size: 15px; font-weight: 600; border-radius: 6px; cursor: pointer; margin-top: 10px; }
    .btn-submit:hover { background: var(--primary-hover); }
    .result-box { padding: 18px; border-radius: 8px; text-align: center; margin-bottom: 16px; display: none; }
    .result-box.danger { background: var(--danger-bg); color: var(--danger-text); border: 1px solid #fca5a5; }
    .result-box.success { background: var(--success-bg); color: var(--success-text); border: 1px solid #86efac; }
    .result-title { font-size: 17px; font-weight: 700; margin-bottom: 4px; }
    .result-sub { font-size: 13px; font-weight: 500; }
    .stat-row { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid var(--border); font-size: 13px; }
    .stat-row:last-child { border-bottom: none; }
    table { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 13px; }
    th, td { padding: 10px; text-align: left; border-bottom: 1px solid var(--border); }
    th { background: #f8fafc; font-weight: 600; }
    .best-tag { background: #dcfce7; color: #15803d; font-size: 11px; font-weight: 700; padding: 2px 6px; border-radius: 4px; margin-left: 6px; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <h1>Multiple Disease Prediction System</h1>
        <p>Machine Learning project for predicting Heart Disease, Diabetes, and Breast Cancer</p>
      </div>
    </header>

    <div class="nav-tabs">
      <button class="tab-btn active" onclick="switchTab('predict')">Single Patient Prediction</button>
      <button class="tab-btn" onclick="switchTab('compare')">Model Accuracy Comparison</button>
    </div>

    <div class="config-box">
      <div>
        <label>Select Disease Dataset:</label>
        <select id="diseaseSelect" onchange="onDiseaseChange()">
          <option value="Heart Disease">Heart Disease (13 Features)</option>
          <option value="Diabetes">Diabetes (8 Features)</option>
          <option value="Breast Cancer">Breast Cancer (10 Features)</option>
        </select>
      </div>
      <div>
        <label>Select Machine Learning Algorithm:</label>
        <select id="modelSelect">
          <option value="Random Forest">Random Forest Classifier</option>
          <option value="Logistic Regression">Logistic Regression</option>
          <option value="Decision Tree">Decision Tree Classifier</option>
          <option value="Support Vector Machine (SVM)">Support Vector Machine (SVM)</option>
          <option value="K-Nearest Neighbors (KNN)">K-Nearest Neighbors (KNN)</option>
        </select>
      </div>
    </div>

    <div id="predictTab" class="grid">
      <div class="card">
        <div class="card-title">
          <span>Patient Test Parameters</span>
          <div class="sample-btns">
            <button class="btn-sm" onclick="loadSample('healthy')">Load Healthy Sample</button>
            <button class="btn-sm" onclick="loadSample('at_risk')">Load At-Risk Sample</button>
          </div>
        </div>
        <div class="form-inputs" id="inputFields"></div>
        <button class="btn-submit" onclick="runPrediction()">Run Prediction</button>
      </div>

      <div class="card">
        <div class="card-title">Prediction Result</div>
        <div id="emptyResult" style="text-align: center; color: var(--text-muted); padding: 40px 10px; font-size: 14px;">
          Click "Run Prediction" or load a sample patient to view the diagnostic result.
        </div>
        <div id="resultBox" class="result-box">
          <div id="resultTitle" class="result-title"></div>
          <div id="resultSub" class="result-sub"></div>
        </div>

        <div id="detailsCard" style="display:none; margin-top: 15px;">
          <h4 style="font-size: 14px; margin-bottom: 10px;">Diagnostic Breakdown:</h4>
          <div class="stat-row"><span>Target Condition:</span><strong id="detDisease">-</strong></div>
          <div class="stat-row"><span>Algorithm Used:</span><strong id="detModel">-</strong></div>
          <div class="stat-row"><span>Confidence Score:</span><strong id="detConfidence">-</strong></div>
          <div class="stat-row"><span>Estimated Risk Probability:</span><strong id="detRisk">-</strong></div>
        </div>
      </div>
    </div>

    <div id="compareTab" class="card" style="display:none;">
      <div class="card-title">
        <span>Algorithm Performance Comparison (<span id="compareDiseaseName"></span>)</span>
        <button class="btn-sm" onclick="fetchComparison()">Refresh Comparison</button>
      </div>
      <div style="height: 260px; margin-bottom: 20px;">
        <canvas id="accuracyChart"></canvas>
      </div>
      <table>
        <thead>
          <tr>
            <th>Algorithm</th>
            <th>Accuracy</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1-Score</th>
          </tr>
        </thead>
        <tbody id="comparisonTableBody"></tbody>
      </table>
    </div>
  </div>

  <script>
    let currentConfig = null;
    let myChart = null;

    async function onDiseaseChange() {
      const disease = document.getElementById('diseaseSelect').value;
      const res = await fetch('/api/config?disease=' + encodeURIComponent(disease));
      currentConfig = await res.json();
      renderInputs();
      resetResult();
      if (document.getElementById('compareTab').style.display !== 'none') {
        fetchComparison();
      }
    }

    function renderInputs() {
      const container = document.getElementById('inputFields');
      container.innerHTML = '';
      currentConfig.features.forEach((f, idx) => {
        const div = document.createElement('div');
        div.className = 'input-row';
        div.innerHTML = `
          <label>${f.name} <span style="color:var(--text-muted);font-weight:400;">(${f.unit})</span></label>
          <input type="number" step="any" id="feat_${idx}" value="${currentConfig.healthy[idx]}">
          <div class="input-desc">${f.desc}</div>
        `;
        container.appendChild(div);
      });
    }

    function loadSample(type) {
      if (!currentConfig) return;
      const sample = type === 'healthy' ? currentConfig.healthy : currentConfig.at_risk;
      sample.forEach((val, idx) => {
        const el = document.getElementById('feat_' + idx);
        if (el) el.value = val;
      });
      runPrediction();
    }

    async function runPrediction() {
      if (!currentConfig) return;
      const disease = document.getElementById('diseaseSelect').value;
      const model = document.getElementById('modelSelect').value;
      const values = currentConfig.features.map((_, idx) => {
        return parseFloat(document.getElementById('feat_' + idx).value) || 0;
      });

      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ disease, model, values })
      });
      const data = await res.json();

      document.getElementById('emptyResult').style.display = 'none';
      const box = document.getElementById('resultBox');
      box.style.display = 'block';
      box.className = 'result-box ' + (data.is_positive ? 'danger' : 'success');
      document.getElementById('resultTitle').textContent = data.result_text;
      document.getElementById('resultSub').textContent = `Model Confidence: ${data.confidence}% (Risk Score: ${data.risk_score}%)`;

      document.getElementById('detailsCard').style.display = 'block';
      document.getElementById('detDisease').textContent = data.disease;
      document.getElementById('detModel').textContent = data.model;
      document.getElementById('detConfidence').textContent = data.confidence + '%';
      document.getElementById('detRisk').textContent = data.risk_score + '%';
    }

    function resetResult() {
      document.getElementById('emptyResult').style.display = 'block';
      document.getElementById('resultBox').style.display = 'none';
      document.getElementById('detailsCard').style.display = 'none';
    }

    function switchTab(tab) {
      document.querySelectorAll('.tab-btn').forEach((b, i) => {
        b.classList.toggle('active', (tab === 'predict' && i === 0) || (tab === 'compare' && i === 1));
      });
      document.getElementById('predictTab').style.display = tab === 'predict' ? 'grid' : 'none';
      document.getElementById('compareTab').style.display = tab === 'compare' ? 'block' : 'none';
      if (tab === 'compare') {
        fetchComparison();
      }
    }

    async function fetchComparison() {
      const disease = document.getElementById('diseaseSelect').value;
      document.getElementById('compareDiseaseName').textContent = disease;
      const res = await fetch('/api/compare?disease=' + encodeURIComponent(disease));
      const data = await res.json();

      const tbody = document.getElementById('comparisonTableBody');
      tbody.innerHTML = '';
      const labels = [];
      const accData = [];

      data.models.forEach(m => {
        labels.push(m.model);
        accData.push(m.accuracy);

        const isBest = (m.model === data.best_model);
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td><strong>${m.model}</strong> ${isBest ? '<span class="best-tag">Highest Accuracy</span>' : ''}</td>
          <td>${m.accuracy}%</td>
          <td>${m.precision}%</td>
          <td>${m.recall}%</td>
          <td>${m.f1}%</td>
        `;
        tbody.appendChild(tr);
      });

      renderChart(labels, accData);
    }

    function renderChart(labels, data) {
      const ctx = document.getElementById('accuracyChart').getContext('2d');
      if (myChart) myChart.destroy();
      myChart = new Chart(ctx, {
        type: 'bar',
        data: {
          labels: labels,
          datasets: [{
            label: 'Accuracy (%)',
            data: data,
            backgroundColor: '#3b82f6',
            borderRadius: 4
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            y: { min: 50, max: 100 }
          }
        }
      });
    }

    window.onload = onDiseaseChange;
  </script>
</body>
</html>
"""

def main():
    print("=" * 60)
    print("      Multiple Disease Prediction System")
    print("=" * 60)
    print("Loaded Datasets:")
    for d_name, d_cfg in DISEASES.items():
        try:
            X, y, _ = load_data(d_name)
            print(f" - {d_name}: {len(y)} samples, {X.shape[1]} features")
        except Exception as e:
            print(f" - {d_name}: Error loading ({e})")

    httpd = None
    selected_port = DEFAULT_PORT
    for port in [5000, 8000, 5050, 8088, 8888]:
        try:
            server_address = ('127.0.0.1', port)
            httpd = HTTPServer(server_address, RequestHandler)
            selected_port = port
            break
        except Exception:
            continue

    if not httpd:
        print("Error: Could not bind to any standard port.")
        return

    url = f"http://127.0.0.1:{selected_port}"
    print(f"\nStarting Web Dashboard at: {url}")
    print("Opening in default browser automatically...")
    print("Press Ctrl+C in terminal to stop server.\n")
    
    try:
        webbrowser.open(url)
    except Exception:
        pass

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        httpd.server_close()

if __name__ == "__main__":
    main()
