# 🚛 Scania APS Failure Prediction Platform

> **Enterprise-grade Predictive Maintenance using Machine Learning**

An end-to-end Machine Learning solution for predicting **Air Pressure System (APS) failures** in Scania heavy-duty trucks. The project focuses on minimizing maintenance costs through intelligent failure prediction, explainable AI, and an interactive dashboard.

---

# 📌 Table of Contents

- [Overview](#overview)
- [Business Problem](#business-problem)
- [Project Objectives](#project-objectives)
- [Features](#features)
- [Project Architecture](#project-architecture)
- [Technology Stack](#technology-stack)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Models Used](#models-used)
- [Performance Metrics](#performance-metrics)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Future Improvements](#future-improvements)
- [License](#license)

---

# 📖 Overview

The **Scania APS Failure Prediction Platform** is an enterprise-style predictive maintenance system that predicts failures in the **Air Pressure System (APS)** of Scania trucks using historical sensor data.

Instead of treating every classification error equally, this project optimizes predictions based on **real-world business costs**, where:

- Missing a real failure (False Negative) is extremely expensive.
- Incorrectly predicting a failure (False Positive) has a much smaller cost.

This cost-sensitive approach produces decisions that are more valuable in industrial environments.

---

# 💼 Business Problem

Unexpected APS failures can lead to:

- Expensive truck downtime
- High repair costs
- Delivery delays
- Customer dissatisfaction
- Increased operational risks

The goal is to identify failures **before they occur**, allowing preventive maintenance.

### Business Cost Function

```
Total Cost = (False Positives × 10) + (False Negatives × 500)
```

Since a missed failure is **50× more expensive** than a false alarm, the model is optimized to minimize overall business cost rather than maximizing accuracy alone.

---

# 🎯 Project Objectives

- Predict APS failures with high reliability.
- Minimize overall maintenance cost.
- Reduce expensive False Negatives.
- Build an explainable AI solution.
- Provide maintenance recommendations.
- Deploy an interactive dashboard.

---

# ✨ Features

- 📊 Cost-sensitive machine learning
- 🚀 Automated preprocessing pipeline
- 🎯 Threshold optimization
- 🔍 SHAP Explainable AI
- 📈 Model comparison dashboard
- 📉 Feature importance visualization
- 🧪 What-if simulation
- ⚙️ Hyperparameter optimization with Optuna
- 🌐 Interactive Streamlit web application

---

# 🏗️ Project Architecture

```
aps-failure-prediction/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_Modeling.ipynb
│   └── 04_Evaluation.ipynb
│
├── src/
│   ├── app/
│   │   └── streamlit_app.py
│   │
│   ├── data/
│   │   ├── preprocessing.py
│   │   └── feature_engineering.py
│   │
│   ├── models/
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── evaluate.py
│   │
│   ├── explainability/
│   │   └── shap_analysis.py
│   │
│   └── utils/
│       └── helper.py
│
├── models/
│
├── reports/
│
├── requirements.txt
│
└── README.md
```

---

# 🛠 Technology Stack

### Programming

- Python 3.10+

### Data Processing

- Pandas
- NumPy

### Machine Learning

- Scikit-learn
- XGBoost
- LightGBM
- CatBoost
- Random Forest
- Logistic Regression

### Hyperparameter Optimization

- Optuna

### Explainability

- SHAP

### Visualization

- Matplotlib
- Seaborn
- Plotly

### Deployment

- Streamlit

---

# 🔄 Machine Learning Pipeline

```
Raw Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Missing Value Handling
      │
      ▼
Feature Engineering
      │
      ▼
Train/Test Split
      │
      ▼
Model Training
      │
      ▼
Hyperparameter Tuning
      │
      ▼
Threshold Optimization
      │
      ▼
Business Cost Evaluation
      │
      ▼
Explainability (SHAP)
      │
      ▼
Streamlit Dashboard
```

---

# 🤖 Models Used

The project compares multiple machine learning algorithms:

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- CatBoost

Each model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- Business Cost

The final model is selected based on the **lowest business cost**, not just predictive accuracy.

---

# 📊 Performance Metrics

| Metric | Target |
|---------|--------|
| Business Cost | Minimized |
| Accuracy | >95% |
| Precision | High |
| Recall | High |
| F1 Score | >0.85 |
| ROC-AUC | >0.95 |

---

# 📁 Project Structure

```
data/
```
Stores raw and processed datasets.

```
notebooks/
```
Contains exploratory analysis and experimentation notebooks.

```
src/
```
Complete machine learning source code.

```
models/
```
Saved trained models.

```
reports/
```
Generated reports and visualizations.

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/aps-failure-prediction.git

cd aps-failure-prediction
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment.

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Application

Launch the Streamlit dashboard:

```bash
streamlit run src/app/streamlit_app.py
```

The application will open automatically in your browser.

---

# 📈 Dashboard Features

The interactive dashboard provides:

- Failure prediction
- Prediction confidence
- SHAP feature importance
- Maintenance recommendations
- Business cost estimation
- What-if scenario analysis
- Model performance comparison

---

# 🚀 Future Improvements

- Docker containerization
- CI/CD pipeline
- Cloud deployment (AWS/Azure/GCP)
- Real-time API with FastAPI
- MLflow experiment tracking
- Automated retraining pipeline
- Kubernetes deployment
- Monitoring with Prometheus & Grafana

---

# 🤝 Contributing

Contributions are welcome.

1. Fork the repository.
2. Create a new feature branch.
3. Commit your changes.
4. Push the branch.
5. Open a Pull Request.

---

# 📄 License

This project is intended for educational, research, and demonstration purposes.

---

# 👨‍💻 Author

**Abiha Ehtisham**

Machine Learning | Data Science | Predictive Maintenance | Explainable AI

---

⭐ If you found this project useful, consider giving it a **Star** on GitHub!