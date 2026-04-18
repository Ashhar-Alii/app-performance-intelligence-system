Markdown
# 🛡️ App Performance Intelligence System

ML-based anomaly detection for mobile app performance monitoring with real-time dashboard, AI-powered root cause analysis, and secure multi-tenant cloud storage.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Supabase-336791.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 🎯 What It Does

Detects anomalies in mobile app telemetry (API latency, FPS, memory, errors) using Isolation Forest ML, displays results on an interactive dashboard, explains anomalies using AI, and securely logs incidents to a multi-tenant PostgreSQL database.

**5 Anomaly Types Detected:** Memory Leak · Latency Spike · FPS Drop · Error Burst · API Timeout

---

## 🏗️ Pipeline

![Pipeline](assets/Pipeline.png)

---

# 🧠 App Performance Intelligence System

> AI-Driven Mobile App Performance Anomaly Detection  
> BCA Final Year Project — Ashhar Ali, 2026

---

## 🚀 Quick Start

### Clone & Install

```bash
git clone https://github.com/yourusername/app-performance-intelligence-system.git
cd app-performance-intelligence-system

pip install -r requirements.txt
```

### Setup Database

Add your credentials to `.streamlit/secrets.toml`:

```toml
SUPABASE_URL = "your_supabase_url"
SUPABASE_KEY = "your_supabase_key"
```

### Run Dashboard

```bash
streamlit run app/streamlit_app.py
```

---

## 🔧 Train From Scratch (Optional)

```bash
python src/data_generator.py        # Generate synthetic data
python src/preprocess.py            # Feature engineering
python src/train_model.py           # Train models
python src/evaluate_model.py        # Evaluate & generate charts
streamlit run app/streamlit_app.py  # Launch dashboard
```

---

## 📊 Model Performance

### Overall Metrics

| Metric    | Baseline (Z-Score) | Isolation Forest | Improvement |
|-----------|--------------------|------------------|-------------|
| Precision | 0.207              | 0.460            | +122.7%     |
| Recall    | 0.951              | 0.815            | −14.4%      |
| F1-Score  | 0.339              | 0.588            | +73.2%      |
| AUC-ROC   | —                  | 0.907            | Excellent   |

### Detection Rate by Anomaly Type

| Anomaly Type     | Detection Rate |
|------------------|----------------|
| Latency Cascade  | 100.0%         |
| API Timeout      | 84.6%          |
| Memory Leak      | 82.4%          |
| Error Burst      | 79.4%          |
| FPS Drop         | 73.7%          |

---

## 🔬 Real-World Validation

The model was trained exclusively on synthetic mobile app data. Real-world tests are **zero-shot** — no retraining was performed. Results validate that the feature engineering + Isolation Forest methodology transfers across domains.

### Benchmark Results

| Dataset   | Domain                 | Files Tested | Avg F1 | Best F1 |
|-----------|------------------------|--------------|--------|---------|
| Synthetic | Mobile app telemetry   | —            | 0.588  | —       |
| SKAB      | Industrial sensors     | 34           | 0.434  | 0.753   |
| NAB       | Server / Cloud metrics | 37           | 0.240  | 0.667   |

### Cross-Domain Comparison

| Metric    | Synthetic | SKAB  | NAB   |
|-----------|-----------|-------|-------|
| Precision | 0.460     | 0.562 | 0.167 |
| Recall    | 0.815     | 0.361 | 0.568 |
| F1-Score  | 0.588     | 0.434 | 0.240 |

> **Key Insight:** Performance is strongest on the trained domain (synthetic mobile data) and decreases on increasingly dissimilar domains (industrial sensors → server metrics). This is expected zero-shot transfer behavior.

Datasets: [NAB (Numenta Anomaly Benchmark)](https://github.com/numenta/NAB) · [SKAB (Skoltech Anomaly Benchmark)](https://github.com/waico/SKAB)

---

## 🖥️ Dashboard Features

| Feature               | Description                                                              |
|-----------------------|--------------------------------------------------------------------------|
| 🔐 Authentication     | Secure multi-tenant login with SHA-256 password hashing                  |
| 🚦 System Status      | Real-time anomaly / normal indicator with severity levels                |
| 🗄️ Historical DB      | Persistent PostgreSQL storage with user-specific data isolation          |
| 📈 Trend Charts       | Live tracking (last 30 events) + DB history time-series analysis         |
| 🎯 Radar Chart        | System health shape across top 8 features                                |
| 🤖 AI Analysis        | LLM-powered root cause, impact assessment, and recommendations           |
| 🔮 What-If Simulation | Projected recovery metrics if corrective action is taken                 |
| 📥 CSV Analysis       | Upload raw telemetry, analyze, and save discovered anomalies to DB       |
| 🔬 Model Verification | Ground truth vs. prediction comparison with live confusion matrix        |

---

## 🤖 AI Explainer Modes

| Mode              | API Key Required | Speed             | Quality       |
|-------------------|------------------|-------------------|---------------|
| Groq (default)    | System key included | ⚡ Fast (0.3–1s) | ⭐⭐⭐⭐⭐     |
| Rule-Based        | Not needed       | ⚡ Instant        | ⭐⭐⭐⭐       |

> The system API key is included. If the rate limit is reached, enter your own key or switch to rule-based mode.  
> Get a free Groq key at [console.groq.com/keys](https://console.groq.com/keys).

---

## 📁 Project Structure

```
├── app/                            # Streamlit Dashboard
│   ├── streamlit_app.py            # Main entry point & auth logic
│   ├── anomaly_detector.py         # Live data generator + model wrapper
│   ├── dashboard.py                # Charts and visualizations (live & DB)
│   ├── ai_explainer.py             # Groq LLM + rule-based fallback
│   └── utils.py                    # Constants and helpers
│
├── src/                            # ML Pipeline
│   ├── data_generator.py           # Synthetic data generation
│   ├── preprocess.py               # Feature engineering + scaling
│   ├── train_model.py              # Model training (grid search + ensemble)
│   ├── evaluate_model.py           # Evaluation (9 visualizations)
│   ├── inference.py                # Production inference module
│   ├── test_on_real_data(NAB).py   # NAB benchmark validation
│   └── test_on_skab_data(SKAB).py  # SKAB benchmark validation
│
├── models/                         # Trained model artifacts (.pkl)
├── evaluation/                     # Evaluation charts and reports
├── data/                           # Raw and processed datasets
├── notebooks/                      # EDA Jupyter notebook
└── assets/                         # Pipeline diagram
```

---

## 🛠️ Tech Stack

| Component       | Technology                  |
|-----------------|-----------------------------|
| Language        | Python 3.9+                 |
| ML Model        | Isolation Forest (scikit-learn) |
| Database        | PostgreSQL (Supabase API)   |
| Dashboard       | Streamlit + Plotly          |
| AI Explainer    | Groq (Llama 3.1)            |
| Security        | Python hashlib (SHA-256)    |
| Data Processing | Pandas · NumPy              |

---

## 📋 Dataset

| Detail           | Value                                                       |
|------------------|-------------------------------------------------------------|
| Total Events     | 10,000 (synthetic)                                          |
| Train / Test     | 8,000 / 2,000                                               |
| Features         | 29 (rolling stats, z-scores, rates)                         |
| Anomaly Rate     | ~12.32%                                                     |
| Anomaly Types    | 5 (memory leak, latency spike, FPS drop, error burst, API timeout) |
| Real-World Validation | NAB (37 files) + SKAB (34 files)                      |

---

## ✨ Key Novelties

| Novelty                | Description                                                              |
|------------------------|--------------------------------------------------------------------------|
| Multi-Tenant SaaS DB   | Secure user auth with isolated PostgreSQL anomaly logging per user       |
| Explainable AI         | LLM-powered root cause analysis with rule-based fallback                 |
| Real-Time Simulation   | Live monitoring with interactive anomaly forcing                         |
| RCA + What-If          | Causal chain analysis with projected recovery simulation                 |
| Feature Engineering    | 29 features derived from 5 raw metrics (rolling stats, z-scores, rates) |
| Ensemble Approach      | Grid search over 36 combos + top-3 model ensemble                       |
| Real-World Validation  | Zero-shot testing on NAB and SKAB benchmarks                            |

---

## 🔮 Future Improvements

- Real APM integration (Datadog, New Relic, Prometheus)
- Supervised models (XGBoost) with labeled production data
- Adaptive thresholding based on recent data patterns
- Email / Slack alerting for critical anomalies
- Kafka / Redis Streams for production-scale event streaming
- Feedback loop for continuous model improvement

---

## 👤 Author

**Ashhar Ali**  
BCA Final Year Project — Department of Computer Science & Engineering  
Jamia Hamdard, New Delhi · 2026