# 🛡️ App Performance Intelligence System

ML-based anomaly detection for mobile app performance monitoring with real-time dashboard and AI-powered root cause analysis.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 🎯 What It Does

Detects anomalies in mobile app telemetry (API latency, FPS, memory, errors) using Isolation Forest ML, displays results on an interactive dashboard, and explains anomalies using AI.

**5 Anomaly Types Detected:** Memory Leak · Latency Spike · FPS Drop · Error Burst · API Timeout

---

## 🏗️ Pipeline

![Pipeline](assets/pipeline.png)

---

## 🚀 Quick Start
```bash
# Clone
git clone https://github.com/yourusername/app-performance-intelligence-system.git
cd app-performance-intelligence-system

# Install
pip install -r requirements.txt

# Run Dashboard
streamlit run app/streamlit_app.py
```

### 🔧 Train From Scratch (Optional)
```bash
python src/data_generator.py        # Generate synthetic data
python src/preprocess.py            # Feature engineering
python src/train_model.py           # Train models
python src/evaluate_model.py        # Evaluate & generate charts
streamlit run app/streamlit_app.py  # Launch dashboard
```

---

## 📊 Model Performance

| Metric | Baseline (Z-Score) | Isolation Forest | Improvement |
|---|---|---|---|
| Precision | 0.207 | 0.460 | +122.7% |
| Recall | 0.951 | 0.815 | -14.4% |
| F1-Score | 0.339 | 0.588 | +73.2% |
| AUC-ROC | — | 0.907 | Excellent |

### Detection by Anomaly Type

| Type | Detection Rate |
|---|---|
| Latency Cascade | 100.0% |
| API Timeout | 84.6% |
| Memory Leak | 82.4% |
| Error Burst | 79.4% |
| FPS Drop | 73.7% |

---

## 🖥️ Dashboard Features

| Feature | Description |
|---|---|
| 🚦 **System Status** | Real-time anomaly / normal indicator |
| 📊 **Anomaly Score** | 0–100% confidence gauge |
| 🎯 **Top Trigger** | Most contributing feature |
| 📈 **Trend Chart** | Last 30 events with anomaly markers |
| 🎯 **Radar Chart** | System health shape (top 8 features) |
| 🤖 **AI Analysis** | Root cause, impact, and recommendations |
| 🔬 **Verification** | Ground truth vs prediction comparison |
| 📥 **Export** | Download predictions as CSV |

---

## 🤖 AI Explainer

| Mode | API Key | Speed | Quality |
|---|---|---|---|
| **Groq** (default) | System key included | ⚡ Fast | ⭐⭐⭐⭐⭐ |
| **Gemini** | Optional | Moderate | ⭐⭐⭐⭐⭐ |
| **Rule-Based** | Not needed | ⚡ Instant | ⭐⭐⭐⭐ |

> System API key is included. If rate limit is reached, enter your own key or switch to rule-based mode.

---

## 📁 Project Structure
```
├── app/                        # Streamlit Dashboard
│   ├── streamlit_app.py        # Main entry point
│   ├── anomaly_detector.py     # Data generator + model wrapper
│   ├── dashboard.py            # Charts and visualizations
│   ├── ai_explainer.py         # LLM integration (Gemini/Groq)
│   └── utils.py                # Constants and helpers
├── src/                        # ML Pipeline
│   ├── data_generator.py       # Synthetic data generation
│   ├── preprocess.py           # Feature engineering + scaling
│   ├── train_model.py          # Model training (grid search + ensemble)
│   ├── evaluate_model.py       # Evaluation (9 visualizations)
│   └── inference.py            # Production inference
├── models/                     # Trained model artifacts (.pkl)
├── evaluation/                 # Evaluation charts and reports
├── data/                       # Raw and processed datasets
├── notebooks/                  # EDA Jupyter notebook
├── reports/                    # Analysis reports
├── assets/                     # Pipeline diagram
└── predictions/                # Inference outputs
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **ML Model** | Isolation Forest (scikit-learn) |
| **Dashboard** | Streamlit + Plotly |
| **AI Explainer** | Groq / Google Gemini |
| **Data** | Pandas · NumPy |
| **Evaluation** | Matplotlib · Seaborn |

---

## 📋 Dataset

| Detail | Value |
|---|---|
| **Total Events** | 10,000 (synthetic) |
| **Train / Test** | 8,000 / 2,000 |
| **Features** | 29 (rolling stats, z-scores, rates) |
| **Anomaly Rate** | ~12.32% |
| **Anomaly Types** | 5 |

---

## 🔮 Future Improvements

- Real data integration (Datadog, New Relic)
- Supervised models (XGBoost) with labeled production data
- Adaptive thresholding
- Email / Slack alerting
- Multi-app monitoring support

---

## 👤 Author

**Ashhar Ali**

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.