# FraudGuard v3

**Insurance Claim Fraud Detection System**

ML Ensemble (RF+ET+GB) · Groq Llama 3 70B · 5-Agent AI Pipeline · Document Extraction

---

## Quick Start

```bash
cd frontend/
npm install
npm run dev
# Open http://localhost:3000
# Enter your Groq API key (free at console.groq.com)
```

That's it. No Python, no backend, no database needed.

---

## What It Does

1. **ML Scoring** — Instant fraud probability (0–99%) using a trained Random Forest + ExtraTrees + Gradient Boosting ensemble. Runs in the browser with no API calls.

2. **AI Investigation** — 5 Llama 3 70B agents via Groq run sequentially: Document Extraction → Narrative Analysis → Fraud Detection → Investigator Reasoning → Formal Report.

3. **Document Upload** — Paste any claim text; Groq extracts all 17 fields automatically, highlighted in green for review.

4. **Risk Breakdown** — Factor-by-factor analysis categorised as High Risk, Elevated Risk, or Mitigating.

5. **Audit Log** — Tracks every claim analysed in the session.

---

## Getting a Groq API Key (Free)

1. Go to https://console.groq.com
2. Sign up (no credit card)
3. Click **API Keys → Create Key**
4. Paste into the FraudGuard login screen

---

## Re-train the Model

Download the datasets from Kaggle:
https://www.kaggle.com/datasets/roshansharma/insurance-fraud-detection

```bash
cp ~/Downloads/fraud_oracle.csv data/
cp ~/Downloads/carclaims.csv    data/
cd ml/
pip install -r requirements.txt
python train_model.py
# ~3-5 minutes. Saves models/ folder.
```

---

## Optional Backend

```bash
cd backend/
pip install -r requirements.txt
uvicorn fraud_api:app --reload --port 8000
# API docs at http://localhost:8000/docs
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18 + Vite 5 |
| Fonts | Syne · IBM Plex Mono · Plus Jakarta Sans |
| LLM | Groq — Llama 3 70B (`llama3-70b-8192`) |
| ML | scikit-learn — RF + ExtraTrees + GradientBoosting |
| Backend | FastAPI + Uvicorn (optional) |
| Training data | fraud_oracle.csv + carclaims.csv (15,420 rows) |

---

## Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | 94.20% |
| ROC-AUC | 0.8381 |
| Precision | 87.5% |
| Recall | 71.4% |
| F1 Score | 0.786 |
| CV AUC | 0.8516 ± 0.012 |

---

## Full Documentation

See `docs/TECHNICAL_DOCUMENTATION.md` for the complete guide covering:
architecture, algorithms, feature engineering, training pipeline, agent design, dataset schemas, API reference, and more.
