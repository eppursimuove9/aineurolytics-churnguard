# 🛡️ ChurnGuard™: Predictive RevOps Intelligence Engine
 
**Enterprise-Grade Churn Prediction System with FastAPI, XGBoost, Modular Architecture, and CI/CD**

ChurnGuard™ is an enterprise-oriented predictive engine built to score customer churn probability in real time.  
It is designed for SaaS RevOps teams seeking automation, predictive insights, and a scalable model-first architecture.

This repository provides:
- A complete ML inference engine  
- A high-performance FastAPI service  
- A reproducible dummy model  
- Automated CI with GitHub Actions  
- Comprehensive tests  
- Fully modular project structure  
- Technical documentation  

---

## Key Features

### Predictive ML Engine
- XGBoost classifier with reproducible dummy model
- Deterministic feature ordering
- Probability scoring + risk segmentation (LOW / MEDIUM / HIGH)
- Embedded model version metadata

### API Layer (FastAPI)
- Real-time inference endpoint: `/api/v1/predict`
- Health endpoint: `/api/health`
- Pydantic validation for input/output
- Auto-generated OpenAPI docs (Swagger & ReDoc)

### MLOps Architecture
- Modular directory structure (`src/api`, `src/ml`, `src/orchestration`)
- Extensible feature engineering and pipeline design
- Reproducible dummy model training
- Test suite included (pytest)

### CI/CD
- GitHub Actions CI pipeline under `.github/workflows/ci.yml`
- Automated testing for every push and PR

### Documentation
- Full Technical Design Document (TDD) located in `/docs`

---

## Repository Structure

```plaintext
aineurolytics-churnguard/
│
├── src/
│   ├── api/
│   │   └── main.py                   # FastAPI inference service
│   ├── ml/
│   │   ├── churn_predictor.py        # ML prediction engine
│   │   └── dummy_model.py            # Dummy XGBoost training script
│   └── orchestration/                # Reserved for pipelines, ETLs, automation
│
├── tests/
│   ├── test_api.py                   # API endpoint tests
│   └── test_predictor.py             # ML engine tests
│
├── models/                           # Trained ML models (dummy model generated here)
│
├── docs/
│   └── aineurolytics_churnguard.pdf  # Full Technical Design Document
│
├── .github/
│   └── workflows/
│       └── ci.yml                    # CI configuration
│
├── requirements.txt
├── .gitignore
└── README.md
