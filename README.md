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
├── .github/
│   └── ci.yml                    # CI configuration
├── docs/
│   └── aineurolytics_churnguard.pdf  # Full Technical Design Document
├── src/
│   ├── api/
│   │   └── main.py                   # FastAPI inference service
│   ├── ml/
│   │   ├── churn_predictor.py        # ML prediction engine
│   │   └── dummy_model.py            # Dummy XGBoost training script
│   └── API py
│   └── engine py
│
├── tests/
│   ├── test_api.py                   # API endpoint tests
    └── test_engine.py
│   └── test_predictor.py             # ML engine tests
│
├── .gitignore
│
├── models/                           # Trained ML models (dummy model generated here)
│
├── requirements.txt
└── README.md
```


## Installation ## 

### Clone the repository

```bash
git clone https://github.com/eppursimuove9/aineurolytics-churnguard.git
cd aineurolytics-churnguard
```

### Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

## Train the Dummy Model (Required Before Running API)

The repository includes a reproducible dummy model (XGBoost) for testing without real customer data.

### Train model

```bash
python src/ml/dummy_model.py
```

This will output: `models/dummy_xgb.pkl`

## Running the API

### Export Python path

```bash
export PYTHONPATH=$(pwd)
```

### Launch the FastAPI service

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Open API documentation

- **Swagger UI** → http://localhost:8000/docs
- **ReDoc** → http://localhost:8000/redoc

## API Endpoints

### GET /api/health

**Example Response:**

```json
{
  "status": "ok",
  "timestamp": "2025-01-10T22:10:21.000Z"
}
```

### POST /api/v1/predict

**Example Request:**

```json
{
  "customer_id": "12345",
  "features": {
    "feature_0": 0.12,
    "feature_1": 0.94,
    "feature_2": 0.33
  }
}
```

**Example Response:**

```json
{
  "customer_id": "12345",
  "prediction_timestamp": "2025-01-10T22:14:12.012Z",
  "churn_probability": 0.6812,
  "churn_risk_level": "MEDIUM",
  "model_version": "v0.0.1"
}
```

## Running Tests

Execute full test suite:

```bash
export PYTHONPATH=$(pwd)
pytest -q
```

**Coverage includes:**

- ML prediction engine
- FastAPI endpoints

## Continuous Integration (CI)

GitHub Actions workflow: `.github/workflows/ci.yml`

**Pipeline runs on:**

- every push
- every pull request

**Stages:**

1. Checkout
2. Python setup
3. Install dependencies
4. Run tests

## Extending the System

### Add New ML Models

Place additional models in: `models/`

Update loader here: `src/ml/churn_predictor.py`

### Add Feature Engineering

`src/ml/feature_pipeline/`

### Add Orchestration / ETL Processes

`src/orchestration/`

### Add Docker Deployment

`docker/`

## Documentation

**Full Technical Design Document (TDD):** `docs/aineurolytics_churnguard.pdf`

**Includes:**

- system architecture
- ML methodology
- feature definitions
- API specifications
- RevOps alignment
- KPIs & metrics
- scaling strategy

## Roadmap

- ML model lifecycle management (MLflow / W&B)
- Feature store integration
- Async inference mode
- Docker & Kubernetes deployment
- Monitoring / Observability layer
- Batch scoring engine
- Additional unit/integration tests
- Advanced feature pipeline

## License

This project is proprietary and part of the **Aineurolytics Predictive Intelligence Suite**.  
All rights reserved.

## Contact

**Alex Rojas Segovia**  
Founder & Architect — Aineurolytics  
LinkedIn: [https://www.linkedin.com/in/alexrojas](https://www.linkedin.com/in/alexrojassegovia/)
