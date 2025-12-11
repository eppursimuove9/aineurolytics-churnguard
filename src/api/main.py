from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
from src.ml.churn_predictor import ChurnPredictor

app = FastAPI(
    title="ChurnGuard™ Predictive API",
    version="0.0.1",
    description="Low-latency inference API for churn risk scoring."
)

# instantiate predictor (will read models/dummy_xgb.pkl)
predictor = ChurnPredictor()

class PredictRequest(BaseModel):
    customer_id: str
    features: dict

class PredictResponse(BaseModel):
    customer_id: str
    prediction_timestamp: str
    churn_probability: float
    churn_risk_level: str
    model_version: str


@app.get("/api/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.post("/api/v1/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    output = predictor.predict(req.features)
    return PredictResponse(
        customer_id=req.customer_id,
        prediction_timestamp=datetime.utcnow().isoformat(),
        **output
    )
