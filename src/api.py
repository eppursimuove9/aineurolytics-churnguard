import logging
from typing import List
from fastapi import FastAPI, HTTPException, Request, Depends
from pydantic import BaseModel
from prometheus_client import start_http_server, Counter, Histogram
from src.engine import ChurnPredictor, PredictionInput

# --- Configuration ---
API_VERSION = "v1"
MODEL_VERSION = "1.0.0" # Placeholder
logger = logging.getLogger('ChurnGuardAPI')
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)

# --- Metrics Setup ---
REQUEST_COUNTER = Counter(
    'churnguard_api_request_total', 'Total number of API requests', 
    ['endpoint', 'http_status']
)
INFERENCE_LATENCY = Histogram(
    'churnguard_inference_latency_seconds', 'Inference latency distribution', 
    ['model_version']
)
PREDICTION_COUNT = Counter(
    'churnguard_prediction_count', 'Count of predictions by risk level',
    ['risk_level']
)

# --- Initialize ---
app = FastAPI(
    title="ChurnGuard™ Predictive RevOps API",
    version=API_VERSION,
    description="Low-latency API for real-time churn risk scoring."
)
predictor = ChurnPredictor()
# Start Prometheus client on a different port (e.g., 8001 for metrics)
# In production K8s, metrics port is typically handled by a sidecar or dedicated port
# start_http_server(8001) 

# --- Pydantic Schema for Response (aligned with TDD) ---
class PredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    churn_risk_level: str
    action_code: str
    top_churn_feature: str

# --- Dependency (Security/Auth Placeholder) ---
def verify_jwt(request: Request):
    """Mock JWT verification for scaffold."""
    auth_header = request.headers.get('Authorization')
    if auth_header is None or not auth_header.startswith('Bearer '):
        # In production, this would be a 401
        logger.warning("Missing/Invalid Authorization Header (Mock bypass).")
        # raise HTTPException(status_code=401, detail="Unauthorized: Missing or invalid JWT")
    return True

# --- API Routes ---

@app.get("/api/v1/health")
async def health_check():
    """Service health check."""
    return {"status": "OK", "model_version": MODEL_VERSION}

@app.post("/api/v1/predict", response_model=List[PredictionResponse])
async def predict_churn(
    data: List[PredictionInput], 
    auth_ok: bool = Depends(verify_jwt),
    request: Request = None
):
    """Endpoint for real-time churn risk prediction."""
    trace_id = request.headers.get("X-Request-ID", f"trace-{np.random.randint(10000, 99999)}")
    logger.info(f"[{trace_id}] Received prediction request for {len(data)} customers.")
    
    try:
        with INFERENCE_LATENCY.labels(model_version=MODEL_VERSION).time():
            predictions = predictor.predict([d.dict() for d in data])
        
        # Log metrics
        for p in predictions:
            if 'churn_risk_level' in p:
                PREDICTION_COUNT.labels(risk_level=p['churn_risk_level']).inc()
        
        REQUEST_COUNTER.labels(endpoint='/predict', http_status=200).inc()
        logger.info(f"[{trace_id}] Prediction complete. Status 200.")
        return predictions
    
    except ValueError as e:
        REQUEST_COUNTER.labels(endpoint='/predict', http_status=400).inc()
        logger.error(f"[{trace_id}] 400 Bad Request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        REQUEST_COUNTER.labels(endpoint='/predict', http_status=500).inc()
        logger.critical(f"[{trace_id}] 500 Internal Server Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error during inference.")

# Prometheus metrics endpoint
from starlette.responses import PlainTextResponse
from prometheus_client import generate_latest

@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    """Exposes Prometheus metrics."""
    return generate_latest()

# To run: uvicorn src.api:app --host 0.0.0.0 --port 8000
