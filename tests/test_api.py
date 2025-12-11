from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_predict():
    payload = {
        "customer_id": "123",
        "features": {"feature_0": 0.1, "feature_1": 0.2, "feature_2": 0.3}
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["customer_id"] == "123"
    assert "churn_probability" in body
