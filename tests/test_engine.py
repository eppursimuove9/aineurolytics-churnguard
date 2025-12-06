import pytest
from src.engine import ChurnPredictor
from pydantic import ValidationError

@pytest.fixture
def predictor():
    return ChurnPredictor()

def test_predictor_initialization(predictor):
    """Test if the mock model is initialized successfully."""
    assert predictor.model is not None
    assert "HIGH" in predictor.RISK_THRESHOLDS

def test_predictor_high_risk_classification(predictor):
    """Test case for data that should result in a HIGH churn risk."""
    high_risk_data = [{
        "customer_id": "test_high",
        "behavioral_score": 10.0,
        "transactional_value_k": 50.0,
        "support_ticket_volume_30d": 15,
        "engagement_score": 0.1,
        "contract_days_remaining": 30
    }]
    results = predictor.predict(high_risk_data)
    assert results[0]['churn_risk_level'] == 'HIGH'

def test_predictor_low_risk_classification(predictor):
    """Test case for data that should result in a LOW churn risk."""
    low_risk_data = [{
        "customer_id": "test_low",
        "behavioral_score": 95.0,
        "transactional_value_k": 1000.0,
        "support_ticket_volume_30d": 0,
        "engagement_score": 0.95,
        "contract_days_remaining": 300
    }]
    results = predictor.predict(low_risk_data)
    assert results[0]['churn_risk_level'] == 'LOW'
    
def test_predictor_invalid_input(predictor):
    """Test error handling for invalid input schema."""
    invalid_data = [{
        "customer_id": "test_invalid",
        "behavioral_score": "not_a_number", # Invalid type
        "transactional_value_k": 100.0,
        "support_ticket_volume_30d": 1,
        "engagement_score": 0.5,
        "contract_days_remaining": 50
    }]
    # Preprocess should raise ValueError due to ValidationError
    with pytest.raises(ValueError):
        predictor.preprocess(invalid_data)

