from src.ml.churn_predictor import ChurnPredictor

def test_predictor_output():
    predictor = ChurnPredictor()
    # features keys must be sorted inside ChurnPredictor; use 3 features matching dummy model
    result = predictor.predict({"feature_0": 0.1, "feature_1": 0.2, "feature_2": 0.3})

    assert "churn_probability" in result
    assert "churn_risk_level" in result
    assert "model_version" in result
    assert 0.0 <= result["churn_probability"] <= 1.0
