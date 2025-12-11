import pickle
from pathlib import Path
import numpy as np

class ChurnPredictor:
    """
    Core predictive class for ChurnGuard™.
    Loads an XGBoost model (or dummy model) and
    exposes a `predict(features: dict)` method.
    """

    def __init__(self, model_path: str = "models/dummy_xgb.pkl", model_version: str = "v0.0.1"):
        self.model_path = Path(model_path)
        self.model_version = model_version

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)

    def predict(self, features: dict):
        # Ensure deterministic feature ordering: sort keys alphabetically
        keys = sorted(features.keys())
        X = np.array([features[k] for k in keys]).reshape(1, -1)

        # If the model was trained on a different ordering/shape, adapt accordingly.
        prob = float(self.model.predict_proba(X)[0][1])
        risk_level = self._risk_bucket(prob)

        return {
            "churn_probability": prob,
            "churn_risk_level": risk_level,
            "model_version": self.model_version
        }

    @staticmethod
    def _risk_bucket(prob: float) -> str:
        if prob >= 0.70:
            return "HIGH"
        elif prob >= 0.35:
            return "MEDIUM"
        return "LOW"
