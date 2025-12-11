import pickle
import argparse
from sklearn.datasets import make_classification
from xgboost import XGBClassifier
from pathlib import Path

def train_dummy_model(save_path="models/dummy_xgb.pkl"):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    X, y = make_classification(
        n_samples=500,
        n_features=3,
        n_informative=3,
        n_redundant=0,
        random_state=42
    )

    model = XGBClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        use_label_encoder=False,
        eval_metric="logloss"
    )

    model.fit(X, y)

    with open(save_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Dummy model saved at {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-path", type=str, default="models/dummy_xgb.pkl")
    args = parser.parse_args()

    train_dummy_model(args.save_path)
