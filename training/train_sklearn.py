import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


def split_features_label(path: str):
    data = pd.read_csv(path)
    return data.drop(columns=["label"]), data["label"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--model-dir", required=True)
    args = parser.parse_args()

    x_train, y_train = split_features_label(args.train)
    x_validation, y_validation = split_features_label(args.validation)

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    model.fit(x_train, y_train)
    predictions = model.predict(x_validation)

    print(classification_report(y_validation, predictions, zero_division=0))
    print(confusion_matrix(y_validation, predictions, labels=sorted(y_train.unique())))

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "sklearn_baseline.joblib")


if __name__ == "__main__":
    main()
