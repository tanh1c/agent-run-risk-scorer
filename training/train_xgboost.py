import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


def split_features_label(path: str):
    data = pd.read_csv(path)
    return data.drop(columns=["label"]), data["label"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--model-dir", required=True)
    args = parser.parse_args()

    x_train, y_train_raw = split_features_label(args.train)
    x_validation, y_validation_raw = split_features_label(args.validation)

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train_raw)
    y_validation = encoder.transform(y_validation_raw)

    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_validation)

    print(classification_report(y_validation, predictions, target_names=encoder.classes_, zero_division=0))

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(model_dir / "xgboost_model.json")
    joblib.dump(encoder, model_dir / "label_encoder.joblib")


if __name__ == "__main__":
    main()
