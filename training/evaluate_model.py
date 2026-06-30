import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score
from xgboost import XGBClassifier


def risky_false_negative_rate(y_true: list[str], y_pred: list[str]) -> float:
    risky_total = sum(label == "risky" for label in y_true)
    if risky_total == 0:
        return 0.0
    risky_false_negatives = sum(true == "risky" and pred != "risky" for true, pred in zip(y_true, y_pred))
    return risky_false_negatives / risky_total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = pd.read_csv(args.test)
    x_test = data.drop(columns=["label"])
    y_test = data["label"].tolist()

    model_dir = Path(args.model_dir)
    # Only load model artifacts produced by this project's training scripts.
    encoder = joblib.load(model_dir / "label_encoder.joblib")
    model = XGBClassifier()
    model.load_model(model_dir / "xgboost_model.json")

    predictions_encoded = model.predict(x_test)
    predictions = encoder.inverse_transform(predictions_encoded).tolist()
    labels = encoder.classes_.tolist()

    report = {
        "accuracy": accuracy_score(y_test, predictions),
        "macro_f1": f1_score(y_test, predictions, average="macro", zero_division=0),
        "risky_recall": recall_score(y_test, predictions, labels=["risky"], average="macro", zero_division=0),
        "risky_false_negative_rate": risky_false_negative_rate(y_test, predictions),
        "hallucinated_success_recall": recall_score(
            y_test,
            predictions,
            labels=["hallucinated_success"],
            average="macro",
            zero_division=0,
        ),
        "classification_report": classification_report(y_test, predictions, labels=labels, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels).tolist(),
        "labels": labels,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ["accuracy", "macro_f1", "risky_recall", "risky_false_negative_rate"]}, indent=2))


if __name__ == "__main__":
    main()
