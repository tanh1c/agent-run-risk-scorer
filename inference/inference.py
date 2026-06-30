import json
from pathlib import Path

import joblib
import pandas as pd
from xgboost import XGBClassifier

from inference.decision_policy import decide


FEATURE_COLUMNS = [
    "num_files_read",
    "num_files_modified",
    "num_tools_called",
    "num_commands_run",
    "diff_total_lines",
    "task_file_relevance_score",
    "latency_total_ms",
    "tests_passed",
    "lint_passed",
    "touched_sensitive_files",
    "destructive_command_detected",
    "used_network_command",
    "summary_claim_supported",
    "tool_sequence_valid",
    "source_simulator",
    "source_mini_llm_agent",
    "source_swe_bench_lite",
]


def model_fn(model_dir):
    model_path = Path(model_dir)
    model = XGBClassifier()
    model.load_model(model_path / "xgboost_model.json")
    return {
        "model": model,
        # Only load model artifacts produced by this project's training scripts.
        "label_encoder": joblib.load(model_path / "label_encoder.joblib"),
    }


def input_fn(request_body, request_content_type):
    if request_content_type != "application/json":
        raise ValueError("Only application/json content type is supported")
    if isinstance(request_body, bytes):
        request_body = request_body.decode("utf-8")
    return json.loads(request_body)


def predict_fn(input_data, model_artifacts):
    features = input_data["features"] if "features" in input_data else input_data
    row = {column: features.get(column, 0) for column in FEATURE_COLUMNS}
    probabilities = model_artifacts["model"].predict_proba(pd.DataFrame([row]))[0]
    labels = model_artifacts["label_encoder"].inverse_transform(range(len(probabilities)))
    return decide(features, dict(zip(labels, probabilities.tolist())))


def output_fn(prediction, response_content_type):
    if response_content_type != "application/json":
        raise ValueError("Only application/json response content type is supported")
    return json.dumps(prediction)
