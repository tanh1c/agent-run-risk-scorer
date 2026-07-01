import json
from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier

try:
    from inference.decision_policy import decide
except ModuleNotFoundError:
    from decision_policy import decide


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
        "labels": json.loads((model_path / "label_classes.json").read_text(encoding="utf-8")),
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
    return decide(features, dict(zip(model_artifacts["labels"], probabilities.tolist())))


def output_fn(prediction, response_content_type):
    if response_content_type != "application/json":
        raise ValueError("Only application/json response content type is supported")
    return json.dumps(prediction)
