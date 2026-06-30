from preprocessing.run_sagemaker_processing import build_s3_paths, processing_image_uri, processing_run_config


def test_build_s3_paths_uses_project_prefix_and_job_name():
    paths = build_s3_paths("my-bucket", "runs/input.jsonl", "agent-risk-processing-123")

    assert paths.raw_input == "s3://my-bucket/agent-risk-scorer/raw/input.jsonl"
    assert paths.processed_output == "s3://my-bucket/agent-risk-scorer/processed/agent-risk-processing-123"


def test_processing_image_uri_uses_sklearn_container_for_region():
    assert processing_image_uri("ap-southeast-1") == "121021644041.dkr.ecr.ap-southeast-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"


def test_processing_run_config_maps_container_paths():
    config = processing_run_config(
        input_s3_uri="s3://my-bucket/agent-risk-scorer/raw/input.jsonl",
        output_s3_uri="s3://my-bucket/agent-risk-scorer/processed/job-1",
        input_name="input.jsonl",
    )

    assert config["arguments"] == [
        "--input",
        "/opt/ml/processing/input/input.jsonl",
        "--output-dir",
        "/opt/ml/processing/output",
    ]
    assert config["input_source"] == "s3://my-bucket/agent-risk-scorer/raw/input.jsonl"
    assert config["output_destination"] == "s3://my-bucket/agent-risk-scorer/processed/job-1"
    assert config["input_destination"] == "/opt/ml/processing/input"
    assert config["output_source"] == "/opt/ml/processing/output"
