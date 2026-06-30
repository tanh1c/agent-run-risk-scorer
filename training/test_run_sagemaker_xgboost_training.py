from training.run_sagemaker_xgboost_training import build_training_paths, training_image_uri, training_request_config


def test_build_training_paths_uses_project_model_prefix():
    paths = build_training_paths("my-bucket", "xgb-job-123")

    assert paths.model_output == "s3://my-bucket/agent-risk-scorer/models/xgb-job-123"
    assert paths.code_prefix == "agent-risk-scorer/code/xgb-job-123"


def test_training_image_uri_uses_sklearn_container_for_region():
    assert training_image_uri("ap-southeast-1") == "121021644041.dkr.ecr.ap-southeast-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"


def test_training_request_config_maps_processed_csv_channels():
    config = training_request_config(
        processed_s3_uri="s3://my-bucket/agent-risk-scorer/processed/job-1",
        model_s3_uri="s3://my-bucket/agent-risk-scorer/models/xgb-job-1",
    )

    assert config["train_s3_uri"] == "s3://my-bucket/agent-risk-scorer/processed/job-1/train.csv"
    assert config["validation_s3_uri"] == "s3://my-bucket/agent-risk-scorer/processed/job-1/validation.csv"
    assert config["model_s3_uri"] == "s3://my-bucket/agent-risk-scorer/models/xgb-job-1"
    assert config["container_args"] == [
        "--train",
        "/opt/ml/input/data/train/train.csv",
        "--validation",
        "/opt/ml/input/data/validation/validation.csv",
        "--model-dir",
        "/opt/ml/model",
    ]
