import json
import tarfile

import joblib
from sklearn.preprocessing import LabelEncoder

from inference.deploy_sagemaker_endpoint import build_endpoint_names, inference_image_uri, package_deploy_artifact


def test_build_endpoint_names_uses_base_name():
    names = build_endpoint_names("agent-risk-local-xgboost")

    assert names.model_name == "agent-risk-local-xgboost-model"
    assert names.endpoint_config_name == "agent-risk-local-xgboost-config"
    assert names.endpoint_name == "agent-risk-local-xgboost-endpoint"


def test_inference_image_uri_uses_xgboost_serving_container():
    assert inference_image_uri("ap-southeast-1") == "121021644041.dkr.ecr.ap-southeast-1.amazonaws.com/sagemaker-xgboost:1.7-1"


def test_package_deploy_artifact_includes_model_and_inference_code(tmp_path):
    source_model = tmp_path / "source_model"
    source_model.mkdir()
    (source_model / "xgboost_model.json").write_text("{}", encoding="utf-8")
    encoder = LabelEncoder().fit(["safe", "needs_review", "risky"])
    joblib.dump(encoder, source_model / "label_encoder.joblib")
    inference_file = tmp_path / "inference.py"
    inference_file.write_text("def model_fn(model_dir): pass", encoding="utf-8")
    policy_file = tmp_path / "decision_policy.py"
    policy_file.write_text("def decide(features, probabilities): pass", encoding="utf-8")
    output_tar = tmp_path / "model.tar.gz"

    package_deploy_artifact(source_model, inference_file, policy_file, output_tar)

    with tarfile.open(output_tar, "r:gz") as archive:
        assert sorted(archive.getnames()) == [
            "code/decision_policy.py",
            "code/inference.py",
            "label_classes.json",
            "xgboost_model.json",
        ]
        assert json.loads(archive.extractfile("label_classes.json").read().decode("utf-8")) == ["needs_review", "risky", "safe"]
