import tarfile

from training.local_xgboost_artifact import build_local_artifact_paths, package_model


def test_build_local_artifact_paths_uses_project_prefix():
    paths = build_local_artifact_paths("my-bucket", "local-xgboost")

    assert paths.model_prefix == "s3://my-bucket/agent-risk-scorer/models/local-xgboost"
    assert paths.model_tar_uri == "s3://my-bucket/agent-risk-scorer/models/local-xgboost/model.tar.gz"


def test_package_model_adds_xgboost_artifacts(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "xgboost_model.json").write_text("{}", encoding="utf-8")
    (model_dir / "label_encoder.joblib").write_bytes(b"encoder")
    output_tar = tmp_path / "model.tar.gz"

    package_model(model_dir, output_tar)

    with tarfile.open(output_tar, "r:gz") as archive:
        assert sorted(archive.getnames()) == ["label_encoder.joblib", "xgboost_model.json"]
