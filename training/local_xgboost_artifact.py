import argparse
import shutil
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path

PROJECT_PREFIX = "agent-risk-scorer"


@dataclass(frozen=True)
class LocalArtifactPaths:
    model_prefix: str
    model_tar_uri: str


def build_local_artifact_paths(bucket: str, model_name: str) -> LocalArtifactPaths:
    model_prefix = f"s3://{bucket}/{PROJECT_PREFIX}/models/{model_name}"
    return LocalArtifactPaths(model_prefix=model_prefix, model_tar_uri=f"{model_prefix}/model.tar.gz")


def split_s3_uri(s3_uri: str) -> tuple[str, str]:
    return tuple(s3_uri.removeprefix("s3://").split("/", 1))


def download_processed_data(processed_s3_uri: str, output_dir: Path, region: str) -> None:
    import boto3

    bucket, prefix = split_s3_uri(processed_s3_uri.rstrip("/"))
    s3 = boto3.client("s3", region_name=region)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ["train.csv", "validation.csv", "test.csv"]:
        s3.download_file(bucket, f"{prefix}/{name}", str(output_dir / name))


def package_model(model_dir: Path, output_tar: Path) -> None:
    with tarfile.open(output_tar, "w:gz") as archive:
        for name in ["xgboost_model.json", "label_encoder.joblib"]:
            archive.add(model_dir / name, arcname=name)


def upload_model_tar(model_tar: Path, model_tar_uri: str, region: str) -> None:
    import boto3

    bucket, key = split_s3_uri(model_tar_uri)
    boto3.client("s3", region_name=region).upload_file(str(model_tar), bucket, key)


def train_package_upload(bucket: str, processed_s3_uri: str, model_name: str, region: str, work_dir: Path) -> LocalArtifactPaths:
    paths = build_local_artifact_paths(bucket, model_name)
    if work_dir.exists():
        shutil.rmtree(work_dir)
    data_dir = work_dir / "data"
    model_dir = work_dir / "model"
    model_tar = work_dir / "model.tar.gz"
    download_processed_data(processed_s3_uri, data_dir, region)
    subprocess.run(
        [
            "python",
            "training/train_xgboost.py",
            "--train",
            str(data_dir / "train.csv"),
            "--validation",
            str(data_dir / "validation.csv"),
            "--model-dir",
            str(model_dir),
        ],
        check=True,
    )
    package_model(model_dir, model_tar)
    upload_model_tar(model_tar, paths.model_tar_uri, region)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--processed-s3-uri", required=True)
    parser.add_argument("--region", default="ap-southeast-1")
    parser.add_argument("--model-name", default="local-xgboost")
    parser.add_argument("--work-dir", default="models/local_xgboost_artifact")
    args = parser.parse_args()

    paths = train_package_upload(
        bucket=args.bucket,
        processed_s3_uri=args.processed_s3_uri,
        model_name=args.model_name,
        region=args.region,
        work_dir=Path(args.work_dir),
    )
    print(f"Model artifact: {paths.model_tar_uri}")


if __name__ == "__main__":
    main()
