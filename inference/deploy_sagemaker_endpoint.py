import argparse
import json
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path

PROJECT_PREFIX = "agent-risk-scorer"


@dataclass(frozen=True)
class EndpointNames:
    model_name: str
    endpoint_config_name: str
    endpoint_name: str


def build_endpoint_names(base_name: str) -> EndpointNames:
    return EndpointNames(
        model_name=f"{base_name}-model",
        endpoint_config_name=f"{base_name}-config",
        endpoint_name=f"{base_name}-endpoint",
    )


def inference_image_uri(region: str) -> str:
    account_by_region = {"ap-southeast-1": "121021644041"}
    return f"{account_by_region[region]}.dkr.ecr.{region}.amazonaws.com/sagemaker-xgboost:1.7-1"


def package_deploy_artifact(source_model_dir: Path, inference_file: Path, policy_file: Path, output_tar: Path) -> None:
    import joblib

    output_tar.parent.mkdir(parents=True, exist_ok=True)
    label_classes = output_tar.parent / "label_classes.json"
    # Only convert label encoders produced locally by this project's training script.
    encoder = joblib.load(source_model_dir / "label_encoder.joblib")
    label_classes.write_text(json.dumps(encoder.classes_.tolist()), encoding="utf-8")
    with tarfile.open(output_tar, "w:gz") as archive:
        archive.add(source_model_dir / "xgboost_model.json", arcname="xgboost_model.json")
        archive.add(label_classes, arcname="label_classes.json")
        archive.add(inference_file, arcname="code/inference.py")
        archive.add(policy_file, arcname="code/decision_policy.py")


def split_s3_uri(s3_uri: str) -> tuple[str, str]:
    return tuple(s3_uri.removeprefix("s3://").split("/", 1))


def upload_artifact(local_tar: Path, bucket: str, model_name: str, region: str) -> str:
    import boto3

    key = f"{PROJECT_PREFIX}/models/{model_name}/deploy-model.tar.gz"
    boto3.client("s3", region_name=region).upload_file(str(local_tar), bucket, key)
    return f"s3://{bucket}/{key}"


def deploy_endpoint(bucket: str, role_arn: str, region: str, model_name: str, instance_type: str, local_model_dir: Path, wait: bool) -> EndpointNames:
    import boto3

    names = build_endpoint_names(model_name)
    work_dir = Path("models") / "deploy_artifact" / model_name
    if work_dir.exists():
        shutil.rmtree(work_dir)
    artifact = work_dir / "model.tar.gz"
    package_deploy_artifact(local_model_dir, Path("inference/inference.py"), Path("inference/decision_policy.py"), artifact)
    model_data_url = upload_artifact(artifact, bucket, model_name, region)
    sm = boto3.client("sagemaker", region_name=region)
    sm.create_model(
        ModelName=names.model_name,
        ExecutionRoleArn=role_arn,
        PrimaryContainer={
            "Image": inference_image_uri(region),
            "ModelDataUrl": model_data_url,
            "Environment": {"SAGEMAKER_PROGRAM": "inference.py", "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code"},
        },
    )
    sm.create_endpoint_config(
        EndpointConfigName=names.endpoint_config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": names.model_name,
                "InitialInstanceCount": 1,
                "InstanceType": instance_type,
            }
        ],
    )
    sm.create_endpoint(EndpointName=names.endpoint_name, EndpointConfigName=names.endpoint_config_name)
    if wait:
        waiter = sm.get_waiter("endpoint_in_service")
        waiter.wait(EndpointName=names.endpoint_name)
    return names


def cleanup_endpoint(endpoint_name: str, endpoint_config_name: str, model_name: str, region: str) -> None:
    import boto3
    from botocore.exceptions import ClientError

    sm = boto3.client("sagemaker", region_name=region)
    for action, kwargs in [
        (sm.delete_endpoint, {"EndpointName": endpoint_name}),
        (sm.delete_endpoint_config, {"EndpointConfigName": endpoint_config_name}),
        (sm.delete_model, {"ModelName": model_name}),
    ]:
        try:
            action(**kwargs)
        except ClientError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--role-arn", required=True)
    parser.add_argument("--region", default="ap-southeast-1")
    parser.add_argument("--model-name", default="agent-risk-local-xgboost")
    parser.add_argument("--instance-type", default="ml.t2.medium")
    parser.add_argument("--local-model-dir", default="models/local_xgboost_artifact/model")
    parser.add_argument("--cleanup", action="store_true")
    parser.add_argument("--no-wait", action="store_true")
    args = parser.parse_args()

    names = build_endpoint_names(args.model_name)
    if args.cleanup:
        cleanup_endpoint(names.endpoint_name, names.endpoint_config_name, names.model_name, args.region)
        print(json.dumps(names.__dict__, indent=2))
        return

    names = deploy_endpoint(
        bucket=args.bucket,
        role_arn=args.role_arn,
        region=args.region,
        model_name=args.model_name,
        instance_type=args.instance_type,
        local_model_dir=Path(args.local_model_dir),
        wait=not args.no_wait,
    )
    print(json.dumps(names.__dict__, indent=2))


if __name__ == "__main__":
    main()
