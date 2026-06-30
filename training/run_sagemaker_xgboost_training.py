import argparse
import json
import time
from dataclasses import dataclass

PROJECT_PREFIX = "agent-risk-scorer"
TRAIN_PATH = "/opt/ml/input/data/train/train.csv"
VALIDATION_PATH = "/opt/ml/input/data/validation/validation.csv"
MODEL_DIR = "/opt/ml/model"


@dataclass(frozen=True)
class TrainingS3Paths:
    model_output: str
    code_prefix: str


def training_image_uri(region: str) -> str:
    return f"121021644041.dkr.ecr.{region}.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"


def build_training_paths(bucket: str, job_name: str) -> TrainingS3Paths:
    return TrainingS3Paths(
        model_output=f"s3://{bucket}/{PROJECT_PREFIX}/models/{job_name}",
        code_prefix=f"{PROJECT_PREFIX}/code/{job_name}",
    )


def training_request_config(processed_s3_uri: str, model_s3_uri: str) -> dict:
    processed = processed_s3_uri.rstrip("/")
    return {
        "train_s3_uri": f"{processed}/train.csv",
        "validation_s3_uri": f"{processed}/validation.csv",
        "model_s3_uri": model_s3_uri,
        "container_args": ["--train", TRAIN_PATH, "--validation", VALIDATION_PATH, "--model-dir", MODEL_DIR],
    }


def upload_training_script(bucket: str, code_prefix: str, region: str) -> str:
    import boto3

    key = f"{code_prefix}/train_xgboost.py"
    boto3.client("s3", region_name=region).upload_file("training/train_xgboost.py", bucket, key)
    return f"s3://{bucket}/{key}"


def run_training_job(bucket: str, role_arn: str, region: str, processed_s3_uri: str, instance_type: str, wait: bool) -> TrainingS3Paths:
    import boto3

    job_name = f"agent-risk-xgboost-{int(time.time())}"
    paths = build_training_paths(bucket, job_name)
    script_s3_uri = upload_training_script(bucket, paths.code_prefix, region)
    config = training_request_config(processed_s3_uri, paths.model_output)
    sm = boto3.client("sagemaker", region_name=region)
    sm.create_training_job(
        TrainingJobName=job_name,
        RoleArn=role_arn,
        AlgorithmSpecification={
            "TrainingImage": training_image_uri(region),
            "TrainingInputMode": "File",
            "ContainerEntrypoint": ["python3", "/opt/ml/input/data/code/train_xgboost.py"],
            "ContainerArguments": config["container_args"],
        },
        InputDataConfig=[
            {
                "ChannelName": "code",
                "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": script_s3_uri, "S3DataDistributionType": "FullyReplicated"}},
            },
            {
                "ChannelName": "train",
                "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": config["train_s3_uri"], "S3DataDistributionType": "FullyReplicated"}},
            },
            {
                "ChannelName": "validation",
                "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": config["validation_s3_uri"], "S3DataDistributionType": "FullyReplicated"}},
            },
        ],
        OutputDataConfig={"S3OutputPath": config["model_s3_uri"]},
        ResourceConfig={"InstanceType": instance_type, "InstanceCount": 1, "VolumeSizeInGB": 30},
        StoppingCondition={"MaxRuntimeInSeconds": 1800},
    )
    if wait:
        waiter = sm.get_waiter("training_job_completed_or_stopped")
        waiter.wait(TrainingJobName=job_name)
        description = sm.describe_training_job(TrainingJobName=job_name)
        if description["TrainingJobStatus"] != "Completed":
            raise RuntimeError(json.dumps({"status": description["TrainingJobStatus"], "failure_reason": description.get("FailureReason")}, indent=2))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--role-arn", required=True)
    parser.add_argument("--processed-s3-uri", required=True)
    parser.add_argument("--region", default="ap-southeast-1")
    parser.add_argument("--instance-type", default="ml.t3.medium")
    parser.add_argument("--no-wait", action="store_true")
    args = parser.parse_args()

    paths = run_training_job(
        bucket=args.bucket,
        role_arn=args.role_arn,
        region=args.region,
        processed_s3_uri=args.processed_s3_uri,
        instance_type=args.instance_type,
        wait=not args.no_wait,
    )
    print(f"Model output: {paths.model_output}")


if __name__ == "__main__":
    main()
