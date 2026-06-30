import argparse
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_PREFIX = "agent-risk-scorer"
INPUT_DIR = "/opt/ml/processing/input"
OUTPUT_DIR = "/opt/ml/processing/output"


@dataclass(frozen=True)
class ProcessingS3Paths:
    raw_input: str
    processed_output: str


def build_s3_paths(bucket: str, input_path: str, job_name: str) -> ProcessingS3Paths:
    input_name = Path(input_path).name
    return ProcessingS3Paths(
        raw_input=f"s3://{bucket}/{PROJECT_PREFIX}/raw/{input_name}",
        processed_output=f"s3://{bucket}/{PROJECT_PREFIX}/processed/{job_name}",
    )


def processing_image_uri(region: str) -> str:
    return f"121021644041.dkr.ecr.{region}.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"


def processing_run_config(input_s3_uri: str, output_s3_uri: str, input_name: str) -> dict:
    return {
        "arguments": ["--input", f"{INPUT_DIR}/{input_name}", "--output-dir", OUTPUT_DIR],
        "input_source": input_s3_uri,
        "input_destination": INPUT_DIR,
        "output_source": OUTPUT_DIR,
        "output_destination": output_s3_uri,
    }


def upload_input(input_path: Path, s3_uri: str, region: str) -> None:
    import boto3

    bucket, key = s3_uri.removeprefix("s3://").split("/", 1)
    boto3.client("s3", region_name=region).upload_file(str(input_path), bucket, key)


def run_processing_job(bucket: str, role_arn: str, region: str, input_path: Path, instance_type: str, wait: bool) -> ProcessingS3Paths:
    import boto3
    from sagemaker.core.helper.session_helper import Session
    from sagemaker.core.processing import ScriptProcessor
    from sagemaker.core.shapes import ProcessingInput, ProcessingOutput, ProcessingS3Input, ProcessingS3Output

    job_name = f"agent-risk-processing-{int(time.time())}"
    paths = build_s3_paths(bucket, str(input_path), job_name)
    upload_input(input_path, paths.raw_input, region)
    config = processing_run_config(paths.raw_input, paths.processed_output, input_path.name)
    session = Session(boto_session=boto3.Session(region_name=region))
    processor = ScriptProcessor(
        image_uri=processing_image_uri(region),
        command=["python3"],
        role=role_arn,
        instance_count=1,
        instance_type=instance_type,
        sagemaker_session=session,
        base_job_name="agent-risk-processing",
    )
    processor.run(
        code="preprocessing/processing_script.py",
        inputs=[
            ProcessingInput(
                input_name="raw-trajectories",
                s3_input=ProcessingS3Input(
                    s3_uri=config["input_source"],
                    s3_data_type="S3Prefix",
                    local_path=config["input_destination"],
                    s3_input_mode="File",
                ),
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="processed-data",
                s3_output=ProcessingS3Output(
                    s3_uri=config["output_destination"],
                    local_path=config["output_source"],
                    s3_upload_mode="EndOfJob",
                ),
            )
        ],
        arguments=config["arguments"],
        job_name=job_name,
        wait=wait,
        logs=wait,
    )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--role-arn", required=True)
    parser.add_argument("--region", default="ap-southeast-1")
    parser.add_argument("--input", default="data_generation/combined_trajectories.jsonl")
    parser.add_argument("--instance-type", default="ml.t3.medium")
    parser.add_argument("--no-wait", action="store_true")
    args = parser.parse_args()

    paths = run_processing_job(
        bucket=args.bucket,
        role_arn=args.role_arn,
        region=args.region,
        input_path=Path(args.input),
        instance_type=args.instance_type,
        wait=not args.no_wait,
    )
    print(f"Input: {paths.raw_input}")
    print(f"Output: {paths.processed_output}")


if __name__ == "__main__":
    main()
