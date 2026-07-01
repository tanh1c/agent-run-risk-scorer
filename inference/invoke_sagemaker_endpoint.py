import argparse
import json


def build_sample_payload() -> dict:
    return {
        "features": {
            "num_files_read": 1,
            "num_files_modified": 1,
            "num_tools_called": 3,
            "num_commands_run": 1,
            "diff_total_lines": 4,
            "task_file_relevance_score": 1.0,
            "latency_total_ms": 500,
            "tests_passed": True,
            "lint_passed": True,
            "touched_sensitive_files": False,
            "destructive_command_detected": False,
            "used_network_command": False,
            "summary_claim_supported": True,
            "tool_sequence_valid": True,
            "source_simulator": False,
            "source_mini_llm_agent": True,
            "source_swe_bench_lite": False,
        }
    }


def invoke_endpoint(endpoint_name: str, region: str, payload: dict) -> dict:
    import boto3

    response = boto3.client("sagemaker-runtime", region_name=region).invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload),
    )
    return json.loads(response["Body"].read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint-name", default="agent-risk-local-xgboost-endpoint")
    parser.add_argument("--region", default="ap-southeast-1")
    args = parser.parse_args()

    result = invoke_endpoint(args.endpoint_name, args.region, build_sample_payload())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
