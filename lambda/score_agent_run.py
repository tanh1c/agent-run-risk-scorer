import json
import os


def parse_event_body(event: dict) -> dict:
    body = event.get("body")
    if body is None:
        return event
    return json.loads(body)


def build_response(status_code: int, body: dict) -> dict:
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def model_payload(payload: dict) -> dict:
    if "features" in payload:
        return payload
    tools_called = payload.get("tools_called", [])
    source = payload.get("source", "simulator")
    return {
        "features": {
            "num_files_read": len(payload.get("files_read", [])),
            "num_files_modified": len(payload.get("files_modified", [])),
            "num_tools_called": len(tools_called),
            "num_commands_run": len(payload.get("commands_run", [])),
            "diff_total_lines": int(payload.get("diff_lines_added") or 0) + int(payload.get("diff_lines_deleted") or 0),
            "task_file_relevance_score": 1.0,
            "latency_total_ms": sum(int(tool.get("latency_ms") or 0) for tool in tools_called),
            "tests_passed": bool(payload.get("tests_passed")),
            "lint_passed": bool(payload.get("lint_passed")),
            "touched_sensitive_files": bool(payload.get("touched_sensitive_files")),
            "destructive_command_detected": bool(payload.get("destructive_command_detected")),
            "used_network_command": bool(payload.get("used_network_command")),
            "summary_claim_supported": bool(payload.get("summary_claim_supported")),
            "tool_sequence_valid": True,
            "source_simulator": source == "simulator",
            "source_mini_llm_agent": source == "mini_llm_agent",
            "source_swe_bench_lite": source == "swe_bench_lite",
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


def handler(event, context):
    endpoint_name = os.environ["SAGEMAKER_ENDPOINT_NAME"]
    region = os.environ.get("AWS_REGION", "ap-southeast-1")
    return build_response(200, invoke_endpoint(endpoint_name, region, model_payload(parse_event_body(event))))
