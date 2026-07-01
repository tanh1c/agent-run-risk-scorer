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
    return build_response(200, invoke_endpoint(endpoint_name, region, parse_event_body(event)))
