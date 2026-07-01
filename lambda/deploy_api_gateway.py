import argparse
import json
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ResourceNames:
    lambda_name: str
    role_name: str
    api_name: str


def build_resource_names(base_name: str) -> ResourceNames:
    return ResourceNames(
        lambda_name=f"{base_name}-lambda",
        role_name=f"{base_name}-lambda-role",
        api_name=f"{base_name}-api",
    )


def package_lambda(handler_file: Path, output_zip: Path) -> None:
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(handler_file, arcname="score_agent_run.py")


def ensure_lambda_role(role_name: str, region: str, endpoint_name: str) -> str:
    import boto3
    from botocore.exceptions import ClientError

    iam = boto3.client("iam", region_name=region)
    try:
        return iam.get_role(RoleName=role_name)["Role"]["Arn"]
    except ClientError as error:
        if error.response["Error"]["Code"] != "NoSuchEntity":
            raise

    role = iam.create_role(
        RoleName=role_name,
        AssumeRolePolicyDocument=json.dumps(
            {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {"Service": "lambda.amazonaws.com"},
                        "Action": "sts:AssumeRole",
                    }
                ],
            }
        ),
    )["Role"]
    iam.attach_role_policy(RoleName=role_name, PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole")
    account_id = boto3.client("sts", region_name=region).get_caller_identity()["Account"]
    iam.put_role_policy(
        RoleName=role_name,
        PolicyName="InvokeAgentRiskEndpoint",
        PolicyDocument=json.dumps(
            {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": "sagemaker:InvokeEndpoint",
                        "Resource": f"arn:aws:sagemaker:{region}:{account_id}:endpoint/{endpoint_name}",
                    }
                ],
            }
        ),
    )
    time.sleep(10)
    return role["Arn"]


def deploy_lambda(lambda_name: str, role_arn: str, endpoint_name: str, region: str, zip_path: Path) -> str:
    import boto3
    from botocore.exceptions import ClientError

    client = boto3.client("lambda", region_name=region)
    code = zip_path.read_bytes()
    try:
        arn = client.create_function(
            FunctionName=lambda_name,
            Runtime="python3.12",
            Role=role_arn,
            Handler="score_agent_run.handler",
            Code={"ZipFile": code},
            Timeout=30,
            Environment={"Variables": {"SAGEMAKER_ENDPOINT_NAME": endpoint_name}},
        )["FunctionArn"]
    except ClientError as error:
        if error.response["Error"]["Code"] != "ResourceConflictException":
            raise
        client.update_function_code(FunctionName=lambda_name, ZipFile=code)
        client.update_function_configuration(
            FunctionName=lambda_name,
            Runtime="python3.12",
            Handler="score_agent_run.handler",
            Timeout=30,
            Environment={"Variables": {"SAGEMAKER_ENDPOINT_NAME": endpoint_name}},
        )
        arn = client.get_function(FunctionName=lambda_name)["Configuration"]["FunctionArn"]
    return arn


def deploy_http_api(api_name: str, lambda_name: str, lambda_arn: str, region: str) -> str:
    import boto3
    from botocore.exceptions import ClientError

    apigw = boto3.client("apigatewayv2", region_name=region)
    account_id = boto3.client("sts", region_name=region).get_caller_identity()["Account"]
    api = apigw.create_api(Name=api_name, ProtocolType="HTTP", CorsConfiguration={"AllowOrigins": ["*"], "AllowMethods": ["POST"]})
    api_id = api["ApiId"]
    integration = apigw.create_integration(
        ApiId=api_id,
        IntegrationType="AWS_PROXY",
        IntegrationUri=lambda_arn,
        PayloadFormatVersion="2.0",
    )
    apigw.create_route(ApiId=api_id, RouteKey="POST /score-agent-run", Target=f"integrations/{integration['IntegrationId']}")
    apigw.create_stage(ApiId=api_id, StageName="$default", AutoDeploy=True)
    try:
        boto3.client("lambda", region_name=region).add_permission(
            FunctionName=lambda_name,
            StatementId=f"AllowInvokeFrom{api_id}",
            Action="lambda:InvokeFunction",
            Principal="apigateway.amazonaws.com",
            SourceArn=f"arn:aws:execute-api:{region}:{account_id}:{api_id}/*/*/score-agent-run",
        )
    except ClientError as error:
        if error.response["Error"]["Code"] != "ResourceConflictException":
            raise
    return api["ApiEndpoint"] + "/score-agent-run"


def deploy(base_name: str, endpoint_name: str, region: str, role_arn: str | None = None) -> str:
    names = build_resource_names(base_name)
    zip_path = Path("models") / "lambda" / f"{names.lambda_name}.zip"
    package_lambda(Path("lambda") / "score_agent_run.py", zip_path)
    role_arn = role_arn or ensure_lambda_role(names.role_name, region, endpoint_name)
    lambda_arn = deploy_lambda(names.lambda_name, role_arn, endpoint_name, region, zip_path)
    return deploy_http_api(names.api_name, names.lambda_name, lambda_arn, region)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-name", default="agent-risk-score")
    parser.add_argument("--endpoint-name", default="agent-risk-local-xgboost-endpoint")
    parser.add_argument("--region", default="ap-southeast-1")
    parser.add_argument("--role-arn")
    args = parser.parse_args()

    print(deploy(args.base_name, args.endpoint_name, args.region, args.role_arn))


if __name__ == "__main__":
    main()
