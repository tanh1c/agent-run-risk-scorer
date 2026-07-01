import importlib
import zipfile

deploy_module = importlib.import_module("lambda.deploy_api_gateway")
build_resource_names = deploy_module.build_resource_names
deploy = deploy_module.deploy
package_lambda = deploy_module.package_lambda


def test_build_resource_names_uses_project_prefix():
    names = build_resource_names("agent-risk-score")

    assert names.lambda_name == "agent-risk-score-lambda"
    assert names.role_name == "agent-risk-score-lambda-role"
    assert names.api_name == "agent-risk-score-api"


def test_package_lambda_includes_handler(tmp_path):
    handler_file = tmp_path / "score_agent_run.py"
    handler_file.write_text("def handler(event, context): pass", encoding="utf-8")
    output_zip = tmp_path / "lambda.zip"

    package_lambda(handler_file, output_zip)

    with zipfile.ZipFile(output_zip) as archive:
        assert archive.namelist() == ["score_agent_run.py"]


def test_deploy_uses_supplied_role_arn(monkeypatch):
    calls = []

    monkeypatch.setattr(deploy_module, "package_lambda", lambda handler_file, output_zip: None)
    monkeypatch.setattr(deploy_module, "ensure_lambda_role", lambda *args: calls.append("ensure-role"))
    monkeypatch.setattr(deploy_module, "deploy_lambda", lambda lambda_name, role_arn, endpoint_name, region, zip_path: calls.append(role_arn) or "lambda-arn")
    monkeypatch.setattr(deploy_module, "deploy_http_api", lambda *args: "https://example.com/score-agent-run")

    api_url = deploy("agent-risk-score", "agent-risk-local-xgboost-endpoint", "ap-southeast-1", "arn:aws:iam::123:role/lambda-role")

    assert api_url == "https://example.com/score-agent-run"
    assert calls == ["arn:aws:iam::123:role/lambda-role"]
