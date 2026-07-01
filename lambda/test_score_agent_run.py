import importlib
import json

handler_module = importlib.import_module("lambda.score_agent_run")
build_response = handler_module.build_response
parse_event_body = handler_module.parse_event_body
handler = handler_module.handler


def test_parse_event_body_accepts_api_gateway_json_body():
    payload = {"features": {"num_tools_called": 3}}

    assert parse_event_body({"body": json.dumps(payload)}) == payload


def test_parse_event_body_accepts_direct_lambda_payload():
    payload = {"features": {"num_tools_called": 3}}

    assert parse_event_body(payload) == payload


def test_build_response_returns_api_gateway_json_response():
    response = build_response(200, {"decision": "allow"})

    assert response["statusCode"] == 200
    assert response["headers"] == {"Content-Type": "application/json"}
    assert json.loads(response["body"]) == {"decision": "allow"}


def test_handler_invokes_configured_sagemaker_endpoint(monkeypatch):
    calls = []

    def fake_invoke_endpoint(endpoint_name, region, payload):
        calls.append((endpoint_name, region, payload))
        return {"decision": "require_review"}

    monkeypatch.setenv("SAGEMAKER_ENDPOINT_NAME", "agent-risk-local-xgboost-endpoint")
    monkeypatch.setenv("AWS_REGION", "ap-southeast-1")
    monkeypatch.setattr(handler_module, "invoke_endpoint", fake_invoke_endpoint)

    response = handler({"body": json.dumps({"features": {"num_tools_called": 3}})}, None)

    assert calls == [("agent-risk-local-xgboost-endpoint", "ap-southeast-1", {"features": {"num_tools_called": 3}})]
    assert json.loads(response["body"]) == {"decision": "require_review"}
