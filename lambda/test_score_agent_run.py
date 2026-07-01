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


def test_handler_converts_trajectory_payload_to_features(monkeypatch):
    calls = []
    trajectory = {
        "source": "mini_llm_agent",
        "task": "Fix login validation bug",
        "tools_called": [{"tool": "read_file", "latency_ms": 10}, {"tool": "run_tests", "latency_ms": 20}],
        "files_read": ["demo_repo/app/auth.py"],
        "files_modified": ["demo_repo/app/auth.py"],
        "commands_run": ["pytest demo_repo/tests/test_auth.py"],
        "tests_passed": True,
        "lint_passed": None,
        "diff_lines_added": 1,
        "diff_lines_deleted": 1,
        "touched_sensitive_files": False,
        "destructive_command_detected": False,
        "used_network_command": False,
        "summary_claim_supported": True,
    }

    monkeypatch.setenv("SAGEMAKER_ENDPOINT_NAME", "agent-risk-local-xgboost-endpoint")
    monkeypatch.setattr(handler_module, "invoke_endpoint", lambda endpoint_name, region, payload: calls.append(payload) or {"decision": "allow"})

    response = handler({"body": json.dumps(trajectory)}, None)

    assert calls[0]["features"]["num_files_read"] == 1
    assert calls[0]["features"]["num_tools_called"] == 2
    assert calls[0]["features"]["diff_total_lines"] == 2
    assert calls[0]["features"]["source_mini_llm_agent"] is True
    assert json.loads(response["body"]) == {"decision": "allow"}
