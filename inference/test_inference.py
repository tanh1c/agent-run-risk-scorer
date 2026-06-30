import json

from inference.inference import input_fn, output_fn


def test_input_fn_accepts_application_json():
    payload = {"task": "Fix login bug", "tests_passed": True}
    result = input_fn(json.dumps(payload), "application/json")
    assert result == payload


def test_input_fn_rejects_non_json_content_type():
    try:
        input_fn("{}", "text/plain")
    except ValueError as error:
        assert "application/json" in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_output_fn_returns_json_response_body():
    prediction = {"decision": "allow", "risk_score": 0.1}
    result = output_fn(prediction, "application/json")
    assert json.loads(result) == prediction
