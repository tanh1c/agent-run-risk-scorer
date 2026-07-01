from inference.invoke_sagemaker_endpoint import build_sample_payload


def test_build_sample_payload_includes_required_feature_defaults():
    payload = build_sample_payload()

    assert payload["features"]["num_tools_called"] == 3
    assert payload["features"]["tests_passed"] is True
    assert payload["features"]["destructive_command_detected"] is False
    assert payload["features"]["source_mini_llm_agent"] is True
