import json
import os
import shutil
import subprocess
import sys

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


def test_inference_module_imports_from_sagemaker_code_dir(tmp_path):
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    shutil.copy("inference/inference.py", code_dir / "inference.py")
    shutil.copy("inference/decision_policy.py", code_dir / "decision_policy.py")

    result = subprocess.run(
        [sys.executable, "-c", "import inference"],
        cwd=code_dir,
        env={**os.environ, "PYTHONPATH": str(code_dir)},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
