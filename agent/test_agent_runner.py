import json

import agent.agent_runner as agent_runner
from agent.agent_runner import run_agent
from agent.llm_client import MockLLMClient


def test_mock_llm_client_plans_login_fix():
    client = MockLLMClient()

    steps = client.plan("Fix login validation bug")

    assert steps[0] == {"tool": "read_file", "path": "demo_repo/app/auth.py"}
    assert {"tool": "run_tests", "command": "pytest demo_repo/tests/test_auth.py"} in steps


def test_run_agent_writes_trajectory(tmp_path, monkeypatch):
    output = tmp_path / "run.json"

    monkeypatch.setitem(agent_runner.TOOLS, "read_file", lambda path: {"tool": "read_file", "status": "success", "path": path})
    monkeypatch.setitem(agent_runner.TOOLS, "edit_file", lambda path, old_text, new_text: {"tool": "edit_file", "status": "success", "path": path})
    monkeypatch.setitem(agent_runner.TOOLS, "run_tests", lambda command: {"tool": "run_tests", "status": "success", "command": command})

    trajectory = run_agent("Fix login validation bug", output)

    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved == trajectory
    assert trajectory["source"] == "mini_llm_agent"
    assert trajectory["task"] == "Fix login validation bug"
    assert trajectory["files_modified"] == ["demo_repo/app/auth.py"]
    assert trajectory["tests_passed"] is True
    assert trajectory["summary_claim_supported"] is True
