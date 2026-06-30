from agent.trajectory_logger import TrajectoryLogger


def test_trajectory_logger_collects_tool_results_and_flags():
    logger = TrajectoryLogger("Fix login validation bug", run_id="run_test")
    logger.append({"tool": "read_file", "status": "success", "path": "demo_repo/app/auth.py", "latency_ms": 10})
    logger.append({"tool": "edit_file", "status": "success", "path": "demo_repo/app/auth.py", "latency_ms": 20})
    logger.append({"tool": "run_tests", "status": "success", "command": "pytest demo_repo/tests/test_auth.py", "latency_ms": 30})

    trajectory = logger.finish("Fixed login validation and tests passed.")

    assert trajectory["run_id"] == "run_test"
    assert trajectory["source"] == "mini_llm_agent"
    assert trajectory["files_read"] == ["demo_repo/app/auth.py"]
    assert trajectory["files_modified"] == ["demo_repo/app/auth.py"]
    assert trajectory["commands_run"] == ["pytest demo_repo/tests/test_auth.py"]
    assert trajectory["tests_passed"] is True
    assert trajectory["touched_sensitive_files"] is False
    assert trajectory["summary_claim_supported"] is True
