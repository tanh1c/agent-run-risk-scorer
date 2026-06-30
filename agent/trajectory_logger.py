from data_generation.label_rules import has_destructive_command, summary_claim_supported, touches_sensitive_file


class TrajectoryLogger:
    def __init__(self, task: str, run_id: str = "run_local"):
        self.task = task
        self.run_id = run_id
        self.tools_called = []

    def append(self, result: dict) -> None:
        self.tools_called.append(result)

    def finish(self, final_summary: str) -> dict:
        files_read = [result["path"] for result in self.tools_called if result.get("tool") == "read_file" and result.get("path")]
        files_modified = [result["path"] for result in self.tools_called if result.get("tool") == "edit_file" and result.get("status") == "success" and result.get("path")]
        commands_run = [result["command"] for result in self.tools_called if result.get("command") and result.get("status") != "blocked"]
        tests = [result for result in self.tools_called if result.get("tool") == "run_tests"]
        lint = [result for result in self.tools_called if result.get("tool") == "run_linter"]
        return {
            "run_id": self.run_id,
            "source": "mini_llm_agent",
            "task": self.task,
            "tools_called": self.tools_called,
            "files_read": list(dict.fromkeys(files_read)),
            "files_modified": list(dict.fromkeys(files_modified)),
            "commands_run": commands_run,
            "tests_passed": tests[-1]["status"] == "success" if tests else None,
            "lint_passed": lint[-1]["status"] == "success" if lint else None,
            "diff_lines_added": sum(int(result.get("diff_lines_added") or 0) for result in self.tools_called),
            "diff_lines_deleted": sum(int(result.get("diff_lines_deleted") or 0) for result in self.tools_called),
            "touched_sensitive_files": touches_sensitive_file(files_modified),
            "used_network_command": any(command.lower().startswith(("curl ", "wget ")) for command in commands_run),
            "destructive_command_detected": has_destructive_command(commands_run),
            "summary_claim_supported": summary_claim_supported(final_summary, commands_run, tests[-1]["status"] == "success" if tests else None),
            "final_summary": final_summary,
        }
