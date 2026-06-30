import subprocess
import time
from pathlib import Path

from agent.tool_policy import is_command_allowed, is_edit_allowed, is_path_allowed, is_sensitive_path


def _result(tool: str, start: float, status: str, **values) -> dict:
    return {"tool": tool, "status": status, "latency_ms": int((time.perf_counter() - start) * 1000), **values}


def read_file(path: str) -> dict:
    start = time.perf_counter()
    if not is_path_allowed(path):
        return _result("read_file", start, "blocked", path=path, reason="Path is outside demo_repo")
    return _result("read_file", start, "success", path=path, output=Path(path).read_text(encoding="utf-8"))


def search_code(query: str) -> dict:
    start = time.perf_counter()
    matches = []
    for path in Path("demo_repo").rglob("*.py"):
        if is_sensitive_path(path):
            continue
        text = path.read_text(encoding="utf-8")
        if query in text:
            matches.append(str(path).replace("\\", "/"))
    return _result("search_code", start, "success", query=query, matches=matches)


def edit_file(path: str, old_text: str, new_text: str) -> dict:
    start = time.perf_counter()
    if not is_edit_allowed(path):
        return _result("edit_file", start, "blocked", path=path, reason="Path is not editable")
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    if old_text not in text:
        if new_text in text:
            return _result("edit_file", start, "success", path=path, diff_lines_added=0, diff_lines_deleted=0)
        return _result("edit_file", start, "failed", path=path, reason="Old text not found")
    file_path.write_text(text.replace(old_text, new_text, 1), encoding="utf-8")
    return _result("edit_file", start, "success", path=path, diff_lines_added=1, diff_lines_deleted=1)


def run_tests(command: str) -> dict:
    return _run_command("run_tests", command)


def run_linter(command: str) -> dict:
    return _run_command("run_linter", command)


def _run_command(tool: str, command: str) -> dict:
    start = time.perf_counter()
    if not is_command_allowed(tool, command):
        return _result(tool, start, "blocked", command=command, reason="Command is not allowed")
    completed = subprocess.run(command.split(), capture_output=True, text=True, timeout=60, check=False)
    return _result(tool, start, "success" if completed.returncode == 0 else "failed", command=command, output=completed.stdout + completed.stderr)


def git_diff() -> dict:
    start = time.perf_counter()
    completed = subprocess.run(["git", "diff", "--", "demo_repo"], capture_output=True, text=True, timeout=60, check=False)
    return _result("git_diff", start, "success" if completed.returncode == 0 else "failed", output=completed.stdout + completed.stderr)
