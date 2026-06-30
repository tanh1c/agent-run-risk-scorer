import argparse
import json
import random
from pathlib import Path

import pandas as pd

SOURCE_COLUMNS = ["source_simulator", "source_mini_llm_agent", "source_swe_bench_lite"]
TASK_KEYWORDS = {
    "login": ["auth", "login", "schemas"],
    "token": ["auth", "login", "schemas"],
    "health": ["main", "health"],
    "status": ["main", "status"],
    "schema": ["schemas", "auth"],
    "validation": ["auth", "schemas"],
}


def as_bool(value) -> int:
    return int(value is True)


def task_file_relevance_score(task: str, files_modified: list[str]) -> float:
    task_lower = task.lower()
    file_text = " ".join(files_modified).lower()
    matched_groups = [keywords for key, keywords in TASK_KEYWORDS.items() if key in task_lower]
    if not matched_groups:
        return 0.5
    return max(1.0 if any(keyword in file_text for keyword in keywords) else 0.0 for keywords in matched_groups)


def tool_sequence_valid(tools_called: list[dict], commands_run: list[str]) -> bool:
    tool_names = [tool.get("tool") for tool in tools_called]
    edited = "edit_file" in tool_names
    read_or_searched = "read_file" in tool_names or "search_code" in tool_names
    ran_tests = any("pytest" in command for command in commands_run)
    if edited and not read_or_searched:
        return False
    if ran_tests and "run_tests" not in tool_names:
        return False
    return True


def latency_total_ms(tools_called: list[dict]) -> int:
    return sum(int(tool.get("latency_ms") or 0) for tool in tools_called)


def extract_features(record: dict) -> dict:
    source = record.get("source", "simulator")
    tools_called = record.get("tools_called", [])
    files_modified = record.get("files_modified", [])
    commands_run = record.get("commands_run", [])
    row = {
        "num_files_read": len(record.get("files_read", [])),
        "num_files_modified": len(files_modified),
        "num_tools_called": len(tools_called),
        "num_commands_run": len(commands_run),
        "diff_total_lines": int(record.get("diff_lines_added") or 0) + int(record.get("diff_lines_deleted") or 0),
        "task_file_relevance_score": task_file_relevance_score(record.get("task", ""), files_modified),
        "latency_total_ms": latency_total_ms(tools_called),
        "tests_passed": as_bool(record.get("tests_passed")),
        "lint_passed": as_bool(record.get("lint_passed")),
        "touched_sensitive_files": as_bool(record.get("touched_sensitive_files")),
        "destructive_command_detected": as_bool(record.get("destructive_command_detected")),
        "used_network_command": as_bool(record.get("used_network_command")),
        "summary_claim_supported": as_bool(record.get("summary_claim_supported")),
        "tool_sequence_valid": as_bool(tool_sequence_valid(tools_called, commands_run)),
        "label": record["label"],
    }
    for column in SOURCE_COLUMNS:
        row[column] = int(column == f"source_{source}")
    return row


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def split_rows(rows: list[dict], seed: int) -> tuple[list[dict], list[dict], list[dict]]:
    shuffled = rows[:]
    random.Random(seed).shuffle(shuffled)
    train_end = int(len(shuffled) * 0.70)
    validation_end = int(len(shuffled) * 0.85)
    return shuffled[:train_end], shuffled[train_end:validation_end], shuffled[validation_end:]


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = read_jsonl(Path(args.input))
    rows = [extract_features(record) for record in records]
    train, validation, test = split_rows(rows, args.seed)
    output_dir = Path(args.output_dir)
    write_csv(train, output_dir / "train.csv")
    write_csv(validation, output_dir / "validation.csv")
    write_csv(test, output_dir / "test.csv")


if __name__ == "__main__":
    main()
