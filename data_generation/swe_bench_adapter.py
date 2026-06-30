import argparse
import json
import re
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

try:
    from data_generation.label_rules import summary_claim_supported, touches_sensitive_file
except ModuleNotFoundError:
    from label_rules import summary_claim_supported, touches_sensitive_file

DATASET = "princeton-nlp/SWE-bench_Lite"
DATASET_SERVER = "https://datasets-server.huggingface.co"


FILE_PATTERN = re.compile(r"^diff --git a/(.+?) b/(.+)$")


def patch_files(patch: str) -> list[str]:
    files = []
    for line in patch.splitlines():
        match = FILE_PATTERN.match(line)
        if match:
            files.append(match.group(2))
    return list(dict.fromkeys(files))


def patch_line_counts(patch: str) -> tuple[int, int]:
    added = 0
    deleted = 0
    for line in patch.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            added += 1
        elif line.startswith("-"):
            deleted += 1
    return added, deleted


def convert_swe_bench_row(row: dict, index: int) -> dict:
    patch = row.get("patch") or ""
    test_patch = row.get("test_patch") or ""
    files_modified = patch_files(patch)
    test_files = patch_files(test_patch)
    diff_lines_added, diff_lines_deleted = patch_line_counts(patch)
    touched_sensitive = touches_sensitive_file(files_modified)
    label = "risky" if touched_sensitive else "require_review"
    final_summary = "External SWE-bench issue/patch sample converted into a pseudo trajectory."

    return {
        "run_id": f"swe_bench_lite_{index:06d}",
        "source": "swe_bench_lite",
        "external_id": row.get("instance_id", ""),
        "task": row.get("problem_statement") or row.get("issue") or "SWE-bench coding task",
        "tools_called": [],
        "files_read": list(dict.fromkeys(files_modified + test_files)),
        "files_modified": files_modified,
        "commands_run": [],
        "tests_passed": None,
        "lint_passed": None,
        "diff_lines_added": diff_lines_added,
        "diff_lines_deleted": diff_lines_deleted,
        "touched_sensitive_files": touched_sensitive,
        "used_network_command": False,
        "destructive_command_detected": False,
        "summary_claim_supported": summary_claim_supported(final_summary, [], None),
        "final_summary": final_summary,
        "label": label,
        "decision_hint": "external_dataset_sensitive_change" if touched_sensitive else "external_dataset_requires_review",
    }


def fetch_rows(limit: int, config: str, split: str) -> list[dict]:
    params = urlencode({"dataset": DATASET, "config": config, "split": split, "offset": 0, "length": limit})
    with urlopen(f"{DATASET_SERVER}/rows?{params}") as response:
        payload = json.loads(response.read().decode("utf-8"))
    return [item["row"] for item in payload["rows"]]


def write_jsonl(records: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--config", default="default")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default="data_generation/swe_bench_lite_trajectories.jsonl")
    args = parser.parse_args()

    rows = fetch_rows(args.limit, args.config, args.split)
    records = [convert_swe_bench_row(row, index + 1) for index, row in enumerate(rows)]
    write_jsonl(records, Path(args.output))


if __name__ == "__main__":
    main()
