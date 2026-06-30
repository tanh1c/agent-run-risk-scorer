import argparse
import json
import sys
from pathlib import Path

if __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from agent.llm_client import MockLLMClient
from agent.tools import edit_file, read_file, run_linter, run_tests, search_code
from agent.trajectory_logger import TrajectoryLogger

TOOLS = {
    "read_file": read_file,
    "search_code": search_code,
    "edit_file": edit_file,
    "run_tests": run_tests,
    "run_linter": run_linter,
}


def run_agent(task: str, output: str | Path, client: MockLLMClient | None = None) -> dict:
    logger = TrajectoryLogger(task)
    for step in (client or MockLLMClient()).plan(task):
        tool = TOOLS[step["tool"]]
        args = {key: value for key, value in step.items() if key != "tool"}
        logger.append(tool(**args))

    trajectory = logger.finish("Fixed login validation and tests passed.")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(trajectory, indent=2), encoding="utf-8")
    return trajectory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_agent(args.task, args.output)


if __name__ == "__main__":
    main()
