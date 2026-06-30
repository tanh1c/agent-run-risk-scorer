import argparse
from pathlib import Path


def merge_jsonl(inputs: list[str], output: str) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as destination:
        for input_path in inputs:
            with Path(input_path).open(encoding="utf-8") as source:
                for line in source:
                    if line.strip():
                        destination.write(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    merge_jsonl(args.inputs, args.output)


if __name__ == "__main__":
    main()
