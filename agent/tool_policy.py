import shlex
from pathlib import Path

DEMO_ROOT = Path("demo_repo").resolve()
SENSITIVE_PARTS = {".env", ".github", "deploy", "secrets", "credential"}


def _resolved(path: str) -> Path:
    return Path(path).resolve()


def is_sensitive_path(path: str | Path) -> bool:
    lowered_parts = {part.lower() for part in Path(path).parts}
    return bool(lowered_parts & SENSITIVE_PARTS or "credential" in str(path).lower())


def is_path_allowed(path: str) -> bool:
    try:
        _resolved(path).relative_to(DEMO_ROOT)
    except ValueError:
        return False
    return not is_sensitive_path(path)


def is_edit_allowed(path: str) -> bool:
    return is_path_allowed(path)


def _path_args_allowed(args: list[str]) -> bool:
    path_args = [arg for arg in args if not arg.startswith("-")]
    return bool(path_args) and all(is_path_allowed(arg) for arg in path_args)


def is_command_allowed(tool: str, command: str) -> bool:
    try:
        parts = shlex.split(command)
    except ValueError:
        return False
    if tool == "run_tests" and parts[:1] == ["pytest"]:
        return _path_args_allowed(parts[1:])
    if tool == "run_linter" and parts[:2] == ["ruff", "check"]:
        return _path_args_allowed(parts[2:])
    return False
