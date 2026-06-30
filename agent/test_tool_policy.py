from agent.tool_policy import is_command_allowed, is_edit_allowed, is_path_allowed
from agent.tools import read_file, search_code


def test_path_policy_allows_demo_repo_files():
    assert is_path_allowed("demo_repo/app/auth.py") is True


def test_path_policy_blocks_files_outside_demo_repo():
    assert is_path_allowed("README.md") is False
    assert is_path_allowed("../secrets.txt") is False


def test_path_policy_blocks_sensitive_paths():
    assert is_path_allowed("demo_repo/.env") is False
    assert is_path_allowed("demo_repo/.github/workflows/deploy.yml") is False
    assert is_path_allowed("demo_repo/secrets/config.py") is False


def test_read_file_blocks_sensitive_paths(tmp_path):
    secret = tmp_path / "demo_repo" / "secrets" / "config.py"
    secret.parent.mkdir(parents=True)
    secret.write_text("TOKEN = 'secret'", encoding="utf-8")

    assert read_file(str(secret))["status"] == "blocked"


def test_search_code_skips_sensitive_paths(tmp_path, monkeypatch):
    repo = tmp_path / "demo_repo"
    (repo / "app").mkdir(parents=True)
    (repo / "secrets").mkdir()
    (repo / "app" / "main.py").write_text("TOKEN = 'public'", encoding="utf-8")
    (repo / "secrets" / "config.py").write_text("TOKEN = 'secret'", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    result = search_code("TOKEN")

    assert "demo_repo/app/main.py" in result["matches"]
    assert all("secrets" not in match for match in result["matches"])


def test_edit_policy_blocks_sensitive_paths():
    assert is_edit_allowed("demo_repo/.env") is False
    assert is_edit_allowed("demo_repo/.github/workflows/deploy.yml") is False


def test_command_policy_allows_only_pytest_and_ruff_check():
    assert is_command_allowed("run_tests", "pytest demo_repo/tests -v") is True
    assert is_command_allowed("run_linter", "ruff check demo_repo") is True
    assert is_command_allowed("run_tests", "rm -rf demo_repo") is False
    assert is_command_allowed("run_linter", "ruff format demo_repo") is False


def test_command_policy_blocks_paths_outside_demo_repo():
    assert is_command_allowed("run_tests", "pytest ../other_repo/tests") is False
    assert is_command_allowed("run_tests", "pytest README.md") is False
    assert is_command_allowed("run_linter", "ruff check ..") is False
    assert is_command_allowed("run_linter", "ruff check demo_repo ../other_repo") is False
