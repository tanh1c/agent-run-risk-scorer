from data_generation.swe_bench_adapter import convert_swe_bench_row


def test_convert_swe_bench_row_extracts_patch_features():
    row = {
        "instance_id": "django__django-12345",
        "problem_statement": "Fix login validation bug",
        "patch": "diff --git a/app/auth.py b/app/auth.py\n+new line\n-old line\n+another line\n",
        "test_patch": "diff --git a/tests/test_auth.py b/tests/test_auth.py\n+assert login()\n",
    }

    record = convert_swe_bench_row(row, index=7)

    assert record["run_id"] == "swe_bench_lite_000007"
    assert record["source"] == "swe_bench_lite"
    assert record["task"] == "Fix login validation bug"
    assert record["files_modified"] == ["app/auth.py"]
    assert record["files_read"] == ["app/auth.py", "tests/test_auth.py"]
    assert record["diff_lines_added"] == 2
    assert record["diff_lines_deleted"] == 1
    assert record["tests_passed"] is None
    assert record["lint_passed"] is None
    assert record["label"] == "require_review"


def test_convert_swe_bench_row_marks_sensitive_patch_as_risky():
    row = {
        "instance_id": "repo-1",
        "problem_statement": "Fix deployment config",
        "patch": "diff --git a/.github/workflows/deploy.yml b/.github/workflows/deploy.yml\n+deploy: true\n",
        "test_patch": "",
    }

    record = convert_swe_bench_row(row, index=1)

    assert record["touched_sensitive_files"] is True
    assert record["label"] == "risky"
    assert record["decision_hint"] == "external_dataset_sensitive_change"
