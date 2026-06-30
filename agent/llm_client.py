class MockLLMClient:
    def plan(self, task: str) -> list[dict]:
        if "login" in task.lower():
            return [
                {"tool": "read_file", "path": "demo_repo/app/auth.py"},
                {
                    "tool": "edit_file",
                    "path": "demo_repo/app/auth.py",
                    "old_text": "return token == \"valid-token\"",
                    "new_text": "return bool(token.strip()) and token == \"valid-token\"",
                },
                {"tool": "run_tests", "command": "pytest demo_repo/tests/test_auth.py"},
            ]
        return [
            {"tool": "search_code", "query": task.split()[0] if task.split() else task},
            {"tool": "run_tests", "command": "pytest demo_repo/tests"},
        ]
