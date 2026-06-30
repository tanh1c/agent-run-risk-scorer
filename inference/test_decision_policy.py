from inference.decision_policy import calculate_risk_score, decide


PROBABILITIES = {
    "safe": 0.0,
    "require_review": 0.0,
    "wrong_tool": 0.0,
    "hallucinated_success": 0.0,
    "risky": 0.0,
    "failed": 0.0,
}


def probabilities(**overrides):
    values = PROBABILITIES.copy()
    values.update(overrides)
    return values


def test_calculate_risk_score_uses_weighted_class_probabilities():
    score = calculate_risk_score(probabilities(risky=0.5, hallucinated_success=0.2, require_review=0.3))
    assert score == 0.8


def test_decide_allows_low_risk_run():
    result = decide(
        {"tests_passed": True, "destructive_command_detected": False, "touched_sensitive_files": False},
        probabilities(safe=0.9, require_review=0.1),
    )
    assert result["decision"] == "allow"
    assert result["predicted_label"] == "safe"
    assert result["quality_score"] == 0.96


def test_decide_blocks_high_risk_run():
    result = decide(
        {"destructive_command_detected": False, "touched_sensitive_files": False},
        probabilities(risky=0.8, failed=0.2),
    )
    assert result["decision"] == "block"
    assert result["predicted_label"] == "risky"


def test_decide_hard_blocks_destructive_command():
    result = decide(
        {"destructive_command_detected": True, "touched_sensitive_files": False},
        probabilities(safe=1.0),
    )
    assert result["decision"] == "block"
    assert "Destructive command detected" in result["reasons"]


def test_decide_requires_review_for_sensitive_file_even_when_model_low_risk():
    result = decide(
        {"destructive_command_detected": False, "touched_sensitive_files": True},
        probabilities(safe=1.0),
    )
    assert result["decision"] == "require_review"
    assert "Sensitive file touched" in result["reasons"]
