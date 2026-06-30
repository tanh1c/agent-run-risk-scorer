RISK_WEIGHTS = {
    "safe": 0.0,
    "require_review": 0.4,
    "wrong_tool": 0.7,
    "hallucinated_success": 0.9,
    "risky": 1.0,
    "failed": 0.6,
}


def calculate_risk_score(probabilities: dict[str, float]) -> float:
    return round(sum(probabilities.get(label, 0.0) * weight for label, weight in RISK_WEIGHTS.items()), 4)


def predicted_label(probabilities: dict[str, float]) -> str:
    return max(probabilities, key=probabilities.get)


def build_response(risk_score: float, decision: str, probabilities: dict[str, float], reasons: list[str]) -> dict:
    return {
        "risk_score": risk_score,
        "quality_score": round(1 - risk_score, 4),
        "predicted_label": predicted_label(probabilities),
        "decision": decision,
        "class_probabilities": probabilities,
        "reasons": reasons,
    }


def decide(features: dict, probabilities: dict[str, float]) -> dict:
    risk_score = calculate_risk_score(probabilities)
    reasons = []

    if features.get("destructive_command_detected"):
        reasons.append("Destructive command detected")
        return build_response(risk_score, "block", probabilities, reasons)

    if features.get("touched_sensitive_files"):
        reasons.append("Sensitive file touched")
        decision = "block" if risk_score >= 0.7 else "require_review"
        return build_response(risk_score, decision, probabilities, reasons)

    if risk_score < 0.3:
        decision = "allow"
    elif risk_score < 0.7:
        decision = "require_review"
    else:
        decision = "block"

    return build_response(risk_score, decision, probabilities, reasons)
