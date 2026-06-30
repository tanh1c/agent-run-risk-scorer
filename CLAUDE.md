# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context

This repository is currently in the planning phase for an AWS internship project: an end-to-end ML/MLOps system that scores the quality and risk of AI Coding Agent runs from trajectory logs.

The main project documents are:

- `plan_de_tai_5_ai_coding_agent_risk_scoring_aws.md` — Vietnamese project proposal and AWS mapping.
- `docs/plans/2026-06-07-ai-coding-agent-risk-scoring.md` — implementation plan and target repository structure.

The project focus is the ML risk scorer and AWS SageMaker workflow, not building a full coding assistant. The mini LLM Coding Agent is a controlled demo client and trajectory producer.

## Planned Architecture

The MVP uses two trajectory sources:

1. `simulator`: generates 500-2000 labeled trajectory logs for model training.
2. `mini_llm_agent`: runs controlled coding tasks in `demo_repo/` and sends the resulting trajectory to the deployed scorer.

High-level flow:

```text
Mini LLM Coding Agent / Simulator
  -> Trajectory Log JSON
  -> S3 raw logs
  -> SageMaker Processing Job
  -> processed tabular features
  -> SageMaker Training Jobs: Scikit-learn baseline + XGBoost main model
  -> SageMaker Experiments + HPO
  -> SageMaker Model Registry
  -> SageMaker Endpoint
  -> Lambda + API Gateway POST /score-agent-run
  -> CloudWatch + Model Monitor
  -> SageMaker Pipelines automation
```

Future Claude Code CLI or OpenCode CLI adapters are out of MVP scope; document them as future work unless the user explicitly asks to implement them.

## Target Code Structure

The implementation plan expects this structure:

```text
agent/              # mini LLM coding agent, tool policy, tools, trajectory logger
demo_repo/          # small FastAPI app used by the agent demo
data_generation/    # simulator and labeling rules
preprocessing/      # JSONL-to-feature processing scripts
training/           # Scikit-learn, XGBoost, and evaluation scripts
inference/          # SageMaker inference entrypoint and decision policy
lambda/             # Lambda handler for invoking SageMaker Endpoint
pipelines/          # SageMaker Pipeline definition
monitoring/         # Model Monitor and CloudWatch config
notebooks/          # SageMaker setup, processing, training, deployment notebooks
report/             # architecture, demo script, final report outline
```

## Common Commands

These commands come from the implementation plan and apply once the corresponding files are created.

Create the planned project skeleton:

```bash
mkdir -p agent demo_repo/app demo_repo/tests data_generation preprocessing training inference lambda pipelines monitoring notebooks report
```

Run the demo FastAPI test suite:

```bash
PYTHONPATH=demo_repo pytest demo_repo/tests -v
```

Run a single demo test:

```bash
PYTHONPATH=demo_repo pytest demo_repo/tests/test_auth.py::test_login_accepts_valid_token -v
```

Generate simulator trajectories:

```bash
python data_generation/generate_scenarios.py --count 1200 --output data_generation/sample_trajectories.jsonl --seed 42
```

Process trajectory JSONL into train/validation/test CSVs:

```bash
python preprocessing/processing_script.py --input data_generation/sample_trajectories.jsonl --output-dir data/processed
```

Train the local Scikit-learn baseline:

```bash
python training/train_sklearn.py --train data/processed/train.csv --validation data/processed/validation.csv --model-dir models
```

Train the local XGBoost model:

```bash
python training/train_xgboost.py --train data/processed/train.csv --validation data/processed/validation.csv --model-dir models
```

Evaluate the trained model:

```bash
python training/evaluate_model.py --test data/processed/test.csv --model-dir models --output models/evaluation_report.json
```

Run the mini agent locally:

```bash
python agent/agent_runner.py --task "Fix login validation bug" --output runs/run_login.json
```

Run the mini agent and score through API Gateway:

```bash
python agent/agent_runner.py --task "Fix login validation bug" --output runs/run_login.json --score-api-url "<api-gateway-url>/score-agent-run"
```

Invoke the scoring API directly:

```bash
curl -X POST "<api-gateway-url>/score-agent-run" \
  -H "Content-Type: application/json" \
  -d @runs/run_login.json
```

Create or update the SageMaker Pipeline:

```bash
python pipelines/sagemaker_pipeline.py --bucket <bucket> --role-arn <role-arn> --region <region>
```

## Implementation Notes

- Keep simulator data generation deterministic with a seed.
- The trajectory schema should remain agent-neutral so future adapters can map Claude Code/OpenCode logs into the same format.
- The mini agent must not expose a generic unrestricted shell command tool. Planned tools are `read_file`, `search_code`, `edit_file`, `run_tests`, `run_linter`, and `git_diff`.
- Tool policy should keep paths inside `demo_repo/`, block sensitive paths, and allow only `pytest ...` and `ruff check ...` command forms.
- Use XGBoost as the main model for tabular trajectory features; Scikit-learn is the baseline.
- Decisioning is hybrid: hard rule-based blocks for destructive/sensitive behavior plus ML-based `risk_score` thresholds.
- Safety metrics matter more than accuracy alone; report risky recall and risky false negative rate.

## AWS Components

The planned AWS workflow uses:

- S3 for raw logs, processed data, model artifacts, monitoring captures, and reports.
- SageMaker Processing Jobs for feature extraction and dataset splitting.
- SageMaker Training Jobs for baseline and XGBoost training.
- SageMaker Experiments and Automatic Model Tuning for comparison and HPO.
- SageMaker Model Registry for versioning the selected model.
- SageMaker Endpoint for real-time inference.
- Lambda and API Gateway for `POST /score-agent-run`.
- CloudWatch and SageMaker Model Monitor for logs, latency/error metrics, data capture, and drift monitoring.
- SageMaker Pipelines for Processing -> Training -> Evaluation -> Register -> optional Deploy.

## Cost Control

When working with AWS resources, keep endpoint and monitoring resources short-lived for demo use. The implementation plan expects cleanup documentation for SageMaker Endpoint, endpoint configuration, Model Monitor schedule, temporary S3 captures, CloudWatch retention, and active Studio apps.
