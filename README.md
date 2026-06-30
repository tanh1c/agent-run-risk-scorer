# AI Coding Agent Risk Scoring on AWS SageMaker

This project builds an end-to-end ML/MLOps system for scoring the quality and risk of AI Coding Agent runs from trajectory logs.

## MVP Components

- Simulator-generated labeled trajectory logs
- Mini LLM Coding Agent demo
- SageMaker Processing and Training Jobs
- XGBoost risk scoring model
- SageMaker Model Registry and Endpoint
- Lambda + API Gateway scoring API
- CloudWatch and Model Monitor
- SageMaker Pipelines automation

## Local Commands

Run the demo app tests:

```bash
PYTHONPATH=demo_repo pytest demo_repo/tests -v
```

Generate sample trajectory logs:

```bash
python data_generation/generate_scenarios.py --count 1200 --output data_generation/sample_trajectories.jsonl --seed 42
```

Generate SWE-bench Lite pseudo trajectories and merge them with simulator logs:

```bash
python data_generation/swe_bench_adapter.py --limit 20 --output data_generation/swe_bench_lite_trajectories.jsonl
python data_generation/merge_trajectories.py --inputs data_generation/sample_trajectories.jsonl data_generation/swe_bench_lite_trajectories.jsonl --output data_generation/combined_trajectories.jsonl
```

Process trajectories into train/validation/test CSVs:

```bash
python preprocessing/processing_script.py --input data_generation/combined_trajectories.jsonl --output-dir data/processed
```

Train local models:

```bash
python training/train_sklearn.py --train data/processed/train.csv --validation data/processed/validation.csv --model-dir models
python training/train_xgboost.py --train data/processed/train.csv --validation data/processed/validation.csv --model-dir models
```

Evaluate the XGBoost model:

```bash
python training/evaluate_model.py --test data/processed/test.csv --model-dir models --output models/evaluation_report.json
```
