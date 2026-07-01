# Demo Script

## 1. Explain the architecture

```text
Mini LLM Agent / Simulator
  -> trajectory JSON
  -> S3 raw data
  -> SageMaker Processing
  -> processed tabular features
  -> XGBoost model artifact
  -> SageMaker Endpoint
  -> Lambda + API Gateway POST /score-agent-run
  -> scored agent run
```

Key point: the mini agent is only a controlled demo client. The project focus is risk scoring and the AWS MLOps workflow.

## 2. Show preprocessing on AWS

Show the S3 processed outputs:

```text
s3://agent-risk-scorer-939169265033-ap-southeast-1/agent-risk-scorer/processed/agent-risk-processing-1782829845/
```

Explain that SageMaker Processing converts JSON trajectory logs into ML features.

## 3. Explain training fallback

SageMaker Training was implemented, but this AWS account had `0` quota for tested training instances. For the demo, the same training script ran locally and uploaded the model artifact to S3:

```text
s3://agent-risk-scorer-939169265033-ap-southeast-1/agent-risk-scorer/models/local-xgboost/model.tar.gz
```

## 4. Deploy temporary SageMaker Endpoint

```bash
python inference/deploy_sagemaker_endpoint.py \
  --bucket agent-risk-scorer-939169265033-ap-southeast-1 \
  --role-arn arn:aws:iam::939169265033:role/agent-risk-scorer-sagemaker-role \
  --region ap-southeast-1 \
  --model-name agent-risk-local-xgboost \
  --instance-type ml.t2.medium
```

## 5. Invoke endpoint directly

```bash
python inference/invoke_sagemaker_endpoint.py \
  --endpoint-name agent-risk-local-xgboost-endpoint \
  --region ap-southeast-1
```

Expected shape:

```json
{
  "risk_score": 0.6003,
  "quality_score": 0.3997,
  "predicted_label": "failed",
  "decision": "require_review"
}
```

## 6. Call through API Gateway

API Gateway URL:

```text
https://ajq5vvvw51.execute-api.ap-southeast-1.amazonaws.com/score-agent-run
```

Run mini agent end-to-end:

```bash
python agent/agent_runner.py \
  --task "Fix login validation bug" \
  --output runs/run_login_api.json \
  --score-api-url "https://ajq5vvvw51.execute-api.ap-southeast-1.amazonaws.com/score-agent-run"
```

Show `runs/run_login_api.json`, especially `score_response.decision`.

## 7. Cleanup endpoint

```bash
python inference/deploy_sagemaker_endpoint.py \
  --bucket agent-risk-scorer-939169265033-ap-southeast-1 \
  --role-arn arn:aws:iam::939169265033:role/agent-risk-scorer-sagemaker-role \
  --region ap-southeast-1 \
  --model-name agent-risk-local-xgboost \
  --cleanup
```

Keep Lambda/API Gateway only if the demo URL is needed; otherwise delete them from the AWS Console to reduce idle resources.
