# Demo Evidence

## AWS resources

- Region: `ap-southeast-1`
- Account: `939169265033`
- S3 bucket: `s3://agent-risk-scorer-939169265033-ap-southeast-1`
- SageMaker execution role: `arn:aws:iam::939169265033:role/agent-risk-scorer-sagemaker-role`
- Lambda execution role: `arn:aws:iam::939169265033:role/agent-risk-score-lambda-role`
- API Gateway route: `POST /score-agent-run`
- API Gateway URL: `https://ajq5vvvw51.execute-api.ap-southeast-1.amazonaws.com/score-agent-run`

## Completed flow

1. Generated trajectory data locally from the simulator and mini agent format.
2. Ran SageMaker Processing to convert raw trajectories into tabular train/validation/test CSVs.
3. Tried SageMaker Training for XGBoost; account training quotas were `0` for available instance families, so the demo used local XGBoost training as a fallback.
4. Uploaded the trained local XGBoost artifact to S3.
5. Packaged the model with SageMaker inference code and deployed a temporary real-time SageMaker Endpoint.
6. Invoked the SageMaker Endpoint directly and received a scoring decision.
7. Deployed Lambda + API Gateway so external clients can call `POST /score-agent-run`.
8. Ran the mini agent against the API Gateway URL and stored the scored trajectory in `runs/run_login_api.json`.

## Key run evidence

### SageMaker Processing output

Processing job completed and wrote processed CSVs under:

```text
s3://agent-risk-scorer-939169265033-ap-southeast-1/agent-risk-scorer/processed/agent-risk-processing-1782829845/train.csv
s3://agent-risk-scorer-939169265033-ap-southeast-1/agent-risk-scorer/processed/agent-risk-processing-1782829845/validation.csv
s3://agent-risk-scorer-939169265033-ap-southeast-1/agent-risk-scorer/processed/agent-risk-processing-1782829845/test.csv
```

### Local fallback model artifact

SageMaker Training was blocked by instance quotas. The local fallback trained XGBoost and uploaded:

```text
s3://agent-risk-scorer-939169265033-ap-southeast-1/agent-risk-scorer/models/local-xgboost/model.tar.gz
```

### SageMaker Endpoint direct invoke

Temporary endpoint:

```text
agent-risk-local-xgboost-endpoint
```

Direct invoke returned:

```json
{
  "risk_score": 0.6003,
  "quality_score": 0.3997,
  "predicted_label": "failed",
  "decision": "require_review"
}
```

### API Gateway invoke

`POST /score-agent-run` returned HTTP `200` with:

```json
{
  "risk_score": 0.6003,
  "quality_score": 0.3997,
  "predicted_label": "failed",
  "decision": "require_review"
}
```

### Mini agent end-to-end run

Command shape:

```bash
python agent/agent_runner.py \
  --task "Fix login validation bug" \
  --output runs/run_login_api.json \
  --score-api-url "https://ajq5vvvw51.execute-api.ap-southeast-1.amazonaws.com/score-agent-run"
```

`runs/run_login_api.json` includes:

```json
{
  "score_response": {
    "risk_score": 0.6298,
    "quality_score": 0.3702,
    "predicted_label": "failed",
    "decision": "require_review"
  }
}
```

## Cleanup status

- SageMaker Endpoint, Endpoint Config, and Model for `agent-risk-local-xgboost` were cleaned up after demo runs.
- Lambda function and API Gateway remain deployed so the demo URL exists, but the URL will return an error unless the SageMaker Endpoint is redeployed first.
- S3 raw/processed/model artifacts remain for evidence and repeatable demos.
- IAM roles and policies remain for repeatable deployment.

## Demo limitation notes

- SageMaker Training could not run in this account because training instance quotas were `0`; the local training fallback keeps the MVP demonstrable while preserving the intended SageMaker workflow shape.
- The generated simulator labels are intentionally simple for the MVP, so model metrics can look unrealistically strong. Report safety metrics and false negatives, not accuracy alone.
- Endpoint resources should remain short-lived for cost control.
