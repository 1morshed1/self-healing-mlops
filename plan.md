## Project C: AI-Powered MLOps Pipeline with Self-Healing & Drift Detection

### Why This Project

You list MLflow, DVC, Dagshub, Airflow, and AstroCloud on your CV. You also list AWS SageMaker, Azure ML, and Vertex AI. But no project or work experience bullet demonstrates any of these tools in action. They're ghost skills right now.

More importantly, your work experience is entirely about building LLM-powered *applications*. You have no evidence of operating ML in production — monitoring model performance, detecting when a model starts degrading, retraining automatically, deploying without downtime. These are the concerns of an ML *engineer* (as opposed to an ML *developer*), and they're what senior roles pay for.

This project bridges your AI application skills with your listed-but-unproven DevOps and MLOps skills. It also has a unique angle: using an LLM to analyze pipeline failures and suggest fixes, which ties back to your core strength.

### What You Build

An end-to-end ML pipeline that trains a model, deploys it behind a FastAPI endpoint, monitors its performance in production, detects data drift and accuracy degradation, and triggers automated retraining — with an LLM-powered diagnostic agent that analyzes failures and suggests root causes.

### Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        KUBERNETES CLUSTER                                    │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │  Training Pipeline (Airflow DAG)                                     │   │
│  │                                                                      │   │
│  │  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────────┐  │   │
│  │  │  Data    │──▶│  Feature │──▶│  Train   │──▶│  Evaluate &     │  │   │
│  │  │  Ingest  │   │  Engine  │   │  Model   │   │  Register       │  │   │
│  │  │          │   │  ering   │   │  (scikit/ │   │  (MLflow)       │  │   │
│  │  │  (S3/DB) │   │          │   │  XGBoost) │   │                 │  │   │
│  │  └──────────┘   └──────────┘   └──────────┘   └────────┬─────────┘  │   │
│  │                                                        │             │   │
│  │                                                        │ if metrics  │   │
│  │                                                        │ pass gate   │   │
│  │                                                        ▼             │   │
│  │                                               ┌──────────────────┐   │   │
│  │                                               │  Deploy (canary  │   │   │
│  │                                               │  or blue/green)  │   │   │
│  │                                               │  via K8s         │   │   │
│  │                                               └──────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │  Serving Layer                                                       │   │
│  │                                                                      │   │
│  │  FastAPI endpoint:  POST /predict                                    │   │
│  │                                                                      │   │
│  │  - Loads model from MLflow registry                                  │   │
│  │  - Logs every prediction + input features to Kafka                  │   │
│  │  - Exposes Prometheus metrics (latency, throughput, error rate)      │   │
│  │  - Canary: routes 10% traffic to new model, 90% to current         │   │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │  Monitoring Layer                                                    │   │
│  │                                                                      │   │
│  │  ┌──────────────┐    ┌─────────────────┐    ┌──────────────────┐    │   │
│  │  │  Data Drift   │    │  Performance    │    │  System Health   │    │   │
│  │  │  Detector     │    │  Monitor        │    │  Monitor         │    │   │
│  │  │              │    │                 │    │                  │    │   │
│  │  │  - Feature   │    │  - Accuracy     │    │  - Latency p99  │    │   │
│  │  │    distribu- │    │    over time    │    │  - Memory usage  │    │   │
│  │  │    tion shift │    │  - Precision/   │    │  - GPU util      │    │   │
│  │  │  - PSI        │    │    Recall drift │    │  - Error rate    │    │   │
│  │  │  - KS test    │    │  - Prediction   │    │                  │    │   │
│  │  │  - Alerts     │    │    confidence   │    │  Prometheus +    │    │   │
│  │  │    when drift │    │    degradation  │    │  Grafana         │    │   │
│  │  │    exceeds    │    │                 │    │                  │    │   │
│  │  │    threshold  │    │                 │    │                  │    │   │
│  │  └──────┬───────┘    └────────┬────────┘    └──────────────────┘    │   │
│  │         │                     │                                      │   │
│  │         │  Drift/degradation detected                                │   │
│  │         ▼                     ▼                                      │   │
│  │  ┌──────────────────────────────────────────────────────────────┐    │   │
│  │  │  LLM Diagnostic Agent                                       │    │   │
│  │  │                                                              │    │   │
│  │  │  Receives alert context:                                     │    │   │
│  │  │  - Which features drifted and by how much                    │    │   │
│  │  │  - Performance metrics over time                             │    │   │
│  │  │  - Recent prediction distribution changes                    │    │   │
│  │  │  - Pipeline execution logs                                   │    │   │
│  │  │                                                              │    │   │
│  │  │  Produces:                                                   │    │   │
│  │  │  - Root cause analysis in natural language                   │    │   │
│  │  │  - Recommended action (retrain, rollback, investigate)       │    │   │
│  │  │  - If "retrain": triggers the training pipeline              │    │   │
│  │  │  - Posts summary to Slack/Discord                            │    │   │
│  │  └──────────────────────────────────────────────────────────────┘    │   │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  Storage: PostgreSQL (metadata), S3 (artifacts), MLflow (model registry)     │
│  Orchestration: Airflow (training DAGs), Kafka (prediction logging)          │
│  Observability: Prometheus + Grafana (dashboards), Loki (logs)               │
└──────────────────────────────────────────────────────────────────────────────┘
```

### The Unique Angle: LLM-Powered Diagnostics

This is what makes this more than a generic MLOps tutorial. When drift is detected, instead of just sending a Slack alert that says "Feature X drifted by 0.3 PSI," your system feeds the full context into an LLM agent that produces a human-readable diagnosis:

```python
class DiagnosticAgent:
    def analyze(self, alert: DriftAlert) -> Diagnosis:
        context = f"""
You are an MLOps diagnostic agent. A drift alert has been triggered.

Alert Type: {alert.type}
Model: {alert.model_name} (version {alert.model_version})
Time: {alert.timestamp}

Drift Details:
{self.format_drift_details(alert.drift_metrics)}

Performance Trend (last 7 days):
{self.format_performance_trend(alert.performance_history)}

Recent Prediction Distribution:
{self.format_prediction_distribution(alert.prediction_stats)}

Training Data Statistics (at training time):
{self.format_training_stats(alert.training_data_stats)}

Based on this information:
1. What is the most likely root cause of this drift?
2. How severe is this? (low / medium / high / critical)
3. What action should be taken? Choose one:
   - RETRAIN: Trigger retraining with recent data
   - ROLLBACK: Revert to the previous model version
   - INVESTIGATE: Alert a human, don't auto-remediate
   - IGNORE: This drift is within acceptable bounds
4. Explain your reasoning in 2-3 sentences.

Respond in JSON format.
"""
        
        response = self.llm.complete(context)
        diagnosis = json.loads(response)
        
        # Auto-remediate if the agent recommends it
        if diagnosis["action"] == "RETRAIN":
            self.trigger_retraining_dag()
        elif diagnosis["action"] == "ROLLBACK":
            self.rollback_model(alert.model_name, alert.previous_version)
        
        # Always notify
        self.send_notification(alert, diagnosis)
        
        return diagnosis
```

### Data Drift Detection Implementation

```python
from scipy import stats
import numpy as np

class DriftDetector:
    def __init__(self, reference_data: pd.DataFrame, thresholds: dict):
        self.reference = reference_data
        self.thresholds = thresholds
    
    def check_drift(self, production_data: pd.DataFrame) -> list[DriftResult]:
        results = []
        
        for feature in self.reference.columns:
            # Population Stability Index (PSI)
            psi = self.calculate_psi(
                self.reference[feature],
                production_data[feature]
            )
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_pvalue = stats.ks_2samp(
                self.reference[feature],
                production_data[feature]
            )
            
            drifted = (
                psi > self.thresholds.get("psi", 0.2) or
                ks_pvalue < self.thresholds.get("ks_pvalue", 0.01)
            )
            
            results.append(DriftResult(
                feature=feature,
                psi=psi,
                ks_statistic=ks_stat,
                ks_pvalue=ks_pvalue,
                drifted=drifted,
                severity="high" if psi > 0.25 else "medium" if psi > 0.1 else "low",
            ))
        
        return results
    
    def calculate_psi(self, reference: pd.Series, production: pd.Series, bins: int = 10) -> float:
        """Population Stability Index — measures how much the distribution has shifted."""
        ref_hist, bin_edges = np.histogram(reference, bins=bins)
        prod_hist, _ = np.histogram(production, bins=bin_edges)
        
        # Normalize to proportions (add small epsilon to avoid log(0))
        ref_pct = (ref_hist + 1e-6) / ref_hist.sum()
        prod_pct = (prod_hist + 1e-6) / prod_hist.sum()
        
        psi = np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct))
        return psi
```

### Grafana Dashboard Contents

Build a Grafana dashboard with these panels:
- **Model accuracy over time** — line chart, with a red line at the minimum acceptable threshold.
- **Feature drift heatmap** — a heatmap where each row is a feature and each column is a day. Color intensity = PSI value. This immediately shows which features are drifting and when.
- **Prediction confidence distribution** — histogram of model confidence scores. A shift toward lower confidence indicates degradation.
- **Retraining timeline** — annotations on the accuracy chart showing when retraining was triggered and whether it improved metrics.
- **LLM diagnostic log** — a table showing each diagnosis the LLM agent produced, its recommended action, and whether it was correct (manually labeled over time).

### What This Proves to Hiring Managers

- You can operate ML in production, not just build it.
- You know statistical methods for drift detection (PSI, KS test), not just vibes.
- You can build real Airflow DAGs, real Grafana dashboards, real K8s deployments — the DevOps skills on your CV are proven.
- The LLM diagnostic agent is a creative touch that ties back to your core AI expertise.

---
---