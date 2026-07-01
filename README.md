# End-to-End MLOps Laboratory

This repository implements the platform described in [PLAN.md](/home/guijas/gui/dev/mlflow-databricks-demo/PLAN.md). It is a self-contained local laboratory that mirrors a Databricks and MLflow-style MLOps architecture using only the Python standard library, so it runs in the current environment without external package installation.

## What is implemented

The workflow covers:

* synthetic event and delayed-label generation
* bronze and silver ingestion layers
* reusable feature engineering with historical feature snapshots
* offline and online feature stores
* point-in-time training dataset creation
* experiment tracking and model artifact logging
* model registry with `Champion`, `Challenger`, and `Candidate` aliases
* automatic challenger evaluation and promotion
* batch inference
* online inference through a REST endpoint
* monitoring for predictions, features, and data quality
* configurable retraining triggers
* end-to-end audit logging

## Structure

* `src/mlops_lab/`: implementation modules
* `config/defaults.json`: configuration-driven behavior
* `artifacts/`: generated lakehouse layers, model artifacts, monitoring, and audit outputs
* `notebooks/`: lightweight notebook entrypoint for Databricks-style exploration

## Run

```bash
PYTHONPATH=src python3 -m mlops_lab.orchestration
```

The command creates the local platform artifacts under `artifacts/` and prints a summary of the latest workflow run.

To start the REST prediction service after the workflow has created the online feature store:

```bash
PYTHONPATH=src python3 -m mlops_lab.serving
```

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"customer_id": "CUST-0001"}'
```

## Databricks alignment

This implementation preserves the same responsibilities you would map into Databricks assets:

* notebooks and jobs map to the orchestration stages
* Unity Catalog / Lakehouse layers map to the persisted bronze, silver, feature, inference, and monitoring datasets
* Feature Engineering and Feature Store map to the offline and online feature store modules
* MLflow Tracking and Registry map to the local experiment tracker and model registry artifacts
* Model Serving maps to the HTTP service in `serving.py`

The local abstractions are deliberately isolated so they can be replaced later with real Databricks and MLflow clients without rewriting the business flow.
