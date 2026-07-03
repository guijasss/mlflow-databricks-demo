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

## Batch Training And Inference

The repository also includes two executable entrypoints for a Databricks / MLflow model lifecycle based on a feature store table that contains a nullable boolean column `fraude`.

### `train.py`

Training loads only rows where `fraude IS NOT NULL`, trains a challenger model using the preprocessing from `src/pipeline.py`, registers it in MLflow / Databricks, compares it to the current `champion`, and promotes the challenger when it achieves a better validation metric. The primary promotion metric is `average_precision`.

Required inputs:

* `FEATURE_TABLE`: source feature store table
* `REGISTERED_MODEL_NAME`: registered model name, ideally `catalog.schema.model_name` in Unity Catalog

Optional inputs:

* `MLFLOW_EXPERIMENT_NAME`: MLflow experiment path/name
* `MLFLOW_TRACKING_URI`: defaults to `databricks`
* `MLFLOW_REGISTRY_URI`: defaults to `databricks-uc`
* `DATABRICKS_HOST`: workspace URL, for example `https://dbc-0d6287ce-5988.cloud.databricks.com`
* `DATABRICKS_SECRET_SCOPE`: secret scope used to fetch the Databricks PAT with `dbutils.secrets`
* `DATABRICKS_SECRET_KEY`: secret key that stores the Databricks PAT
* `MLFLOW_MODEL_ARTIFACT`: defaults to `fraud-model`
* `VALIDATION_FRACTION`: defaults to `0.2`
* `RANDOM_STATE`: defaults to `42`
* `FRAUD_THRESHOLD`: defaults to `0.5`
* `PROMOTION_METRIC`: defaults to `average_precision`

Example:

```bash
FEATURE_TABLE=main.risk.feature_store_transacoes \
REGISTERED_MODEL_NAME=main.risk.fraud_model \
MLFLOW_EXPERIMENT_NAME=/Shared/fraud-training \
DATABRICKS_HOST=https://dbc-0d6287ce-5988.cloud.databricks.com \
DATABRICKS_SECRET_SCOPE=mlops \
DATABRICKS_SECRET_KEY=databricks-pat \
python3 train.py
```

### `predict.py`

Inference loads only rows where `fraude IS NULL`, fetches a model from MLflow / Databricks by alias, applies the same transformations from `src/pipeline.py`, and writes predictions to the output table `model_output` by default.

Output columns:

* `prediction_timestamp`
* `model_name`
* `model_alias`
* `model_version`
* `id_transacao`
* `id_cliente`
* `fraud_probability`
* `predicted_fraude`
* `classificacao_modelo`

Required inputs:

* `FEATURE_TABLE`
* `REGISTERED_MODEL_NAME`

Optional inputs:

* `MODEL_OUTPUT_TABLE`: defaults to `model_output`
* `MODEL_ALIAS`: defaults to `champion`
* `MLFLOW_TRACKING_URI`: defaults to `databricks`
* `MLFLOW_REGISTRY_URI`: defaults to `databricks-uc`
* `DATABRICKS_HOST`: workspace URL
* `DATABRICKS_SECRET_SCOPE`: secret scope used to fetch the Databricks PAT with `dbutils.secrets`
* `DATABRICKS_SECRET_KEY`: secret key that stores the Databricks PAT
* `FRAUD_THRESHOLD`: defaults to `0.5`
* `MODEL_OUTPUT_WRITE_MODE`: `append` or `overwrite`, default `append`

Example:

```bash
FEATURE_TABLE=main.risk.feature_store_transacoes \
REGISTERED_MODEL_NAME=main.risk.fraud_model \
MODEL_OUTPUT_TABLE=main.risk.model_output \
DATABRICKS_HOST=https://dbc-0d6287ce-5988.cloud.databricks.com \
DATABRICKS_SECRET_SCOPE=mlops \
DATABRICKS_SECRET_KEY=databricks-pat \
python3 predict.py
```

## Databricks alignment

This implementation preserves the same responsibilities you would map into Databricks assets:

* notebooks and jobs map to the orchestration stages
* Unity Catalog / Lakehouse layers map to the persisted bronze, silver, feature, inference, and monitoring datasets
* Feature Engineering and Feature Store map to the offline and online feature store modules
* MLflow Tracking and Registry map to the local experiment tracker and model registry artifacts
* Model Serving maps to the HTTP service in `serving.py`

The local abstractions are deliberately isolated so they can be replaced later with real Databricks and MLflow clients without rewriting the business flow.
