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

The training flow is audit-oriented:

* reads a specific Delta snapshot of the feature table
* creates `train`, `validation`, and `test` splits
* logs the three splits in MLflow with dataset lineage
* optionally persists full split snapshots to a Databricks Volume for exact replay
* tags the registered model version with source table version, feature-set digest, and split manifest path

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
* `TEST_FRACTION`: defaults to `0.1`
* `RANDOM_STATE`: defaults to `42`
* `FRAUD_THRESHOLD`: defaults to `0.5`
* `PROMOTION_METRIC`: defaults to `average_precision`
* `FEATURE_TABLE_VERSION`: optional Delta version to replay a previous snapshot
* `TRAINING_SPLIT_VOLUME_PATH`: optional Volume path such as `/Volumes/main/mlops/audit/fraud_splits`

Example:

```bash
FEATURE_TABLE=main.risk.feature_store_transacoes \
REGISTERED_MODEL_NAME=main.risk.fraud_model \
MLFLOW_EXPERIMENT_NAME=/Shared/fraud-training \
TRAINING_SPLIT_VOLUME_PATH=/Volumes/main/mlops/audit/fraud_splits \
DATABRICKS_HOST=https://dbc-0d6287ce-5988.cloud.databricks.com \
DATABRICKS_SECRET_SCOPE=mlops \
DATABRICKS_SECRET_KEY=databricks-pat \
python3 train.py
```

Training writes:

* MLflow dataset inputs with contexts `training`, `validation`, and `testing`
* MLflow artifacts `audit/feature_spec.json` and `audit/split_snapshot_manifest.json`
* Volume snapshots under `TRAINING_SPLIT_VOLUME_PATH/run_id=<mlflow-run-id>/`

### `predict.py`

Inference loads only rows where `fraude IS NULL`, fetches a model from MLflow / Databricks by alias, applies the same transformations from `src/pipeline.py`, and writes predictions to the output table `model_output` by default.

Output columns:

* `prediction_timestamp`
* `model_name`
* `model_alias`
* `model_version`
* `model_source_run_id`
* `feature_table`
* `feature_table_version`
* `feature_table_timestamp`
* `feature_set_digest`
* `feature_vector_digest`
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
* `FEATURE_TABLE_VERSION`: optional Delta version to replay a previous snapshot

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

### Databricks Model Serving Audit

For online serving, deploy the registered model in Databricks Model Serving and enable inference tables on the endpoint. The recommended pattern is:

* serve the Unity Catalog model version selected by alias such as `champion`
* enable inference tables in Unity Catalog
* send a stable business identifier such as `id_transacao` as `client_request_id`
* join the inference table with your feature and ground-truth tables for debugging and monitoring

With this setup, Databricks captures request and response payloads plus endpoint metadata, including the model version used by each serving request.

## Databricks alignment

This implementation preserves the same responsibilities you would map into Databricks assets:

* notebooks and jobs map to the orchestration stages
* Unity Catalog / Lakehouse layers map to the persisted bronze, silver, feature, inference, and monitoring datasets
* Feature Engineering and Feature Store map to the offline and online feature store modules
* MLflow Tracking and Registry map to the local experiment tracker and model registry artifacts
* Model Serving maps to the HTTP service in `serving.py`

The local abstractions are deliberately isolated so they can be replaced later with real Databricks and MLflow clients without rewriting the business flow.

## Databricks Asset Bundle

O projeto também contém um Databricks Asset Bundle em `databricks.yml`, com
dois jobs baseados no repositório GitHub:

* `feature_store`, com a task de notebook `gold_online_transactions`.
* `model_training`, com a task Python `fraud_detection_model` e o ambiente
  Databricks contendo as dependências de ML, seguida da criação ou atualização
  do endpoint `fraud-detection` para o alias `champion`.

O endpoint não faz parte do deploy padrão do bundle, porque depende de uma
versão de modelo já registrada e em estado `READY`. A task `deploy_champion`,
implementada em `src/deploy_champion.py`, resolve o alias `champion` no Unity
Catalog e aplica o estado desejado de forma idempotente: cria o endpoint quando
ele ainda não existe, atualiza quando aponta para outra versão e não altera nada
quando já serve a versão champion atual.

Para validar e publicar usando o Databricks CLI:

```bash
databricks bundle validate -t qas
databricks bundle deploy -t qas
databricks bundle run feature_store -t qas
databricks bundle run model_training -t qas
```

### CI/CD com GitHub Actions

O workflow `.github/workflows/databricks-bundle.yml` valida e publica o bundle
com dois targets no mesmo workspace:

* `qas`, usando o catalogo `qas_main` e o endpoint `fraud-detection-qas`.
* `prd`, usando o catalogo `prd_main` e o endpoint `fraud-detection-prd`.

Pull requests para `qas`, incluindo branches `feat/*`, executam os checks
Python e `databricks bundle validate --target qas`. Pull requests de `qas` para
`master` executam os mesmos checks para `prd`. Tags `qas-*` fazem deploy em QAS,
desde que o commit tagueado pertença à branch `qas`; tags `prd-*` ou `prod-*`
fazem deploy em PRD, desde que o commit tagueado pertença à branch `master`.

O workflow usa autenticação OIDC do GitHub Actions com Databricks. Configure no
GitHub as variáveis `DATABRICKS_HOST` e `DATABRICKS_CLIENT_ID`, e habilite os
environments `qas` e `prd` se quiser aprovações manuais antes do deploy.

Fluxo recomendado de promoção:

```bash
# depois do merge para qas
git checkout qas
git pull origin qas
git tag qas-v0.1.0
git push origin qas-v0.1.0

# depois da aceitacao em QAS, abra PR qas -> master e faca merge sem squash
git checkout master
git pull origin master
git tag prd-v0.1.0 qas-v0.1.0
git push origin prd-v0.1.0
```

Criar a tag `prd-*` apontando para a tag `qas-*` garante que PRD implante o
mesmo commit aprovado em QAS. O merge `qas -> master` precisa preservar esse
commit no historico de `master`; por isso, evite squash/rebase nesse fluxo.
