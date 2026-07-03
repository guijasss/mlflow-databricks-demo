from __future__ import annotations

import argparse
import json
import os

import pandas as pd
from pyspark.sql import SparkSession, functions as F

from src.databricks_auth import configure_databricks_auth
from src.features import FEATURE_COLUMNS, METADATA_COLUMNS, TARGET_COLUMN
from src.model_registry import (
    CHAMPION_ALIAS,
    configure_mlflow,
    get_client,
    get_model_uri,
    get_model_version_by_alias,
)
from src.pipeline import FraudModel


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Carrega o champion do MLflow/Databricks e gera inferencia "
            "para transacoes sem label."
        )
    )
    parser.add_argument(
        "--feature-table",
        default=os.getenv("FEATURE_TABLE"),
        required=os.getenv("FEATURE_TABLE") is None,
        help="Tabela da feature store com a coluna 'fraude'.",
    )
    parser.add_argument(
        "--output-table",
        default=os.getenv("MODEL_OUTPUT_TABLE", "model_output"),
        help="Tabela de saida das predicoes.",
    )
    parser.add_argument(
        "--registered-model-name",
        default=os.getenv("REGISTERED_MODEL_NAME"),
        required=os.getenv("REGISTERED_MODEL_NAME") is None,
        help="Nome do modelo no registry, idealmente em UC: catalog.schema.nome.",
    )
    parser.add_argument(
        "--model-alias",
        default=os.getenv("MODEL_ALIAS", CHAMPION_ALIAS),
        help="Alias a ser usado na inferencia.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI", "databricks"),
    )
    parser.add_argument(
        "--registry-uri",
        default=os.getenv("MLFLOW_REGISTRY_URI", "databricks-uc"),
    )
    parser.add_argument(
        "--databricks-host",
        default=os.getenv("DATABRICKS_HOST"),
        help="Workspace URL do Databricks, sem token.",
    )
    parser.add_argument(
        "--databricks-secret-scope",
        default=os.getenv("DATABRICKS_SECRET_SCOPE"),
        help="Secret scope com o token de acesso do Databricks.",
    )
    parser.add_argument(
        "--databricks-secret-key",
        default=os.getenv("DATABRICKS_SECRET_KEY"),
        help="Chave do secret que contem o token do Databricks.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=float(os.getenv("FRAUD_THRESHOLD", "0.5")),
    )
    parser.add_argument(
        "--write-mode",
        default=os.getenv("MODEL_OUTPUT_WRITE_MODE", "append"),
        choices=["append", "overwrite"],
    )
    return parser.parse_args()


def load_unlabeled_dataset(
    spark: SparkSession,
    feature_table: str,
) -> pd.DataFrame:

    return (
        spark.table(feature_table)
        .where(F.col(TARGET_COLUMN).isNull())
        .select(*(METADATA_COLUMNS + FEATURE_COLUMNS))
        .toPandas()
    )


def main() -> None:

    args = parse_args()
    spark = SparkSession.builder.getOrCreate()
    configure_databricks_auth(
        spark,
        host=args.databricks_host,
        token_secret_scope=args.databricks_secret_scope,
        token_secret_key=args.databricks_secret_key,
    )

    configure_mlflow(
        tracking_uri=args.tracking_uri,
        registry_uri=args.registry_uri,
    )
    client = get_client(
        tracking_uri=args.tracking_uri,
        registry_uri=args.registry_uri,
    )

    model_version = get_model_version_by_alias(
        client,
        args.registered_model_name,
        args.model_alias,
    )
    if model_version is None:
        raise RuntimeError(
            f"Nao existe modelo com alias '{args.model_alias}' para "
            f"{args.registered_model_name}."
        )

    inference_df = load_unlabeled_dataset(
        spark=spark,
        feature_table=args.feature_table,
    )
    if inference_df.empty:
        summary = {
            "feature_table": args.feature_table,
            "output_table": args.output_table,
            "rows_written": 0,
            "model_alias": args.model_alias,
            "model_version": model_version.version,
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    model = FraudModel.load_from_mlflow(
        get_model_uri(
            args.registered_model_name,
            alias=args.model_alias,
        )
    )

    scores = model.predict_proba(inference_df)
    labels = scores >= args.threshold
    output_df = pd.DataFrame(
        {
            "prediction_timestamp": pd.Timestamp.now(
                tz="UTC"
            ).tz_localize(None),
            "model_name": args.registered_model_name,
            "model_alias": args.model_alias,
            "model_version": str(model_version.version),
            "id_transacao": inference_df["id_transacao"],
            "id_cliente": inference_df["id_cliente"],
            "fraud_probability": scores,
            "predicted_fraude": labels.astype(bool),
            "classificacao_modelo": [
                "fraud" if is_fraud else "not_fraud"
                for is_fraud in labels
            ],
        }
    )

    spark.createDataFrame(output_df).write.mode(
        args.write_mode
    ).saveAsTable(args.output_table)

    summary = {
        "feature_table": args.feature_table,
        "output_table": args.output_table,
        "rows_written": int(len(output_df)),
        "model_alias": args.model_alias,
        "model_version": model_version.version,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
