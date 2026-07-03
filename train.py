from __future__ import annotations

import argparse
import json
import os

from pyspark.sql import SparkSession, functions as F
from sklearn.model_selection import train_test_split

from src.databricks_auth import configure_databricks_auth
from src.features import FEATURE_COLUMNS, TARGET_COLUMN
from src.model_registry import (
    PRIMARY_METRIC,
    compare_challenger_vs_champion,
    configure_mlflow,
    ensure_registered_model,
    get_client,
    infer_model_signature,
    promote_model_if_better,
    resolve_model_version_for_run,
    set_model_version_tags,
)
from src.pipeline import FraudModel


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Treina um challenger, registra no MLflow/Databricks e "
            "promove para champion quando superar o modelo atual."
        )
    )
    parser.add_argument(
        "--feature-table",
        default=os.getenv("FEATURE_TABLE"),
        required=os.getenv("FEATURE_TABLE") is None,
        help="Tabela da feature store com a coluna 'fraude'.",
    )
    parser.add_argument(
        "--registered-model-name",
        default=os.getenv("REGISTERED_MODEL_NAME"),
        required=os.getenv("REGISTERED_MODEL_NAME") is None,
        help="Nome do modelo no registry, idealmente em UC: catalog.schema.nome.",
    )
    parser.add_argument(
        "--experiment-name",
        default=os.getenv("MLFLOW_EXPERIMENT_NAME"),
        help="Experimento do MLflow para registrar a run.",
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
        "--model-artifact-name",
        default=os.getenv("MLFLOW_MODEL_ARTIFACT", "fraud-model"),
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=float(os.getenv("VALIDATION_FRACTION", "0.2")),
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=int(os.getenv("RANDOM_STATE", "42")),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=float(os.getenv("FRAUD_THRESHOLD", "0.5")),
    )
    parser.add_argument(
        "--promotion-metric",
        default=os.getenv("PROMOTION_METRIC", PRIMARY_METRIC),
        help="Metrica para comparar challenger e champion.",
    )
    return parser.parse_args()


def load_labeled_dataset(
    spark: SparkSession,
    feature_table: str,
):

    return (
        spark.table(feature_table)
        .where(F.col(TARGET_COLUMN).isNotNull())
        .select(*(FEATURE_COLUMNS + [TARGET_COLUMN]))
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

    labeled_df = load_labeled_dataset(
        spark=spark,
        feature_table=args.feature_table,
    )
    if labeled_df.empty:
        raise RuntimeError(
            "Nenhuma linha com fraude preenchida foi encontrada para treino."
        )

    labeled_df[TARGET_COLUMN] = labeled_df[TARGET_COLUMN].astype(int)
    if labeled_df[TARGET_COLUMN].nunique() < 2:
        raise RuntimeError(
            "O treino requer exemplos positivos e negativos em 'fraude'."
        )

    train_df, validation_df = train_test_split(
        labeled_df,
        test_size=args.validation_fraction,
        random_state=args.random_state,
        stratify=labeled_df[TARGET_COLUMN],
    )

    challenger = FraudModel().fit(train_df)

    mlflow = configure_mlflow(
        experiment_name=args.experiment_name,
        tracking_uri=args.tracking_uri,
        registry_uri=args.registry_uri,
    )
    client = get_client(
        tracking_uri=args.tracking_uri,
        registry_uri=args.registry_uri,
    )

    ensure_registered_model(
        client,
        args.registered_model_name,
        description=(
            "Modelo de deteccao de fraude com aliases champion/challenger."
        ),
    )

    (
        challenger_metrics,
        champion_metrics,
        champion_version,
        _,
    ) = compare_challenger_vs_champion(
        client=client,
        model_name=args.registered_model_name,
        challenger_model=challenger,
        evaluation_df=validation_df,
        metric_name=args.promotion_metric,
        threshold=args.threshold,
    )

    signature = infer_model_signature(challenger, validation_df)
    input_example = validation_df.loc[:, FEATURE_COLUMNS].head(5)

    with mlflow.start_run(run_name="fraud-training") as run:
        mlflow.log_param("feature_table", args.feature_table)
        mlflow.log_param("registered_model_name", args.registered_model_name)
        mlflow.log_param("validation_fraction", args.validation_fraction)
        mlflow.log_param("threshold", args.threshold)
        mlflow.log_param("promotion_metric", args.promotion_metric)
        mlflow.log_params(challenger.classifier_params)
        mlflow.log_metrics(
            {
                f"challenger_{key}": value
                for key, value in challenger_metrics.items()
            }
        )
        if champion_metrics is not None:
            mlflow.log_metrics(
                {
                    f"champion_{key}": value
                    for key, value in champion_metrics.items()
                }
            )

        challenger.log_model(
            artifact_path=args.model_artifact_name,
            registered_model_name=args.registered_model_name,
            input_example=input_example,
            signature=signature,
            params={
                "promotion_metric": args.promotion_metric,
                "threshold": args.threshold,
            },
            tags={
                "model_role": "challenger",
                "source_table": args.feature_table,
            },
        )

        model_version = resolve_model_version_for_run(
            client,
            args.registered_model_name,
            run.info.run_id,
        )

        comparison = promote_model_if_better(
            client=client,
            model_name=args.registered_model_name,
            challenger_version=model_version.version,
            metric_name=args.promotion_metric,
            challenger_score=challenger_metrics[args.promotion_metric],
            champion_score=(
                champion_metrics[args.promotion_metric]
                if champion_metrics is not None
                else None
            ),
        )

        set_model_version_tags(
            client=client,
            model_name=args.registered_model_name,
            model_version=model_version.version,
            tags={
                "promotion_metric": args.promotion_metric,
                "challenger_score": challenger_metrics[args.promotion_metric],
                "champion_score": (
                    champion_metrics[args.promotion_metric]
                    if champion_metrics is not None
                    else "None"
                ),
                "promoted_to_champion": comparison.promoted,
                "previous_champion_version": champion_version or "None",
                "source_table": args.feature_table,
            },
        )

        mlflow.log_param("registered_model_version", model_version.version)
        mlflow.log_param("promoted_to_champion", comparison.promoted)

    summary = {
        "registered_model_name": args.registered_model_name,
        "challenger_version": model_version.version,
        "previous_champion_version": champion_version,
        "promotion_metric": args.promotion_metric,
        "challenger_metrics": challenger_metrics,
        "champion_metrics": champion_metrics,
        "promoted_to_champion": comparison.promoted,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
