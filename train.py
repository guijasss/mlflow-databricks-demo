from __future__ import annotations

import argparse
import json
import os

from pyspark.sql import SparkSession, functions as F
from sklearn.model_selection import train_test_split

from src.audit import (
    build_split_mlflow_dataset,
    compute_feature_set_digest,
    load_table_snapshot,
    persist_split_snapshots,
    resolve_table_snapshot,
)
from src.databricks_auth import configure_databricks_auth
from src.features import FEATURE_COLUMNS, METADATA_COLUMNS, TARGET_COLUMN
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
        "--test-fraction",
        type=float,
        default=float(os.getenv("TEST_FRACTION", "0.1")),
        help="Fracao do dataset reservada para teste/auditoria.",
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
    parser.add_argument(
        "--feature-table-version",
        type=int,
        default=(
            int(os.getenv("FEATURE_TABLE_VERSION"))
            if os.getenv("FEATURE_TABLE_VERSION")
            else None
        ),
        help="Versao Delta da feature table a ser usada no treino.",
    )
    parser.add_argument(
        "--split-volume-path",
        default=os.getenv("TRAINING_SPLIT_VOLUME_PATH"),
        help="Path em Volume para persistir os snapshots de train/validation/test.",
    )
    return parser.parse_args()


def load_labeled_dataset(
    spark: SparkSession,
    feature_table: str,
    *,
    feature_table_version: int | None = None,
):

    return (
        load_table_snapshot(
            spark,
            feature_table,
            version=feature_table_version,
        )
        .where(F.col(TARGET_COLUMN).isNotNull())
        .select(*(METADATA_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN]))
        .toPandas()
    )


def split_dataset(
    labeled_df,
    *,
    validation_fraction: float,
    test_fraction: float,
    random_state: int,
):

    if validation_fraction <= 0 or validation_fraction >= 1:
        raise ValueError("validation_fraction deve estar entre 0 e 1.")
    if test_fraction < 0 or test_fraction >= 1:
        raise ValueError("test_fraction deve estar entre 0 e 1.")
    if validation_fraction + test_fraction >= 1:
        raise ValueError(
            "validation_fraction + test_fraction deve ser menor que 1."
        )

    stratify = labeled_df[TARGET_COLUMN]

    if test_fraction == 0:
        train_df, validation_df = train_test_split(
            labeled_df,
            test_size=validation_fraction,
            random_state=random_state,
            stratify=stratify,
        )
        return train_df, validation_df, validation_df.copy()

    train_df, holdout_df = train_test_split(
        labeled_df,
        test_size=validation_fraction + test_fraction,
        random_state=random_state,
        stratify=stratify,
    )

    validation_share = validation_fraction / (
        validation_fraction + test_fraction
    )
    validation_df, test_df = train_test_split(
        holdout_df,
        test_size=1 - validation_share,
        random_state=random_state,
        stratify=holdout_df[TARGET_COLUMN],
    )
    return train_df, validation_df, test_df


def main() -> None:

    args = parse_args()
    spark = SparkSession.builder.getOrCreate()
    configure_databricks_auth(
        spark,
        host=args.databricks_host,
        token_secret_scope=args.databricks_secret_scope,
        token_secret_key=args.databricks_secret_key,
    )

    (
        feature_table_version,
        feature_table_timestamp,
    ) = resolve_table_snapshot(
        spark,
        args.feature_table,
        requested_version=args.feature_table_version,
    )

    labeled_df = load_labeled_dataset(
        spark=spark,
        feature_table=args.feature_table,
        feature_table_version=feature_table_version,
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

    train_df, validation_df, test_df = split_dataset(
        labeled_df,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        random_state=args.random_state,
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
    feature_set_digest = compute_feature_set_digest()

    with mlflow.start_run(run_name="fraud-training") as run:
        train_dataset = build_split_mlflow_dataset(
            mlflow,
            train_df,
            table_name=args.feature_table,
            split_name="train",
            table_version=feature_table_version,
        )
        validation_dataset = build_split_mlflow_dataset(
            mlflow,
            validation_df,
            table_name=args.feature_table,
            split_name="validation",
            table_version=feature_table_version,
        )
        test_dataset = build_split_mlflow_dataset(
            mlflow,
            test_df,
            table_name=args.feature_table,
            split_name="test",
            table_version=feature_table_version,
        )
        mlflow.log_input(train_dataset, context="training")
        mlflow.log_input(validation_dataset, context="validation")
        mlflow.log_input(test_dataset, context="testing")

        split_snapshot_manifest = None
        if args.split_volume_path:
            split_snapshot_manifest = persist_split_snapshots(
                spark,
                volume_path=args.split_volume_path,
                run_id=run.info.run_id,
                feature_table=args.feature_table,
                feature_table_version=feature_table_version,
                feature_table_timestamp=feature_table_timestamp,
                split_dfs={
                    "train": train_df,
                    "validation": validation_df,
                    "test": test_df,
                },
                extra_metadata={
                    "registered_model_name": args.registered_model_name,
                    "validation_fraction": args.validation_fraction,
                    "test_fraction": args.test_fraction,
                    "random_state": args.random_state,
                },
            )
            mlflow.log_text(
                json.dumps(
                    split_snapshot_manifest,
                    indent=2,
                    sort_keys=True,
                ),
                "audit/split_snapshot_manifest.json",
            )

        mlflow.log_param("feature_table", args.feature_table)
        mlflow.log_param("feature_table_version", feature_table_version)
        mlflow.log_param(
            "feature_table_timestamp",
            feature_table_timestamp,
        )
        mlflow.log_param("registered_model_name", args.registered_model_name)
        mlflow.log_param("validation_fraction", args.validation_fraction)
        mlflow.log_param("test_fraction", args.test_fraction)
        mlflow.log_param("threshold", args.threshold)
        mlflow.log_param("promotion_metric", args.promotion_metric)
        mlflow.log_param("feature_set_digest", feature_set_digest)
        mlflow.log_param(
            "split_volume_path",
            args.split_volume_path or "None",
        )
        mlflow.log_params(challenger.classifier_params)
        mlflow.log_text(
            json.dumps(
                {
                    "feature_columns": FEATURE_COLUMNS,
                    "metadata_columns": METADATA_COLUMNS,
                    "target_column": TARGET_COLUMN,
                    "feature_set_digest": feature_set_digest,
                },
                indent=2,
                sort_keys=True,
            ),
            "audit/feature_spec.json",
        )
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
                "source_table_version": str(feature_table_version),
                "feature_set_digest": feature_set_digest,
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
                "source_table_version": feature_table_version,
                "source_table_timestamp": feature_table_timestamp or "None",
                "feature_set_digest": feature_set_digest,
                "split_snapshot_manifest": (
                    split_snapshot_manifest["manifest_path"]
                    if split_snapshot_manifest is not None
                    else "None"
                ),
            },
        )

        mlflow.log_param("registered_model_version", model_version.version)
        mlflow.log_param("promoted_to_champion", comparison.promoted)

    summary = {
        "registered_model_name": args.registered_model_name,
        "challenger_version": model_version.version,
        "previous_champion_version": champion_version,
        "feature_table_version": feature_table_version,
        "feature_table_timestamp": feature_table_timestamp,
        "promotion_metric": args.promotion_metric,
        "challenger_metrics": challenger_metrics,
        "champion_metrics": champion_metrics,
        "promoted_to_champion": comparison.promoted,
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(validation_df)),
        "test_rows": int(len(test_df)),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
