from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.features import FEATURE_COLUMNS, TARGET_COLUMN
from src.pipeline import FraudModel

CHAMPION_ALIAS = "champion"
CHALLENGER_ALIAS = "challenger"
PRIMARY_METRIC = "average_precision"


@dataclass
class ChampionComparison:
    metric_name: str
    challenger_score: float
    champion_score: float | None
    promoted: bool
    champion_version: str | None
    challenger_version: str


def _import_mlflow():

    try:
        import mlflow
        from mlflow import MlflowClient
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "mlflow nao esta instalado. Instale a dependencia no ambiente "
            "que executara treino e inferencia no Databricks."
        ) from exc

    return mlflow, MlflowClient


def configure_mlflow(
    experiment_name: str | None = None,
    tracking_uri: str = "databricks",
    registry_uri: str = "databricks-uc",
):

    mlflow, _ = _import_mlflow()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(registry_uri)
    if (
        tracking_uri == "databricks"
        and not experiment_name
    ):
        raise ValueError(
            "MLFLOW_EXPERIMENT_NAME ou --experiment-name eh obrigatorio "
            "para treinos com tracking_uri='databricks'. Exemplo: "
            "/Shared/fraud-training."
        )
    if experiment_name:
        mlflow.set_experiment(experiment_name)
    return mlflow


def get_client(
    tracking_uri: str = "databricks",
    registry_uri: str = "databricks-uc",
):

    _, mlflow_client = _import_mlflow()
    return mlflow_client(
        tracking_uri=tracking_uri,
        registry_uri=registry_uri,
    )


def ensure_registered_model(
    client,
    model_name: str,
    description: str | None = None,
) -> None:

    try:
        client.get_registered_model(model_name)
    except Exception:
        client.create_registered_model(
            name=model_name,
            description=description,
        )


def get_model_uri(
    model_name: str,
    *,
    alias: str | None = None,
    version: str | None = None,
) -> str:

    if alias:
        return f"models:/{model_name}@{alias}"

    if version:
        return f"models:/{model_name}/{version}"

    raise ValueError("Informe alias ou version para montar a model URI.")


def get_model_version_by_alias(
    client,
    model_name: str,
    alias: str,
):

    try:
        return client.get_model_version_by_alias(model_name, alias)
    except Exception:
        return None


def load_model_by_alias(
    model_name: str,
    alias: str = CHAMPION_ALIAS,
) -> FraudModel:

    return FraudModel.load_from_mlflow(
        get_model_uri(model_name, alias=alias)
    )


def evaluate_binary_classifier(
    model: FraudModel,
    evaluation_df: pd.DataFrame,
    threshold: float = 0.5,
) -> dict[str, float]:

    y_true = evaluation_df.loc[:, TARGET_COLUMN].astype(int)
    y_score = model.predict_proba(evaluation_df)
    y_pred = (y_score >= threshold).astype(int)

    metrics = {
        "average_precision": float(
            average_precision_score(y_true, y_score)
        ),
        "f1": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
    }

    if y_true.nunique() > 1:
        metrics["roc_auc"] = float(
            roc_auc_score(y_true, y_score)
        )

    return metrics


def compare_challenger_vs_champion(
    client,
    model_name: str,
    challenger_model: FraudModel,
    evaluation_df: pd.DataFrame,
    metric_name: str = PRIMARY_METRIC,
    threshold: float = 0.5,
) -> tuple[dict[str, float], dict[str, float] | None, str | None, bool]:

    challenger_metrics = evaluate_binary_classifier(
        challenger_model,
        evaluation_df,
        threshold=threshold,
    )

    champion_version = get_model_version_by_alias(
        client,
        model_name,
        CHAMPION_ALIAS,
    )
    if champion_version is None:
        return challenger_metrics, None, None, True

    champion_model = load_model_by_alias(
        model_name,
        alias=CHAMPION_ALIAS,
    )
    champion_metrics = evaluate_binary_classifier(
        champion_model,
        evaluation_df,
        threshold=threshold,
    )
    promoted = challenger_metrics[metric_name] > champion_metrics[metric_name]

    return (
        challenger_metrics,
        champion_metrics,
        champion_version.version,
        promoted,
    )


def resolve_model_version_for_run(
    client,
    model_name: str,
    run_id: str,
):

    versions = [
        version for version in client.search_model_versions(
            filter_string=f"name = '{model_name}'"
        )
        if (
            version.name == model_name
            and getattr(version, "run_id", None) == run_id
        )
    ]

    if not versions:
        raise RuntimeError(
            "Nao foi possivel localizar a versao registrada do modelo "
            f"{model_name} para a run {run_id}."
        )

    return max(
        versions,
        key=lambda version: int(version.version),
    )


def set_model_version_tags(
    client,
    model_name: str,
    model_version: str,
    tags: dict[str, Any],
) -> None:

    for key, value in tags.items():
        client.set_model_version_tag(
            model_name,
            model_version,
            key,
            str(value),
        )


def promote_model_if_better(
    client,
    model_name: str,
    challenger_version: str,
    metric_name: str,
    challenger_score: float,
    champion_score: float | None,
) -> ChampionComparison:

    client.set_registered_model_alias(
        model_name,
        CHALLENGER_ALIAS,
        challenger_version,
    )

    promoted = (
        champion_score is None or challenger_score > champion_score
    )

    previous_champion = get_model_version_by_alias(
        client,
        model_name,
        CHAMPION_ALIAS,
    )
    previous_champion_version = (
        previous_champion.version if previous_champion is not None else None
    )

    if promoted:
        client.set_registered_model_alias(
            model_name,
            CHAMPION_ALIAS,
            challenger_version,
        )

    return ChampionComparison(
        metric_name=metric_name,
        challenger_score=challenger_score,
        champion_score=champion_score,
        promoted=promoted,
        champion_version=previous_champion_version,
        challenger_version=challenger_version,
    )


def infer_model_signature(
    model: FraudModel,
    df: pd.DataFrame,
):

    mlflow, _ = _import_mlflow()

    features = df.loc[:, FEATURE_COLUMNS].head(100)
    predictions = model.predict(features)
    return mlflow.models.infer_signature(features, predictions)
