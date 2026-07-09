from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from pyspark.sql import SparkSession

from src.features import FEATURE_COLUMNS, METADATA_COLUMNS, TARGET_COLUMN


def compute_feature_set_digest(
    feature_columns: list[str] | tuple[str, ...] = FEATURE_COLUMNS,
) -> str:

    payload = json.dumps(
        list(feature_columns),
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_dataframe_digest(
    df: pd.DataFrame,
    *,
    columns: list[str] | None = None,
) -> str:

    selected = df.loc[:, columns].copy() if columns else df.copy()
    hashed = pd.util.hash_pandas_object(selected, index=True)
    return hashlib.sha256(hashed.to_numpy().tobytes()).hexdigest()


def resolve_table_snapshot(
    spark: SparkSession,
    table_name: str,
    requested_version: int | None = None,
) -> tuple[int | None, str | None]:

    if requested_version is not None:
        history = spark.sql(
            f"DESCRIBE HISTORY {table_name}"
        ).where(f"version = {requested_version}")
    else:
        history = spark.sql(
            f"DESCRIBE HISTORY {table_name}"
        ).orderBy("version", ascending=False).limit(1)

    row = history.select("version", "timestamp").first()
    if row is None:
        if requested_version is not None:
            raise ValueError(
                f"A versao {requested_version} nao existe para {table_name}."
            )
        return None, None

    version = int(row["version"]) if row["version"] is not None else None
    timestamp = (
        row["timestamp"].isoformat()
        if row["timestamp"] is not None
        else None
    )
    return version, timestamp


def load_table_snapshot(
    spark: SparkSession,
    table_name: str,
    *,
    version: int | None = None,
):

    if version is None:
        return spark.table(table_name)

    return spark.sql(
        f"SELECT * FROM {table_name} VERSION AS OF {version}"
    )


def build_dataset_source(
    table_name: str,
    *,
    table_version: int | None = None,
) -> str:

    if table_version is None:
        return f"databricks-table://{table_name}"

    return f"databricks-table://{table_name}@v{table_version}"


def build_split_mlflow_dataset(
    mlflow,
    split_df: pd.DataFrame,
    *,
    table_name: str,
    split_name: str,
    table_version: int | None = None,
):

    dataset_name = f"{table_name.replace('.', '_')}_{split_name}"
    return mlflow.data.from_pandas(
        split_df.loc[:, METADATA_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN]],
        source=build_dataset_source(
            table_name,
            table_version=table_version,
        ),
        name=dataset_name,
        targets=TARGET_COLUMN,
    )


def persist_split_snapshots(
    spark: SparkSession,
    *,
    volume_path: str,
    run_id: str,
    feature_table: str,
    feature_table_version: int | None,
    feature_table_timestamp: str | None,
    split_dfs: dict[str, pd.DataFrame],
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:

    root_path = Path(volume_path.rstrip("/")) / f"run_id={run_id}"
    root_path.mkdir(parents=True, exist_ok=True)

    split_paths: dict[str, str] = {}
    split_digests: dict[str, str] = {}
    split_row_counts: dict[str, int] = {}

    ordered_columns = METADATA_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN]
    for split_name, split_df in split_dfs.items():
        split_path = root_path / split_name
        spark.createDataFrame(
            split_df.loc[:, ordered_columns]
        ).write.mode("overwrite").parquet(str(split_path))
        split_paths[split_name] = str(split_path)
        split_digests[split_name] = compute_dataframe_digest(
            split_df,
            columns=ordered_columns,
        )
        split_row_counts[split_name] = int(len(split_df))

    manifest = {
        "run_id": run_id,
        "feature_table": feature_table,
        "feature_table_version": feature_table_version,
        "feature_table_timestamp": feature_table_timestamp,
        "feature_set_digest": compute_feature_set_digest(),
        "feature_columns": list(FEATURE_COLUMNS),
        "metadata_columns": list(METADATA_COLUMNS),
        "target_column": TARGET_COLUMN,
        "split_paths": split_paths,
        "split_digests": split_digests,
        "split_row_counts": split_row_counts,
    }
    if extra_metadata:
        manifest.update(extra_metadata)

    manifest_path = root_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest["manifest_path"] = str(manifest_path)
    return manifest
