from __future__ import annotations

import os

from pyspark.sql import SparkSession


def _get_dbutils(spark: SparkSession):

    try:
        from pyspark.dbutils import DBUtils
    except ModuleNotFoundError:
        DBUtils = None

    if DBUtils is not None:
        try:
            return DBUtils(spark)
        except Exception:
            pass

    try:
        import builtins

        return builtins.dbutils
    except AttributeError:
        return None


def configure_databricks_auth(
    spark: SparkSession,
    *,
    host: str | None = None,
    token_secret_scope: str | None = None,
    token_secret_key: str | None = None,
) -> None:

    if host:
        os.environ["DATABRICKS_HOST"] = host.rstrip("/")

    if token_secret_scope is None and token_secret_key is None:
        return

    if not token_secret_scope or not token_secret_key:
        raise ValueError(
            "Informe databricks_secret_scope e databricks_secret_key em conjunto."
        )

    dbutils = _get_dbutils(spark)
    if dbutils is None:
        raise RuntimeError(
            "dbutils nao esta disponivel neste ambiente para ler o token do secret scope."
        )

    os.environ["DATABRICKS_TOKEN"] = dbutils.secrets.get(
        scope=token_secret_scope,
        key=token_secret_key,
    )
