"""
Pipeline de Machine Learning para detecção de fraudes.

Toda a lógica de pré-processamento fica encapsulada neste módulo.
Os consumidores (treino, batch e API) nunca manipulam OneHotEncoder,
ColumnTransformer etc.
"""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted
from typing import Any, TYPE_CHECKING

from xgboost import XGBClassifier

from src.features import (
    BOOLEAN_COLUMNS,
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
)

if TYPE_CHECKING:
    from mlflow.models.model import ModelInfo
    from mlflow.models.signature import ModelSignature


DEFAULT_CLASSIFIER_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}

MODEL_PIP_REQUIREMENTS = [
    "mlflow>=3.0.0",
    "pandas>=2.2.3,<3",
    "scikit-learn>=1.9.0",
    "xgboost>=3.3.0",
    "cloudpickle>=3.0.0",
]


def _cast_boolean_columns(values):

    return pd.DataFrame(values).fillna(False).astype("int8").to_numpy()


def _import_mlflow():

    try:
        import mlflow
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "mlflow nao esta instalado. Instale a dependencia no ambiente "
            "que fara o log do modelo no Databricks."
        ) from exc

    return mlflow


class FraudModel:

    def __init__(self, classifier_params: dict[str, Any] | None = None):

        self.classifier_params = {
            **DEFAULT_CLASSIFIER_PARAMS,
            **(classifier_params or {}),
        }
        self.pipeline = self._build_pipeline()


    def _build_pipeline(self) -> Pipeline:

        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                )
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent"),
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                    ),
                ),
            ]
        )

        boolean_pipeline = Pipeline(
            steps=[
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                    ),
                )
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    numeric_pipeline,
                    NUMERIC_COLUMNS,
                ),
                (
                    "categorical",
                    categorical_pipeline,
                    CATEGORICAL_COLUMNS,
                ),
                (
                    "boolean",
                    boolean_pipeline,
                    BOOLEAN_COLUMNS,
                ),
            ]
        )

        classifier = XGBClassifier(**self.classifier_params)

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor,
                ),
                (
                    "classifier",
                    classifier,
                ),
            ]
        )

        return pipeline


    def _missing_columns(
        self,
        df: pd.DataFrame,
        required_columns: list[str],
    ) -> list[str]:

        return [column for column in required_columns if column not in df.columns]


    def _validate_columns(
        self,
        df: pd.DataFrame,
        required_columns: list[str],
        context: str,
    ) -> None:

        missing_columns = self._missing_columns(df, required_columns)

        if missing_columns:
            missing = ", ".join(missing_columns)
            raise ValueError(
                f"{context}: colunas obrigatorias ausentes: {missing}"
            )


    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:

        self._validate_columns(
            df,
            FEATURE_COLUMNS,
            context="Dados invalidos para inferencia",
        )

        return df.loc[:, FEATURE_COLUMNS].copy()


    def _ensure_fitted(self) -> None:

        try:
            check_is_fitted(self.pipeline)
        except NotFittedError as exc:
            raise NotFittedError(
                "O modelo precisa ser treinado com fit() antes da inferencia "
                "ou do log no MLflow."
            ) from exc


    def split_xy(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:

        self._validate_columns(
            df,
            FEATURE_COLUMNS + [TARGET_COLUMN],
            context="Dados invalidos para treino",
        )

        X = self._select_features(df)

        y = df.loc[:, TARGET_COLUMN].copy()

        return X, y


    def fit(self, df: pd.DataFrame) -> FraudModel:

        X, y = self.split_xy(df)

        self.pipeline.fit(X, y)

        return self


    def predict(self, df: pd.DataFrame):

        self._ensure_fitted()

        X = self._select_features(df)

        return self.pipeline.predict(X)


    def predict_proba(self, df: pd.DataFrame):

        self._ensure_fitted()

        X = self._select_features(df)

        return self.pipeline.predict_proba(X)[:, 1]


    def feature_importance(self) -> pd.DataFrame:

        self._ensure_fitted()

        preprocessor = self.pipeline.named_steps["preprocessor"]
        classifier = self.pipeline.named_steps["classifier"]

        feature_names = preprocessor.get_feature_names_out()
        importances = classifier.feature_importances_

        return (
            pd.DataFrame(
                {
                    "feature_name": feature_names,
                    "importance": importances,
                }
            )
            .sort_values(
                "importance",
                ascending=False,
                ignore_index=True,
            )
        )


    def log_model(
        self,
        artifact_path: str = "fraud-model",
        registered_model_name: str | None = None,
        input_example: pd.DataFrame | None = None,
        await_registration_for: int | None = 300,
        signature: ModelSignature | None | bool = None,
        tags: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> ModelInfo:

        self._ensure_fitted()

        mlflow = _import_mlflow()

        example = None
        if input_example is not None:
            example = self._select_features(input_example).head(5)

        active_run = mlflow.active_run()

        if active_run is not None:
            return mlflow.sklearn.log_model(
                sk_model=self.pipeline,
                name=artifact_path,
                registered_model_name=registered_model_name,
                input_example=example,
                await_registration_for=await_registration_for,
                serialization_format="cloudpickle",
                signature=signature,
                tags=tags,
                params=params,
                pip_requirements=MODEL_PIP_REQUIREMENTS,
            )

        with mlflow.start_run():
            return mlflow.sklearn.log_model(
                sk_model=self.pipeline,
                name=artifact_path,
                registered_model_name=registered_model_name,
                input_example=example,
                await_registration_for=await_registration_for,
                serialization_format="cloudpickle",
                signature=signature,
                tags=tags,
                params=params,
                pip_requirements=MODEL_PIP_REQUIREMENTS,
            )


    @classmethod
    def load_from_mlflow(cls, model_uri: str) -> FraudModel:

        mlflow = _import_mlflow()

        loaded_model: BaseEstimator = mlflow.sklearn.load_model(model_uri)
        return cls.from_pipeline(loaded_model)


    @classmethod
    def from_pipeline(cls, pipeline: BaseEstimator) -> FraudModel:

        model = cls.__new__(cls)
        model.classifier_params = DEFAULT_CLASSIFIER_PARAMS.copy()
        model.pipeline = pipeline
        return model


    @classmethod
    def load(cls, model_uri: str) -> FraudModel:

        return cls.load_from_mlflow(model_uri)


    def save(
        self,
        artifact_path: str = "fraud-model",
        registered_model_name: str | None = None,
        input_example: pd.DataFrame | None = None,
        await_registration_for: int | None = 300,
        signature: ModelSignature | None | bool = None,
        tags: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> ModelInfo:

        return self.log_model(
            artifact_path=artifact_path,
            registered_model_name=registered_model_name,
            input_example=input_example,
            await_registration_for=await_registration_for,
            signature=signature,
            tags=tags,
            params=params,
        )
