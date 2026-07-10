"""Atualiza o endpoint para a versão apontada pelo alias champion."""

from __future__ import annotations

import argparse
import json
import os

import mlflow
from mlflow import MlflowClient
from mlflow.deployments import get_deploy_client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--registered-model-name",
        default=os.getenv(
            "REGISTERED_MODEL_NAME",
            "main.models.fraud_detection",
        ),
    )
    parser.add_argument(
        "--model-alias",
        default=os.getenv("MODEL_ALIAS", "champion"),
    )
    parser.add_argument(
        "--endpoint-name",
        default=os.getenv("MODEL_SERVING_ENDPOINT", "fraud-detection"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mlflow.set_registry_uri("databricks-uc")
    model_client = MlflowClient()
    champion = model_client.get_model_version_by_alias(
        args.registered_model_name,
        args.model_alias,
    )
    if champion is None:
        raise RuntimeError(
            f"O modelo {args.registered_model_name} nao possui o alias "
            f"{args.model_alias}."
        )

    version = str(champion.version)
    served_model_name = f"fraud-detection-{args.model_alias}"
    deploy_client = get_deploy_client("databricks")
    deploy_client.update_endpoint_config(
        endpoint=args.endpoint_name,
        config={
            "served_entities": [
                {
                    "name": served_model_name,
                    "entity_name": args.registered_model_name,
                    "entity_version": version,
                    "workload_size": "Small",
                    "scale_to_zero_enabled": True,
                }
            ],
            "traffic_config": {
                "routes": [
                    {
                        "served_model_name": served_model_name,
                        "traffic_percentage": 100,
                    }
                ]
            },
        },
    )

    print(
        json.dumps(
            {
                "endpoint": args.endpoint_name,
                "model_name": args.registered_model_name,
                "model_alias": args.model_alias,
                "model_version": version,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
