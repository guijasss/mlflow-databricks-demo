"""Cria ou atualiza o endpoint para a versao apontada pelo alias champion."""

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


def build_endpoint_config(
    *,
    registered_model_name: str,
    model_alias: str,
    model_version: str,
) -> tuple[str, dict]:
    served_model_name = f"fraud-detection-{model_alias}"
    return served_model_name, {
        "served_entities": [
            {
                "name": served_model_name,
                "entity_name": registered_model_name,
                "entity_version": model_version,
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
    }


def get_endpoint_or_none(deploy_client, endpoint_name: str):
    try:
        return deploy_client.get_endpoint(endpoint=endpoint_name)
    except Exception as exc:
        message = str(exc).lower()
        if "not_found" in message or "not found" in message:
            return None
        raise


def get_current_served_entity(endpoint, served_model_name: str):
    config = endpoint.get("config", {}) if isinstance(endpoint, dict) else {}
    served_entities = (
        config.get("served_entities")
        or config.get("served_models")
        or []
    )
    for served_entity in served_entities:
        if served_entity.get("name") == served_model_name:
            return served_entity
    return None


def traffic_points_to_served_entity(endpoint, served_model_name: str) -> bool:
    config = endpoint.get("config", {}) if isinstance(endpoint, dict) else {}
    traffic_config = config.get("traffic_config", {})
    routes = traffic_config.get("routes", [])
    return routes == [
        {
            "served_model_name": served_model_name,
            "traffic_percentage": 100,
        }
    ]


def endpoint_points_to_version(
    endpoint,
    *,
    served_model_name: str,
    registered_model_name: str,
    model_version: str,
) -> bool:
    served_entity = get_current_served_entity(endpoint, served_model_name)
    if served_entity is None:
        return False
    return (
        served_entity.get("entity_name") == registered_model_name
        and str(served_entity.get("entity_version")) == model_version
        and traffic_points_to_served_entity(endpoint, served_model_name)
    )


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
    served_model_name, endpoint_config = build_endpoint_config(
        registered_model_name=args.registered_model_name,
        model_alias=args.model_alias,
        model_version=version,
    )
    deploy_client = get_deploy_client("databricks")
    endpoint = get_endpoint_or_none(deploy_client, args.endpoint_name)
    if endpoint is None:
        deploy_client.create_endpoint(
            name=args.endpoint_name,
            config=endpoint_config,
        )
        action = "created"
    elif endpoint_points_to_version(
        endpoint,
        served_model_name=served_model_name,
        registered_model_name=args.registered_model_name,
        model_version=version,
    ):
        action = "unchanged"
    else:
        deploy_client.update_endpoint_config(
            endpoint=args.endpoint_name,
            config=endpoint_config,
        )
        action = "updated"

    print(
        json.dumps(
            {
                "action": action,
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
