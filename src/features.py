"""
Definições da Feature Store utilizada pelo modelo de detecção de fraudes.

Este módulo centraliza todas as colunas utilizadas pelo projeto para evitar
duplicação de código entre geração de dados, treinamento, inferência e API.
"""

METADATA_COLUMNS = [
    "id_transacao",
    "id_cliente",
]

CATEGORICAL_COLUMNS = [
    "moeda",
    "canal",
    "forma_pagamento",
    "categoria_estabelecimento",
    "dia_semana",
    "sistema_operacional",
    "navegador",
    "modelo_dispositivo",
    "pais",
    "estado",
    "cidade",
]

NUMERIC_COLUMNS = [
    "valor",
    "parcelas",
    "hora",
    "idade_cliente",
    "tempo_relacionamento_dias",
    "renda",
    "score_credito",
    "limite_credito",
    "saldo_atual",
    "compras_ultima_hora",
    "compras_24h",
    "compras_7dias",
    "valor_total_24h",
    "media_valor_30d",
    "desvio_padrao_30d",
    "maior_compra_30d",
    "minutos_desde_ultima_compra",
    "distancia_da_ultima_compra",
    "idade_dispositivo",
    "risco_estabelecimento",
    "historico_fraudes_estabelecimento",
    "valor_relativo",
]

BOOLEAN_COLUMNS = [
    "dispositivo_conhecido",
    "vpn_detectada",
    "proxy_detectado",
    "acima_percentil95",
]

FEATURE_COLUMNS = (
    NUMERIC_COLUMNS +
    BOOLEAN_COLUMNS +
    CATEGORICAL_COLUMNS
)

TARGET_COLUMN = "fraude"

# -----------------------------------------------------------------------------
# Todas as colunas da Feature Store
# -----------------------------------------------------------------------------

ALL_COLUMNS = (
    METADATA_COLUMNS +
    FEATURE_COLUMNS +
    [TARGET_COLUMN]
)