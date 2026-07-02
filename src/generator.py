from datetime import datetime, timedelta
import random
from uuid import uuid4

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import (
    StructType,
    StructField,
    IntegerType,
    DoubleType,
    StringType,
    BooleanType,
    TimestampType,
)
from pyspark.sql.functions import col, lit, rand, when

from src.data import *

spark = SparkSession.builder.getOrCreate()


def generate_fraud_labels(df: DataFrame) -> DataFrame:
    """
    Randomly assigns fraud labels:
    - 45% True
    - 45% False
    - 10% NULL (unlabeled)
    """

    return (
        df.withColumn("_rand", rand(seed=42))
          .withColumn(
              "fraude",
              when(col("_rand") < 0.45, lit(True))
               .when(col("_rand") < 0.90, lit(False))
               .otherwise(lit(None).cast("boolean"))
          )
          .drop("_rand")
    )


def generate_feature_store_data(
    spark: SparkSession,
    n_transacoes: int = 10000,
    seed: int = 42
    ):

    random.seed(seed)

    clientes = {}

    agora = datetime.now()

    rows = []

    for _ in range(n_transacoes):

        id_transacao = str(uuid4())

        id_cliente = random.randint(1, max(100, n_transacoes // 10))

        if id_cliente not in clientes:

            media = random.uniform(80, 1200)

            clientes[id_cliente] = {
                "media": media,
                "ultima_data": agora - timedelta(days=random.randint(1, 30)),
                "ultima_lat": random.uniform(-33, -1),
                "ultima_lon": random.uniform(-73, -34),
                "historico": [],
            }

        cliente = clientes[id_cliente]

        data = cliente["ultima_data"] + timedelta(
            minutes=random.randint(5, 5000)
        )

        valor = max(
            5,
            random.gauss(cliente["media"], cliente["media"] * 0.45)
        )

        historico = cliente["historico"]

        historico.append(valor)

        media_30d = sum(historico) / len(historico)

        if len(historico) == 1:
            desvio = 0.0
        else:
            m = media_30d
            desvio = (
                sum((x - m) ** 2 for x in historico) / len(historico)
            ) ** 0.5

        maior = max(historico)

        valor_relativo = valor / media_30d if media_30d else 1

        compras_24h = random.randint(1, 12)
        compras_7dias = compras_24h + random.randint(0, 40)
        compras_ultima_hora = random.randint(0, 5)

        minutos_desde_ultima = (
            data - cliente["ultima_data"]
        ).total_seconds() / 60

        lat = random.uniform(-33, -1)
        lon = random.uniform(-73, -34)

        distancia = (
            ((lat - cliente["ultima_lat"]) ** 2 +
             (lon - cliente["ultima_lon"]) ** 2)
            ** 0.5
        ) * 111

        idade = random.randint(18, 80)

        tempo_rel = random.randint(30, 3650)

        renda = random.randint(1500, 35000)

        score = random.randint(300, 900)

        limite = renda * random.uniform(1.0, 4.5)

        saldo = random.uniform(0, limite)

        risco_estabelecimento = round(random.uniform(0, 1), 2)

        historico_fraudes = round(random.uniform(0, 0.15), 3)

        row = (
            id_transacao,
            id_cliente,
            round(valor, 2),
            "BRL",
            random.choice(canais),
            random.choice(formas_pagamento),
            random.choice(categorias),
            random.randint(1, 12),
            data,
            data.hour,
            data.strftime("%A"),
            idade,
            tempo_rel,
            renda,
            score,
            round(limite, 2),
            round(saldo, 2),
            compras_ultima_hora,
            compras_24h,
            compras_7dias,
            round(sum(historico[-compras_24h:]), 2),
            round(media_30d, 2),
            round(desvio, 2),
            round(maior, 2),
            round(minutos_desde_ultima, 2),
            round(distancia, 2),
            random.choice([True, False]),
            random.randint(1, 1000),
            random.choice(sistemas),
            random.choice(navegadores),
            random.choice(modelos),
            random.choice([True, False]),
            random.choice([True, False]),
            "Brasil",
            random.choice(estados),
            random.choice(cidades),
            round(risco_estabelecimento, 2),
            historico_fraudes,
            round(valor_relativo, 2),
            valor > media_30d * 2,
        )

        rows.append(row)

        cliente["ultima_data"] = data
        cliente["ultima_lat"] = lat
        cliente["ultima_lon"] = lon

    schema = StructType([
        StructField("id_transacao", StringType()),
        StructField("id_cliente", IntegerType()),
        StructField("valor", DoubleType()),
        StructField("moeda", StringType()),
        StructField("canal", StringType()),
        StructField("forma_pagamento", StringType()),
        StructField("categoria_estabelecimento", StringType()),
        StructField("parcelas", IntegerType()),
        StructField("data_hora", TimestampType()),
        StructField("hora", IntegerType()),
        StructField("dia_semana", StringType()),
        StructField("idade_cliente", IntegerType()),
        StructField("tempo_relacionamento_dias", IntegerType()),
        StructField("renda", IntegerType()),
        StructField("score_credito", IntegerType()),
        StructField("limite_credito", DoubleType()),
        StructField("saldo_atual", DoubleType()),
        StructField("compras_ultima_hora", IntegerType()),
        StructField("compras_24h", IntegerType()),
        StructField("compras_7dias", IntegerType()),
        StructField("valor_total_24h", DoubleType()),
        StructField("media_valor_30d", DoubleType()),
        StructField("desvio_padrao_30d", DoubleType()),
        StructField("maior_compra_30d", DoubleType()),
        StructField("minutos_desde_ultima_compra", DoubleType()),
        StructField("distancia_da_ultima_compra", DoubleType()),
        StructField("dispositivo_conhecido", BooleanType()),
        StructField("idade_dispositivo", IntegerType()),
        StructField("sistema_operacional", StringType()),
        StructField("navegador", StringType()),
        StructField("modelo_dispositivo", StringType()),
        StructField("vpn_detectada", BooleanType()),
        StructField("proxy_detectado", BooleanType()),
        StructField("pais", StringType()),
        StructField("estado", StringType()),
        StructField("cidade", StringType()),
        StructField("risco_estabelecimento", DoubleType()),
        StructField("historico_fraudes_estabelecimento", DoubleType()),
        StructField("valor_relativo", DoubleType()),
        StructField("acima_percentil95", BooleanType()),
    ])

    return spark.createDataFrame(rows, schema)
