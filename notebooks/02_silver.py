# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: .venv-databricks (3.11.0)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # LH Nautical - Camada Silver

# %%
from pyspark.sql.functions import (
    col, explode, to_date, expr, coalesce,
    when, regexp_replace, regexp_extract, trim, lower
)

# Silver Vendas
silver_vendas = (spark.table("workspace.lh_nautical.bronze_vendas")
    .withColumn("sale_date",
        expr("coalesce(try_to_date(sale_date, 'dd-MM-yyyy'), try_to_date(sale_date, 'yyyy-MM-dd'))"))
    .dropDuplicates(["id"])
    .filter(col("total") > 0))
spark.sql("DROP TABLE IF EXISTS workspace.lh_nautical.silver_vendas")
silver_vendas.write.format("delta").saveAsTable("workspace.lh_nautical.silver_vendas")
print(f"silver_vendas: {spark.table('workspace.lh_nautical.silver_vendas').count()} linhas")

# %%
# Silver Clientes — correcao de emails + extracao de city/state
# Identificado na EDA: 30/49 emails com "#" no lugar de "@"; location sem padrao uniforme
_ESTADOS = "AC|AL|AP|AM|BA|CE|DF|ES|GO|MA|MT|MS|MG|PA|PB|PR|PE|PI|RJ|RN|RS|RO|RR|SC|SP|SE|TO"

silver_clientes = (spark.table("workspace.lh_nautical.bronze_clientes")
    .withColumnRenamed("code", "id_client")
    # 30 de 49 emails tinham "#" no lugar de "@"
    .withColumn("email",
        when(~col("email").contains("@"),
             regexp_replace(col("email"), "#", "@"))
        .otherwise(col("email")))
    # Extrair estado (sigla UF) e cidade de location
    .withColumn("state",
        regexp_extract(col("location"), r"\b(" + _ESTADOS + r")\b", 1))
    .withColumn("city",
        trim(regexp_replace(
            regexp_replace(col("location"), r"\b(" + _ESTADOS + r")\b", ""),
            r"[-/,\s]+", " ")))
    .dropDuplicates(["id_client"])
    .filter(col("id_client").isNotNull()))
spark.sql("DROP TABLE IF EXISTS workspace.lh_nautical.silver_clientes")
silver_clientes.write.format("delta").saveAsTable("workspace.lh_nautical.silver_clientes")
print(f"silver_clientes: {spark.table('workspace.lh_nautical.silver_clientes').count()} linhas")

# %%
# Silver Produtos — conversao de price + normalizacao de actual_category
# Identificado na EDA: price como string "R$ xxx"; 39 variacoes de categoria para 3 reais
silver_produtos = (spark.table("workspace.lh_nautical.bronze_produtos")
    .withColumnRenamed("code", "id_product")
    # Converter price de string "R$ 33122.52" para double
    .withColumn("price",
        trim(regexp_replace(col("price").cast("string"), r"R\$\s*", "")).cast("double"))
    # Mapear 39 variacoes de actual_category para 3 categorias canonicas
    .withColumn("_cat",
        lower(regexp_replace(col("actual_category"), r"\s+", "")))
    .withColumn("actual_category",
        when(col("_cat").startswith("eletr"), "eletrônicos")
        .when(col("_cat").startswith("prop"), "propulsão")
        .when(col("_cat").startswith("ancor") | col("_cat").startswith("encor"), "ancoragem")
        .otherwise(col("actual_category")))
    .drop("_cat")
    .dropDuplicates(["id_product"])
    .filter(col("id_product").isNotNull()))
spark.sql("DROP TABLE IF EXISTS workspace.lh_nautical.silver_produtos")
silver_produtos.write.format("delta").saveAsTable("workspace.lh_nautical.silver_produtos")
print(f"silver_produtos: {spark.table('workspace.lh_nautical.silver_produtos').count()} linhas")

# %%
# Silver Custos - explode historic_data (1 linha por periodo de preco)
silver_custos = (spark.table("workspace.lh_nautical.bronze_custos")
    .select(
        col("product_id").alias("id_product"),
        col("product_name"),
        col("category"),
        explode(col("historic_data")).alias("hist")
    )
    .select(
        col("id_product"),
        col("product_name"),
        col("category"),
        to_date(col("hist.start_date"), "dd/MM/yyyy").alias("start_date"),
        col("hist.usd_price").alias("usd_price")
    )
    .filter(col("usd_price") > 0)
    .dropDuplicates())
spark.sql("DROP TABLE IF EXISTS workspace.lh_nautical.silver_custos")
silver_custos.write.format("delta").saveAsTable("workspace.lh_nautical.silver_custos")
print(f"silver_custos: {spark.table('workspace.lh_nautical.silver_custos').count()} linhas")

# %%
# Silver Cambio - busca USD/BRL diretamente da API do BCB (PTAX)
import requests
import pandas as pd

url_bcb = (
    "https://olinda.bcb.gov.br/olinda/servico/PTAX/versao/v1/odata/"
    "CotacaoDolarPeriodo(dataInicial=@di,dataFinalCotacao=@df)"
    "?@di='01-01-2023'&@df='12-31-2024'&$format=json&$select=cotacaoVenda,dataHoraCotacao"
)
resp = requests.get(url_bcb, timeout=30)
resp.raise_for_status()

df_cam = pd.DataFrame(resp.json()["value"])
df_cam["data"] = pd.to_datetime(df_cam["dataHoraCotacao"]).dt.normalize()
df_cam_diario = (df_cam
    .groupby("data")["cotacaoVenda"]
    .mean()
    .reset_index())

# Preencher fins de semana e feriados com o ultimo valor disponivel (forward fill)
calendario = pd.DataFrame({"data": pd.date_range("2023-01-01", "2024-12-31", freq="D")})
df_final = calendario.merge(df_cam_diario, on="data", how="left")
df_final["taxa_brl"] = df_final["cotacaoVenda"].ffill().bfill().astype(float)
df_final = df_final[["data", "taxa_brl"]]

silver_cambio = (spark.createDataFrame(df_final)
    .withColumn("data", to_date(col("data")))
    .dropDuplicates(["data"]))
spark.sql("DROP TABLE IF EXISTS workspace.lh_nautical.silver_cambio")
silver_cambio.write.format("delta").saveAsTable("workspace.lh_nautical.silver_cambio")
print(f"silver_cambio: {spark.table('workspace.lh_nautical.silver_cambio').count()} linhas")

# %%
print("=== RESUMO SILVER ===")
for t in ["silver_vendas","silver_clientes","silver_produtos","silver_custos","silver_cambio"]:
    n = spark.table(f"workspace.lh_nautical.{t}").count()
    print(f"{t}: {n} linhas")
