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
# # LH Nautical — Camada Gold: fct_vendas_gold

# %%
from pyspark.sql.functions import col, round as spark_round, year, month, date_format
import pyspark.sql.functions as F

vendas = spark.table("workspace.lh_nautical.silver_vendas")
custos = F.broadcast(spark.table("workspace.lh_nautical.silver_custos"))  # 1260 linhas
cambio = F.broadcast(spark.table("workspace.lh_nautical.silver_cambio"))  # 731 linhas

# Passo 1: combinacoes unicas (produto, data) das vendas -> max start_date do custo vigente
datas_unicas = F.broadcast(vendas.select("id_product", "sale_date").distinct())

agg_pre = (datas_unicas.alias("d")
    .join(custos.alias("c"),
          (col("d.id_product") == col("c.id_product")) &
          (col("c.start_date") <= col("d.sale_date")), "left")
    .groupBy(col("d.id_product"), col("d.sale_date"))
    .agg(F.max("c.start_date").alias("start_date_vigente"))
).alias("pre")  # alias apos groupBy — alias "d" nao existe mais aqui

# Passo 2: join de volta para obter o usd_price vigente
custo_vigente = F.broadcast(agg_pre
    .join(custos.alias("c2"),
          (col("pre.id_product") == col("c2.id_product")) &
          (col("pre.start_date_vigente") == col("c2.start_date")), "left")
    .select(
        col("pre.id_product"),
        col("pre.sale_date"),
        col("c2.usd_price")
    )
)

gold = (vendas.alias("v")
    .join(custo_vigente.alias("cv"),
          (col("v.id_product") == col("cv.id_product")) &
          (col("v.sale_date") == col("cv.sale_date")), "left")
    .join(cambio.alias("cam"), col("v.sale_date") == col("cam.data"), "inner")
    .filter(col("cv.usd_price").isNotNull())
    .select(
        col("v.id"),
        col("v.id_client"),
        col("v.id_product"),
        col("v.qtd"),
        col("v.total"),
        col("v.sale_date"),
        col("cv.usd_price").alias("usd_price_vigente"),
        col("cam.taxa_brl"),
        spark_round(col("v.qtd") * col("cv.usd_price") * col("cam.taxa_brl"), 2).alias("custo_brl"),
        spark_round(col("v.total") - col("v.qtd") * col("cv.usd_price") * col("cam.taxa_brl"), 2).alias("lucro"),
        spark_round(
            (col("v.total") - col("v.qtd") * col("cv.usd_price") * col("cam.taxa_brl"))
            / F.nullif(col("v.total"), F.lit(0)) * 100, 2
        ).alias("margem_pct"),
        year(col("v.sale_date")).alias("ano"),
        month(col("v.sale_date")).alias("mes"),
        date_format(col("v.sale_date"), "yyyy-MM").alias("ano_mes")
    )
)

spark.sql("DROP TABLE IF EXISTS workspace.lh_nautical.gold_fct_vendas")
gold.write.format("delta").saveAsTable("workspace.lh_nautical.gold_fct_vendas")
print(f"gold_fct_vendas: {spark.table('workspace.lh_nautical.gold_fct_vendas').count()} linhas")

# %%
print("=== VALIDACAO GOLD ===")
spark.sql("""
    SELECT
        COUNT(*)                                    AS total_linhas,
        COUNT(DISTINCT id_client)                   AS clientes,
        COUNT(DISTINCT id_product)                  AS produtos,
        ROUND(SUM(total)/1e9, 3)                    AS receita_bi,
        ROUND(SUM(lucro)/1e6, 1)                    AS lucro_m,
        ROUND(SUM(lucro)/SUM(total)*100, 2)         AS margem_pct
    FROM workspace.lh_nautical.gold_fct_vendas
""").show()
