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
# # LH Nautical — Camada Bronze

# %%
# Verificar arquivos no volume
files = dbutils.fs.ls("/Volumes/workspace/lh_nautical/raw_files/")
for f in files:
    print(f.name, f"({f.size/1024:.1f} KB)")

# %%
# Bronze Vendas
spark.sql("DROP TABLE IF EXISTS workspace.lh_nautical.bronze_vendas")
bronze_vendas = (spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv("/Volumes/workspace/lh_nautical/raw_files/vendas_2023_2024.csv"))
bronze_vendas.write.format("delta").saveAsTable("workspace.lh_nautical.bronze_vendas")
saved = spark.table("workspace.lh_nautical.bronze_vendas")
print(f"bronze_vendas: {saved.count()} linhas | colunas: {saved.columns}")

# %%
# Bronze Produtos
spark.sql("DROP TABLE IF EXISTS workspace.lh_nautical.bronze_produtos")
bronze_produtos = (spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv("/Volumes/workspace/lh_nautical/raw_files/produtos_raw.csv"))
bronze_produtos.write.format("delta").saveAsTable("workspace.lh_nautical.bronze_produtos")
saved = spark.table("workspace.lh_nautical.bronze_produtos")
print(f"bronze_produtos: {saved.count()} linhas | colunas: {saved.columns}")

# %%
# Bronze Clientes (JSON multiline)
spark.sql("DROP TABLE IF EXISTS workspace.lh_nautical.bronze_clientes")
bronze_clientes = (spark.read
    .option("multiLine", "true")
    .json("/Volumes/workspace/lh_nautical/raw_files/clientes_crm.json"))
bronze_clientes.write.format("delta").saveAsTable("workspace.lh_nautical.bronze_clientes")
saved = spark.table("workspace.lh_nautical.bronze_clientes")
print(f"bronze_clientes: {saved.count()} linhas | colunas: {saved.columns}")

# %%
# Bronze Custos (JSON multiline - mantem historic_data como array para Silver explodir)
spark.sql("DROP TABLE IF EXISTS workspace.lh_nautical.bronze_custos")
bronze_custos = (spark.read
    .option("multiLine", "true")
    .json("/Volumes/workspace/lh_nautical/raw_files/custos_importacao.json"))
bronze_custos.write.format("delta").saveAsTable("workspace.lh_nautical.bronze_custos")
saved = spark.table("workspace.lh_nautical.bronze_custos")
print(f"bronze_custos: {saved.count()} linhas | colunas: {saved.columns}")

# %%
print("=== RESUMO BRONZE ===")
for t in ["bronze_vendas", "bronze_produtos", "bronze_clientes", "bronze_custos"]:
    n = spark.table(f"workspace.lh_nautical.{t}").count()
    print(f"{t}: {n} linhas")
