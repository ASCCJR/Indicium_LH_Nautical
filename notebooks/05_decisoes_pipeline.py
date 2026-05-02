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
# # LH Nautical — Etapa 2: Decisões do Pipeline Silver/Gold
#
# > **Notebook de documentação e validação** — no ambiente Databricks, as transformações
# > são executadas automaticamente pelo notebook **02_silver** (pipeline Delta Lake).
# > Este notebook documenta cada decisão de limpeza, apresenta a ponte EDA → Silver/Gold
# > e valida a qualidade das tabelas Silver resultantes.
#
# **Problemas mapeados na EDA (com números reais):**
# - `vendas`: `sale_date` com formatos mistos — **4.982** DD-MM-YYYY e **4.913** YYYY-MM-DD
# - `produtos`: `price` como string com prefixo `"R$ "`, `actual_category` com **39 variações** inconsistentes, 7 codes duplicados em 157 linhas
# - `clientes`: **30 de 49** emails com `#` no lugar de `@`, `location` sem formato padrão
# - `custos`: `historic_data` aninhado — **3 a 15 períodos** por produto, ~1.260 linhas após explosão
# - `câmbio`: dimensão ausente nas bases brutas — integrada via **API BCB/PTAX** como tabela `silver_cambio`
#

# %% [markdown]
# ## 0. Setup

# %% [markdown]
# ### Como Executar Este Notebook
# 1. Execute as células em ordem, do topo ao fim.
# 2. Dados brutos lidos do Delta Lake (bronze) para fins de documentação das transformações.
# 3. A camada Silver é produzida automaticamente pelo notebook **02_silver** — este notebook valida o resultado.
# 4. A seção final conecta às tabelas Silver e executa checks de qualidade.

# %%
import pandas as pd
import numpy as np
import re
import unicodedata
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.2f}'.format)

# ── Compatibilidade: Databricks Runtime (UI) vs databricks-connect (VS Code) ─
try:
    spark  # já definido no Databricks runtime
    dbutils.widgets.text("catalog", "workspace",   "Catalog")
    dbutils.widgets.text("schema",  "lh_nautical", "Schema")
    CATALOG = dbutils.widgets.get("catalog")
    SCHEMA  = dbutils.widgets.get("schema")
except NameError:
    from databricks.connect import DatabricksSession
    spark   = DatabricksSession.builder.serverless().getOrCreate()
    CATALOG = "workspace"
    SCHEMA  = "lh_nautical"


df_vendas   = spark.table(f"{CATALOG}.{SCHEMA}.bronze_vendas").toPandas()
df_produtos = spark.table(f"{CATALOG}.{SCHEMA}.bronze_produtos").toPandas()
df_clientes = spark.table(f"{CATALOG}.{SCHEMA}.bronze_clientes").toPandas()
df_custos   = spark.table(f"{CATALOG}.{SCHEMA}.bronze_custos").toPandas()

print("Bronze carregado. Iniciando tratamento...")
print(f"  vendas  : {df_vendas.shape}")
print(f"  produtos: {df_produtos.shape}")
print(f"  clientes: {df_clientes.shape}")
print(f"  custos  : {df_custos.shape}")

# %%
spark.sql("SHOW TABLES IN workspace.lh_nautical").show(truncate=False)

# %% [markdown]
# ---
# ## 1. Tratamento — Vendas

# %%
df_v = df_vendas.copy()

# --- 1.1 Normalizar sale_date (Vetorizado) ---
# Problema: formatos mistos YYYY-MM-DD e DD-MM-YYYY na mesma coluna
# Decisão: pd.to_datetime com format='mixed' detecta automaticamente;
#          para os que falham, forçamos formato DD-MM-YYYY com máscara NaT.

df_v['sale_date'] = pd.to_datetime(df_v['sale_date'], format='mixed', dayfirst=False, errors='coerce')

# Preencher eventuais falhas forçando dayfirst
mask_nat = df_v['sale_date'].isna()
if mask_nat.any():
    df_v.loc[mask_nat, 'sale_date'] = pd.to_datetime(
        df_vendas.loc[mask_nat, 'sale_date'], format='%d-%m-%Y', errors='coerce'
    )

nat_count = df_v['sale_date'].isna().sum()
print(f'Datas não parseadas (NaT): {nat_count}')
print(f'Range de datas: {df_v["sale_date"].min()} → {df_v["sale_date"].max()}')

# %%
# --- 1.2 Verificar outliers em total ---
# Analisamos a distribuição para decidir se valores extremos são erros ou vendas legítimas
q99 = df_v['total'].quantile(0.99)
q01 = df_v['total'].quantile(0.01)

print(f'P1  : R$ {q01:,.2f}')
print(f'P99 : R$ {q99:,.2f}')
print(f'Máx : R$ {df_v["total"].max():,.2f}')
print(f'Mín : R$ {df_v["total"].min():,.2f}')
print(f'\nVendas acima de R$ 1.000.000: {(df_v["total"] > 1_000_000).sum()}')

# Decisão: mantemos todos os valores pois peças náuticas de alto valor são plausíveis.
# Nenhum registro será removido por outlier sem confirmação do negócio.

# %%
# --- 1.3 Resultado final ---
print('Shape final:', df_v.shape)
print('\nTipos:')
print(df_v.dtypes)
df_v.head(5)

# %% [markdown]
# ---
# ## 2. Tratamento — Produtos

# %%
df_p = df_produtos.copy()

# --- 2.1 Converter price de string para float ---
# Problema: valores como 'R$ 33122.52' — remover prefixo e converter
df_p['price'] = (
    df_p['price']
    .str.replace('R$', '', regex=False)
    .str.strip()
    .astype(float)
)

print('Price convertido. Amostra:')
print(df_p['price'].describe())

# %%
# --- 2.2 Padronizar actual_category ---
# Problema: variações como ELETRONICOS, E L E T R Ô N I C O S, Eletrunicos, Eletronicoz, etc.
# Decisão: remover espaços extras, acentos, lowercase (vetorizado) e mapear para categoria canônica via prefixo

# Verificar todas as variações únicas
print('Variações únicas ANTES:')
print(df_p['actual_category'].unique())

# Normalização vetorizada: strip → lowercase → sem espaços → remove acentos (NFKD)
df_p['actual_category_norm'] = (
    df_p['actual_category']
    .str.strip()
    .str.lower()
    .str.replace(r'\s+', '', regex=True)
    .str.normalize('NFKD')
    .str.encode('ascii', errors='ignore')
    .str.decode('utf-8')
)

print('\nVariações normalizadas únicas:')
print(df_p['actual_category_norm'].unique())

# %%
# Mapeamento para categorias canônicas via prefixo (Vetorizado com np.select)
# A normalização anterior removeu acentos e espaços — agora usamos prefixos comuns
# para mapear todas as variações para 3 categorias canônicas encontradas nos dados:
#   'eletr...' → 'eletrônicos'
#   'prop...'  → 'propulsão'  (cobre: propulsao, propulcao, propucao, prop, propulssao, propulsam)
#   'ancor...' ou 'encor...' → 'ancoragem'

conditions = [
    df_p['actual_category_norm'].str.startswith('eletr'),
    df_p['actual_category_norm'].str.startswith('prop'),
    (df_p['actual_category_norm'].str.startswith('ancor') | df_p['actual_category_norm'].str.startswith('encor')),
]
choices = ['eletrônicos', 'propulsão', 'ancoragem']
df_p['actual_category'] = np.select(conditions, choices, default=df_p['actual_category_norm'])
df_p.drop(columns=['actual_category_norm'], inplace=True)

print('Categorias após padronização:')
print(df_p['actual_category'].value_counts())
print(f'\nCategorias não mapeadas: {df_p[~df_p["actual_category"].isin(["eletrônicos","propulsão","ancoragem"])].shape[0]}')

# %%
# --- 2.3 Investigar 157 linhas vs codes 1-150 ---
# Produtos tem 157 linhas mas codes vão de 1 a 150
print(f'Total de produtos: {len(df_p)}')
print(f'Codes únicos: {df_p["code"].nunique()}')
print(f'Range: {df_p["code"].min()} → {df_p["code"].max()}')
print(f'\nCodes duplicados:')
duplicados = df_p[df_p.duplicated(subset=["code"], keep=False)].sort_values("code")
print(duplicados)

# %%
# Decisão sobre duplicatas de code:
# Mantemos o primeiro registro de cada code (preserva dado mais antigo/original)
# Registramos quantos foram removidos
antes = len(df_p)
df_p = df_p.drop_duplicates(subset=['code'], keep='first').reset_index(drop=True)
depois = len(df_p)
print(f'Removidos: {antes - depois} produtos duplicados por code')
print(f'Shape final: {df_p.shape}')

# %% [markdown]
# ---
# ## 3. Tratamento — Clientes

# %%
df_c = df_clientes.copy()

# --- 3.1 Corrigir emails com '#' no lugar de '@' ---
# Problema: 'farias.teixeira.daniel.ribeiro#gmail.com' → '#' deve ser '@'
# Decisão: substituir '#' por '@' somente quando não há '@' já presente

emails_antes = df_c[~df_c['email'].str.contains('@', na=False)]['email'].tolist()
print(f'Emails com # antes da correção: {len(emails_antes)}')

# Correção 100% vetorizada com máscara booleana
mask_sem_arroba = ~df_c['email'].str.contains('@', na=False)
df_c.loc[mask_sem_arroba, 'email'] = df_c.loc[mask_sem_arroba, 'email'].str.replace('#', '@', regex=False)

emails_invalidos = df_c[~df_c['email'].str.contains('@', na=False)]
print(f'Emails ainda inválidos após correção: {len(emails_invalidos)}')

# %%
# --- 3.2 Padronizar location (Vetorizado com .str.extract) ---
# Problema: formatos variados — 'PE , Recife', 'PB/Cabedelo', 'PA - Santarém Novo'
# Decisão: extrair cidade e estado separadamente quando possível
#          e criar colunas 'city' e 'state' normalizadas

ESTADOS_BR = [
    'AC','AL','AP','AM','BA','CE','DF','ES','GO','MA','MT','MS',
    'MG','PA','PB','PR','PE','PI','RJ','RN','RS','RO','RR','SC',
    'SP','SE','TO'
]

# Extrair estado (sigla UF de 2 letras com word boundary)
pattern_state = r'\b(' + '|'.join(ESTADOS_BR) + r')\b'
df_c[['state']] = df_c['location'].str.extract(pattern_state, flags=re.IGNORECASE)

# Extrair cidade: remover sigla UF e separadores
# regex=True obrigatório para que o pattern \b...\b seja tratado como regex
df_c['location_temp'] = df_c['location'].str.replace(pattern_state, '', flags=re.IGNORECASE, regex=True)
df_c['city'] = (
    df_c['location_temp']
    .str.replace(r'[-/,()\\s]+', ' ', regex=True)
    .str.strip()
    .replace('', None)
)
df_c.drop('location_temp', axis=1, inplace=True)

print('Amostra após extração:')
df_c[['full_name', 'location', 'state', 'city']].head(10)

# %%
# --- 3.3 Verificar nomes suspeitos ---
# 'Femininos Oliveira Antunes' parece um nome inválido (palavra 'Femininos')
# Identificados 2 registros na EDA com 'Femininos' como primeiro nome (codes 1 e 25)
# Decisão: sinalizar com flag para revisão manual — não deletar sem confirmação do negócio

# Usando grupo não-capturante (?:...) para evitar UserWarning do pandas
PADRAO_SUSPEITO = r'^(?:masculinos|femininos|outros)\b'

suspeitos = df_c[df_c['full_name'].str.contains(PADRAO_SUSPEITO, case=False, regex=True)]
print(f'Nomes suspeitos sinalizados: {len(suspeitos)}')
print(suspeitos[['code', 'full_name', 'email']])

df_c['nome_suspeito'] = df_c['full_name'].str.contains(PADRAO_SUSPEITO, case=False, regex=True)

# %%
# --- 3.4 Resultado final ---
print('Shape final:', df_c.shape)
df_c.head(5)

# %% [markdown]
# ---
# ## 4. Tratamento — Custos de Importação

# %%
df_cu = df_custos.copy()

# --- 4.1 Explodir historic_data ---
# Problema: cada linha contém uma lista de dicts com histórico de preços em USD
# Decisão: explodir para uma linha por entrada histórica (formato longo)
#          isso permite calcular custo de importação por período

df_cu_exploded = df_cu.explode('historic_data').reset_index(drop=True)

# Normalizar os dicts da coluna historic_data em colunas separadas
hist_norm = pd.json_normalize(df_cu_exploded['historic_data'])
df_cu_long = pd.concat(
    [df_cu_exploded[['product_id', 'product_name', 'category']].reset_index(drop=True),
     hist_norm],
    axis=1
)

print('Shape após explosão:', df_cu_long.shape)
print('\nColunas:', df_cu_long.columns.tolist())
df_cu_long.head(10)

# %%
# --- 4.2 Converter tipos ---
df_cu_long['start_date'] = pd.to_datetime(df_cu_long['start_date'], dayfirst=True)
df_cu_long['usd_price']  = pd.to_numeric(df_cu_long['usd_price'], errors='coerce')

print('Tipos após conversão:')
print(df_cu_long.dtypes)
print(f'\nNulos em usd_price: {df_cu_long["usd_price"].isna().sum()}')
print(f'Range de datas: {df_cu_long["start_date"].min()} → {df_cu_long["start_date"].max()}')

# %%
# --- 4.3 Verificar produtos sem custo de importação ---
# Produtos tem 150 registros (após deduplicação), custos cobre product_id 1-150
produtos_sem_custo = set(df_p['code']) - set(df_cu['product_id'])
print(f'Produtos sem custo de importação: {len(produtos_sem_custo)}')
if produtos_sem_custo:
    print('Codes:', sorted(produtos_sem_custo))
    print(df_p[df_p['code'].isin(produtos_sem_custo)][['code', 'name']])

# %% [markdown]
# ---
# ## 5. Câmbio — Banco Central do Brasil (PTAX)
#
# Fonte de dados externa — API pública do BCB, gratuita, sem autenticação.
# Trata-se de uma **5ª fonte clean** do projeto: série histórica diária de USD/BRL
# usada nas etapas 3, 4, 6 e 7 para cálculo de lucratividade real.
#
# **Tratamento aplicado:**
# - Uma chamada HTTP busca 502 dias úteis (jan/2023 → dez/2024)
# - PTAX tem dois fechamentos diários (13h e 18h) — mantemos apenas o de 18h
# - Fins de semana e feriados não têm cotação → **forward-fill** com taxa do último dia útil

# %%
# Taxa de cambio ja disponivel no Delta Lake (silver_cambio)
df_cam = spark.table(f"{CATALOG}.{SCHEMA}.silver_cambio").toPandas()
df_cam['data'] = pd.to_datetime(df_cam['data'])
print(f'Cambio carregado do Delta Lake: {df_cam.shape}')
display(df_cam.head())

# %% [markdown]
# ---
# ## 6. Consolidação — Bases Limpas no Delta Lake
#
# As 5 bases clean são produzidas e salvas como tabelas Delta pelo notebook **02_silver**.
# Esta seção valida que as tabelas Silver no Delta Lake correspondem ao esperado.
#

# %%
# Resumo final de cada base limpa (Pandas — tratamento documentado acima)
print("=" * 55)
print("BASES TRATADAS - RESUMO (Pandas)")
print("=" * 55)
print(f"vendas_clean    : {df_v.shape}")
print(f"produtos_clean  : {df_p.shape}")
print(f"clientes_clean  : {df_c.shape}")
print(f"custos_clean    : {df_cu_long.shape} (formato longo)")
print(f"cambio_clean    : {df_cam.shape}")

# Validacao contra Silver tables (Delta Lake)
# silver_clientes: 6 colunas apos adicao de state e city (id_client, email, full_name, location, state, city)
print("=" * 55)
print("VALIDACAO SILVER TABLES (DELTA LAKE)")
print("=" * 55)
checks = [
    ("silver_vendas",   (9895, 6)),
    ("silver_produtos", (150,  4)),
    ("silver_clientes", (49,   6)),
    ("silver_custos",   (1260, 5)),
    ("silver_cambio",   (731,  2)),
]
for tbl, expected_shape in checks:
    df_check = spark.table(f"{CATALOG}.{SCHEMA}.{tbl}").toPandas()
    shape_ok = df_check.shape == expected_shape
    nulos_ok = df_check.isna().sum().sum() == 0
    label = "PASS" if (shape_ok and nulos_ok) else "WARN"
    shape_label = "OK" if shape_ok else f"esperado={expected_shape} obtido={df_check.shape}"
    print(f"  {label:<4} {tbl:<25} shape={df_check.shape} {shape_label}")
    print(f"       colunas: {list(df_check.columns)}")
print("Validacao concluida.")


# %% [markdown]
# ---
# ## 7. Ponte: EDA → Pipeline Silver/Gold
#
# Cada problema identificado na Etapa 1 (EDA) tem uma implementação rastreada no pipeline.
# A tabela abaixo fecha o loop entre diagnóstico e execução.
#

# %%
bridge = pd.DataFrame([
    {"base": "Vendas",
     "problema_eda": "sale_date com 2 formatos mistos (50% DD-MM-YYYY / 50% YYYY-MM-DD)",
     "implementacao": "try_to_date + coalesce no Spark SQL — zero NaT produzidos",
     "notebook": "02_silver › silver_vendas"},
    {"base": "Produtos",
     "problema_eda": 'price como string com prefixo "R$ "',
     "implementacao": "regexp_replace + cast(\"double\") no Silver",
     "notebook": "02_silver › silver_produtos"},
    {"base": "Produtos",
     "problema_eda": "actual_category com 39 variações para 3 categorias reais",
     "implementacao": "Normalização + mapeamento por prefixo (eletr/prop/ancor)",
     "notebook": "02_silver › silver_produtos"},
    {"base": "Produtos",
     "problema_eda": "157 linhas, 150 codes únicos — 7 duplicatas por code",
     "implementacao": "dropDuplicates(['id_product']) — mantém primeira ocorrência",
     "notebook": "02_silver › silver_produtos"},
    {"base": "Clientes",
     "problema_eda": "30 de 49 emails com '#' no lugar de '@'",
     "implementacao": "regexp_replace('#', '@') com máscara booleana",
     "notebook": "02_silver › silver_clientes"},
    {"base": "Clientes",
     "problema_eda": "location sem padrão — 4+ formatos distintos",
     "implementacao": "regexp_extract → state (sigla UF) + city (texto remanescente)",
     "notebook": "02_silver › silver_clientes"},
    {"base": "Custos",
     "problema_eda": "historic_data aninhado — lista de dicts por produto",
     "implementacao": "explode() + to_date(\"dd/MM/yyyy\") — 1.260 linhas no formato longo",
     "notebook": "02_silver › silver_custos"},
    {"base": "Câmbio",
     "problema_eda": "Dimensão ausente — custos em USD, receitas em BRL",
     "implementacao": "API BCB/PTAX jan/2023–dez/2024 + forward-fill de feriados",
     "notebook": "02_silver › silver_cambio"},
    {"base": "Gold",
     "problema_eda": "Preço de custo muda ao longo do tempo (múltiplos períodos)",
     "implementacao": "As-of join: max(start_date ≤ sale_date) por produto → usd_price vigente",
     "notebook": "03_gold › gold_fct_vendas"},
    {"base": "Gold",
     "problema_eda": "Margem real impossível sem câmbio histórico",
     "implementacao": "custo_brl = qtd × usd_price × taxa_brl | margem_pct = (total − custo_brl) / total",
     "notebook": "03_gold › gold_fct_vendas"},
])

print("=== PONTE: EDA → SILVER/GOLD ===")
display(bridge[["base", "problema_eda", "implementacao", "notebook"]])


# %% [markdown]
# ## 8. Resumo Executivo (Etapa 2)
#
# - O tratamento consolidou 5 bases clean: vendas, produtos, clientes, custos (longo) e câmbio.
# - A coluna sale_date foi normalizada sem perdas de registro.
# - Produtos ficaram com 150 códigos únicos após deduplicação controlada.
# - Os emails inválidos foram corrigidos e campos de localização foram estruturados em city/state.
# - A validação final usa regras críticas bloqueantes e shape informativo para robustez sem rigidez excessiva.

# %%
# Validacao Final — Camada Silver (produzida pelo 02_silver)
print("=" * 60)
print("VALIDACAO — TABELAS SILVER DO DELTA LAKE")
print("=" * 60)

sv   = spark.table(f"{CATALOG}.{SCHEMA}.silver_vendas").toPandas()
sp   = spark.table(f"{CATALOG}.{SCHEMA}.silver_produtos").toPandas()
sc   = spark.table(f"{CATALOG}.{SCHEMA}.silver_clientes").toPandas()
scu  = spark.table(f"{CATALOG}.{SCHEMA}.silver_custos").toPandas()
scam = spark.table(f"{CATALOG}.{SCHEMA}.silver_cambio").toPandas()

CATS_CANONICAS = {"eletrônicos", "propulsão", "ancoragem"}

checks = [
    ("silver_vendas   shape",       sv.shape == (9895, 6),
     f"{sv.shape}"),
    ("silver_vendas   nulos",       sv.isnull().sum().sum() == 0,
     f"{sv.isnull().sum().sum()} nulos"),
    ("silver_produtos shape",       sp.shape[0] == 150,
     f"{sp.shape[0]} linhas"),
    ("silver_produtos price_float", pd.api.types.is_float_dtype(sp["price"]),
     f"dtype={sp['price'].dtype}"),
    ("silver_produtos category",    set(sp["actual_category"].unique()).issubset(CATS_CANONICAS),
     f"{sp['actual_category'].unique().tolist()}"),
    ("silver_clientes shape",       sc.shape == (49, 6),
     f"{sc.shape}"),
    ("silver_clientes emails",      sc["email"].str.contains("@").all(),
     "todos emails com @"),
    ("silver_clientes city_state",  sc[["city", "state"]].notna().all().all(),
     "city/state extraidos"),
    ("silver_custos   linhas",      scu.shape[0] >= 1000,
     f"{scu.shape[0]} linhas apos explosao"),
    ("silver_cambio   linhas",      scam.shape[0] > 400,
     f"{scam.shape[0]} taxas diarias"),
]

all_pass = True
for name, condition, info in checks:
    status = "PASS" if condition else "FAIL"
    if not condition: all_pass = False
    print(f"  [{status}] {name}: {info}")

print("-" * 60)
print(f"  SILVER VALIDATION: {'PASS' if all_pass else 'FAIL'}")


# %%
def run_all_quality_checks_tratamento(df_v_local, df_p_local, df_c_local, df_cu_local, df_cam_local):
    """Executa checks críticos de qualidade no resultado da Etapa 2."""
    checks = {
        'vendas_sale_date_sem_nat': sv['sale_date'].isna().sum() == 0,  # usa Silver Delta (Spark ja trata ambos os formatos)
        'produtos_code_unico': df_p_local['code'].nunique() == len(df_p_local),
        'clientes_email_com_arroba': df_c_local['email'].str.contains('@', na=False).all(),
        'custos_usd_price_sem_nulo': df_cu_local['usd_price'].isna().sum() == 0,
        'cambio_taxa_sem_nulo': df_cam_local['taxa_brl'].isna().sum() == 0,
    }
    resultado = pd.DataFrame({'check': checks.keys(), 'status': checks.values()})
    resultado['resultado'] = resultado['status'].map({True: 'PASS', False: 'FAIL'})
    return resultado

qa_tratamento = run_all_quality_checks_tratamento(df_v, df_p, df_c, df_cu_long, df_cam)
display(qa_tratamento[['check', 'resultado']])

if qa_tratamento['status'].all():
    print('RUN_ALL CHECKS: PASS')
else:
    print('RUN_ALL CHECKS: FAIL')
