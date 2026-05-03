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
# # LH Nautical — Etapa 1: Análise Exploratória de Dados (EDA)
#
# > **Notebook exploratório** — lê os dados brutos (bronze) do Delta Lake.
# > As transformações documentadas aqui são executadas automaticamente pelo notebook **02_silver** (pipeline Delta Lake).
#
# **Objetivo:** Inspecionar as quatro bases brutas, identificar problemas de qualidade e documentar as decisões que guiarão o tratamento de dados.
#
# **Tabelas bronze lidas:**
# - `bronze_vendas` — histórico de vendas
# - `bronze_produtos` — catálogo de produtos
# - `bronze_clientes` — dados de clientes
# - `bronze_custos` — custos de importação por produto (em **USD**)
#
# > **Dimensão ausente:** custos estão em USD, receitas em BRL. A conversão para calcular margem real exige câmbio histórico — dado **não presente** nas bases brutas. Integrado via API BCB/PTAX pelo notebook 02_silver como tabela `silver_cambio`.

# %% [markdown]
# ## 0. Setup — Bibliotecas e Caminhos

# %% [markdown]
# ### Como Executar Este Notebook
# 1. Execute as células em ordem, do topo ao fim.
# 2. Dados lidos diretamente do Delta Lake (bronze) — sem dependências locais.
# 3. Esta etapa é diagnóstica: identifica problemas e define decisões para o pipeline Silver.
# 4. O checklist final separa diagnóstico do estado bruto e critérios de Go/No-Go.

# %%
import pandas as pd
import numpy as np
import ast
import re

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

print('Bases carregadas do Delta Lake!')
print(f'  vendas    : {df_vendas.shape}')
print(f'  produtos  : {df_produtos.shape}')
print(f'  clientes  : {df_clientes.shape}')
print(f'  custos    : {df_custos.shape}')

# %%
spark.sql("SHOW TABLES IN workspace.lh_nautical").show(truncate=False)

# %% [markdown]
# ## 1. Carregamento das Bases

# %%
# df_vendas, df_produtos, df_clientes, df_custos ja carregados no setup via spark.table()

# %% [markdown]
# ## 2. Inspeção Geral de Cada Base
#
# Para cada base vamos verificar:
# - Primeiras linhas
# - Tipos de dados
# - Nulos
# - Duplicatas

# %% [markdown]
# ### 2.1 Vendas

# %%
df_vendas.head(10)

# %%
df_vendas.info()

# %%
print('=== NULOS ===')
print(df_vendas.isnull().sum())
print(f'\n=== DUPLICATAS ===')
print(f'Linhas duplicadas: {df_vendas.duplicated().sum()}')

# Range do campo id — confirma as lacunas mencionadas no resumo
print(f'\n=== RANGE DO CAMPO id ===')
print(df_vendas['id'].agg(['min', 'max', 'count']))
id_max = df_vendas['id'].max()
print(f'IDs esperados (0..{id_max}): {id_max + 1} | Linhas reais: {len(df_vendas)} | Lacunas: {id_max + 1 - len(df_vendas)}')

# Estatísticas apenas das colunas com significado analítico
print(f'\n=== ESTATÍSTICAS (qtd e total) ===')
display(df_vendas[['qtd', 'total']].describe().round(2))

# Distribuição dos formatos de sale_date
def detectar_formato(d):
    d = str(d)
    if re.match(r'^\d{4}-\d{2}-\d{2}$', d):
        return 'YYYY-MM-DD'
    elif re.match(r'^\d{2}-\d{2}-\d{4}$', d):
        return 'DD-MM-YYYY'
    else:
        return 'outro'

formatos = df_vendas['sale_date'].apply(detectar_formato)
print(f'\n=== FORMATOS DE sale_date ===')
print(formatos.value_counts().to_string())
print(f'\nTotal: {len(df_vendas)} datas — ambos os formatos precisam ser normalizados no tratamento')

# %% [markdown]
# ### 2.2 Produtos

# %%
df_produtos.head(10)

# %%
df_produtos.info()

# %%
print('=== NULOS ===')
print(df_produtos.isnull().sum())
print(f'\n=== DUPLICATAS ===')
print(f'Linhas duplicadas: {df_produtos.duplicated().sum()}')

# actual_category: lista todas as variações para expor o problema real
print(f'\n=== actual_category — {df_produtos["actual_category"].nunique()} variações únicas (para 3 categorias reais) ===')
print(df_produtos['actual_category'].value_counts().to_string())

# %%
# Investigação: 157 linhas mas code máximo = 150 (detectado no describe acima)
# Esperamos 150 codes únicos — vamos confirmar
print(f'Total de linhas    : {len(df_produtos)}')
print(f'Codes únicos       : {df_produtos["code"].nunique()}')
print(f'Diferença          : {len(df_produtos) - df_produtos["code"].nunique()} linhas duplicadas por code')
print()
print('Codes com mais de 1 ocorrência:')
print(df_produtos["code"].value_counts()[df_produtos["code"].value_counts() > 1])

# %% [markdown]
# ### 2.3 Clientes

# %%
df_clientes.head(10)

# %%
df_clientes.info()

# %%
print('=== NULOS ===')
print(df_clientes.isnull().sum())
print(f'\n=== DUPLICATAS ===')
print(f'Linhas duplicadas: {df_clientes.duplicated().sum()}')

# Emails: conta os inválidos (# no lugar de @)
emails_invalidos = df_clientes[~df_clientes['email'].str.contains('@', na=False)]
print(f'\n=== EMAILS ===')
print(f'Emails inválidos (# no lugar de @): {len(emails_invalidos)}/{len(df_clientes)}')

# Location: mostra diversidade de formatos
print(f'\n=== LOCATION — formatos variados (amostra) ===')
print(df_clientes['location'].head(10).to_string())
print('\nPadrões identificados: "UF , Cidade", "Cidade/UF", "UF - Cidade", "Cidade,UF" — sem padronização')

# %% [markdown]
# ### 2.4 Custos de Importação

# %%
df_custos.head(10)

# %%
df_custos.info()

# %%
print('=== NULOS ===')
print(df_custos.isnull().sum())
print(f'\n=== DUPLICATAS ===')
cols_hashable = ['product_id', 'product_name', 'category']
print(f'Linhas duplicadas: {df_custos.duplicated(subset=cols_hashable).sum()}')

# Explora a estrutura aninhada de historic_data
print(f'\n=== HISTORIC_DATA — estrutura interna ===')
def parse_hd(x):
    if isinstance(x, str):
        return ast.literal_eval(x)
    return list(x)  # numpy array ou list ja parseado

primeiro = parse_hd(df_custos['historic_data'].iloc[0])

print(f'Tipo       : {type(primeiro).__name__} de dicts')
print(f'Campos     : {list(primeiro[0].keys())}')
print(f'1º período : {primeiro[0]}')
print(f'Últ período: {primeiro[-1]}')

# Quantos períodos de preço por produto
n_periodos = df_custos['historic_data'].apply(
    lambda x: len(parse_hd(x))
)
print(f'\nPeríodos de preço por produto:')
print(f'  Mínimo : {n_periodos.min()}')
print(f'  Máximo : {n_periodos.max()}')
print(f'  Média  : {n_periodos.mean():.1f}')
print(f'  Total de linhas após explosão (unnest): ~{n_periodos.sum()}')
print(f'\nAção necessária: explodir historic_data em linhas para cruzar com vendas por data de vigência')

# %% [markdown]
# ## 3. Integridade Referencial entre Bases
#
# Verificamos se todos os IDs nas bases transacionais (`vendas`) têm correspondência nas bases de referência (`clientes`, `produtos`, `custos`). Um ID órfão invalidaria qualquer cruzamento posterior.

# %%
# Verifica se todos os IDs em vendas existem nas bases de referência

clientes_em_vendas   = set(df_vendas['id_client'].unique())
clientes_cadastrados = set(df_clientes['code'].unique())
orfaos_cli = clientes_em_vendas - clientes_cadastrados
print(f'id_client em vendas  : {len(clientes_em_vendas)} únicos')
print(f'codes em clientes    : {len(clientes_cadastrados)}')
print(f'Órfãos (sem cadastro): {len(orfaos_cli)} → {"nenhum ✓" if not orfaos_cli else orfaos_cli}')

print()
produtos_em_vendas   = set(df_vendas['id_product'].unique())
produtos_no_catalogo = set(df_produtos['code'].unique())
orfaos_prod = produtos_em_vendas - produtos_no_catalogo
print(f'id_product em vendas : {len(produtos_em_vendas)} únicos')
print(f'codes em produtos    : {len(produtos_no_catalogo)}')
print(f'Órfãos (sem cadastro): {len(orfaos_prod)} → {"nenhum ✓" if not orfaos_prod else orfaos_prod}')

print()
custos_cadastrados = set(df_custos['product_id'].unique())
sem_custo = produtos_em_vendas - custos_cadastrados
print(f'Produtos vendidos sem custo cadastrado: {len(sem_custo)} → {"nenhum ✓" if not sem_custo else sem_custo}')

# %% [markdown]
# ## 4. Resumo dos Problemas Encontrados
#
# ### Vendas (bronze_vendas) — 9.895 linhas, 6 colunas
# | Problema | Evidência | Ação necessária |
# |---|---|---|
# | `sale_date` em formato misto | **4.982** DD-MM-YYYY e **4.913** YYYY-MM-DD — quase 50/50 | Normalizar para `datetime64` com detecção dinâmica de formato |
# | IDs com lacunas | `id` vai de 0 a 9999 mas há 9.895 linhas | Não é erro — indica registros deletados historicamente |
# | Valores extremos em `total` | P75 = R$ 339K, máx = R$ 2.2M | Manter — produtos náuticos de alto valor são plausíveis |
#
# ---
#
# ### Produtos (bronze_produtos) — 157 linhas, 4 colunas
# | Problema | Evidência | Ação necessária |
# |---|---|---|
# | `price` como string | Valores com prefixo `"R$ "` — dtype `object` | Remover prefixo e converter para `float` |
# | `actual_category` inconsistente | **39 variações** para apenas 3 categorias reais | Normalizar para categorias canônicas |
# | 157 linhas com 150 codes únicos | codes 62 (4×), 145 (3×), 37 (2×), 127 (2×) duplicados | Deduplicar mantendo o primeiro registro |
#
# ---
#
# ### Clientes (`clientes_crm.json`) — 49 linhas, 4 colunas
# | Problema | Evidência | Ação necessária |
# |---|---|---|
# | Emails com `#` no lugar de `@` | **30 de 49** registros afetados | Substituir `#` por `@` |
# | `location` sem padrão | 4+ formatos: `"PE , Recife"`, `"PB/Cabedelo"`, `"PA - Santarém Novo"`, `"Rio Grande,RS"` | Extrair `city` e `state` separadamente |
# | Nomes suspeitos | `"Femininos Oliveira Antunes"` — palavra genérica como primeiro nome | Sinalizar com flag, não deletar sem validação |
#
# ---
#
# ### Custos (`custos_importacao.json`) — 150 linhas, 4 colunas
# | Problema | Evidência | Ação necessária |
# |---|---|---|
# | `historic_data` aninhado | Lista de dicts `{start_date, usd_price}` — **3 a 15 períodos** por produto (~1.260 linhas após explosão) | Explodir para formato longo (uma linha por período) |
# | `start_date` como string | Formato `"10/08/2016"` dentro dos dicts | Converter para `datetime64` após explosão |
# | Custos em **USD** | Receitas em BRL — conversão impossível sem câmbio histórico | Integrar taxa BCB/PTAX na Etapa 2 |
#
# ---
#
# ### Câmbio — Dimensão Ausente nas Bases Brutas
# | Observação | Evidência | Ação necessária |
# |---|---|---|
# | Nenhuma base contém taxa de câmbio | Custos em USD, receitas em BRL — sem conversão não há margem real | Integrar câmbio histórico via API BCB/PTAX (Etapa 2) |
# | Impacto crítico | Sem câmbio, análise de custo vs receita é **impossível** | 02_silver gera a tabela `silver_cambio` no Delta Lake como 5ª base |
#
# ---
#
# ### Integridade Referencial entre Bases — ✓ Sem problemas
#
# | Verificação | Resultado |
# |---|---|
# | `id_client` em vendas ↔ `code` em clientes | **0 órfãos** — 49/49 clientes com cadastro ✓ |
# | `id_product` em vendas ↔ `code` em produtos | **0 órfãos** — 150/150 produtos com cadastro ✓ |
# | `id_product` em vendas ↔ `product_id` em custos | **0 produtos** sem custo cadastrado ✓ |

# %% [markdown]
# ## 5. Data Contract Inicial (ponte para o tratamento)
#
# Este contrato define as expectativas minimas de tipo, completude e regra de negocio para cada coluna critica.
# Ele sera usado como referencia de validacao na Etapa 2 (`02_tratamento.ipynb`).

# %%
data_contract = pd.DataFrame([
    {
        'base': 'vendas',
        'coluna': 'sale_date',
        'tipo_esperado': 'datetime64[ns]',
        'regra': 'Data valida e coerente no periodo 2023-2024',
        'nulos_tolerados': 0
    },
    {
        'base': 'vendas',
        'coluna': 'id_client',
        'tipo_esperado': 'int',
        'regra': 'Deve existir em clientes.code',
        'nulos_tolerados': 0
    },
    {
        'base': 'vendas',
        'coluna': 'id_product',
        'tipo_esperado': 'int',
        'regra': 'Deve existir em produtos.code e custos.product_id',
        'nulos_tolerados': 0
    },
    {
        'base': 'vendas',
        'coluna': 'total',
        'tipo_esperado': 'float',
        'regra': 'Valor monetario positivo',
        'nulos_tolerados': 0
    },
    {
        'base': 'produtos',
        'coluna': 'code',
        'tipo_esperado': 'int',
        'regra': 'Chave unica apos deduplicacao',
        'nulos_tolerados': 0
    },
    {
        'base': 'produtos',
        'coluna': 'price',
        'tipo_esperado': 'float',
        'regra': 'Remover prefixo R$ e converter',
        'nulos_tolerados': 0
    },
    {
        'base': 'produtos',
        'coluna': 'actual_category',
        'tipo_esperado': 'string',
        'regra': 'Padronizar para categorias canonicas',
        'nulos_tolerados': 0
    },
    {
        'base': 'clientes',
        'coluna': 'email',
        'tipo_esperado': 'string',
        'regra': 'Deve conter @ apos correcao',
        'nulos_tolerados': 0
    },
    {
        'base': 'clientes',
        'coluna': 'location',
        'tipo_esperado': 'string',
        'regra': 'Extrair city/state em colunas separadas',
        'nulos_tolerados': 0
    },
    {
        'base': 'custos',
        'coluna': 'historic_data.start_date',
        'tipo_esperado': 'datetime64[ns]',
        'regra': 'Explodir lista e converter data',
        'nulos_tolerados': 0
    },
    {
        'base': 'custos',
        'coluna': 'historic_data.usd_price',
        'tipo_esperado': 'float',
        'regra': 'Explodir lista e converter para numerico',
        'nulos_tolerados': 0
    },
])

print('=== DATA CONTRACT INICIAL ===')
display(data_contract)

# %% [markdown]
# ## 6. Matriz de Decisoes de Limpeza (EDA -> Tratamento)
#
# Cada decisao abaixo inclui risco e validacao esperada na Etapa 2 para evitar ajustes sem rastreabilidade.

# %%
duplicated_codes = df_produtos[df_produtos.duplicated(subset=['code'], keep=False)]['code'].unique()
impacto_receita_duplicados = (
    df_vendas[df_vendas['id_product'].isin(duplicated_codes)]['total'].sum()
    / df_vendas['total'].sum() * 100
)

formatos_local = df_vendas['sale_date'].astype(str).str.strip()
pct_datas_mistas = (
    formatos_local.str.match(r'^\d{2}-\d{2}-\d{4}$').sum() / len(formatos_local) * 100
)

matriz_decisoes = pd.DataFrame([
    {
        'problema': 'sale_date em formato misto',
        'decisao_eda': 'Normalizar para datetime com parser de multiplos formatos',
        'risco': 'Baixo (com teste de NaT e range)',
        'validacao_etapa_2': 'sale_date sem NaT e no periodo esperado'
    },
    {
        'problema': 'actual_category com variacoes',
        'decisao_eda': 'Padronizar para categorias canonicas',
        'risco': 'Medio (mapeamento pode forcar classe errada)',
        'validacao_etapa_2': 'Contagem final por categoria e lista de nao mapeados'
    },
    {
        'problema': 'Produtos duplicados por code',
        'decisao_eda': 'Deduplicar por code mantendo primeiro registro',
        'risco': f"Medio (produtos afetados por {impacto_receita_duplicados:.1f}% da receita)",
        'validacao_etapa_2': '150 codes unicos e log de removidos'
    },
    {
        'problema': 'Emails com # no lugar de @',
        'decisao_eda': 'Substituir # por @ quando necessario',
        'risco': 'Baixo (regra deterministica)',
        'validacao_etapa_2': '0 emails invalidos apos correcao'
    },
    {
        'problema': 'historic_data aninhado em custos',
        'decisao_eda': 'Explodir para formato longo',
        'risco': 'Baixo (transformacao estrutural controlada)',
        'validacao_etapa_2': 'shape esperado e tipos convertidos'
    },
    {
        'problema': 'Custo em USD sem cambio historico',
        'decisao_eda': 'Integrar PTAX/BCB diario com preenchimento de feriados',
        'risco': 'Medio (drift se artefato for regenerado sem controle)',
        'validacao_etapa_2': 'cambio_clean sem nulos e periodo completo 2023-2024'
    },
])

print('=== MATRIZ DE DECISOES (EDA -> ETAPA 2) ===')
display(matriz_decisoes)
print(f'\nIndicador de risco adicional: {pct_datas_mistas:.1f}% das datas estavam em DD-MM-YYYY.')

# %% [markdown]
# ## 7. Checklist Go/No-Go para Etapa 2
#
# A Etapa 2 so deve iniciar quando os checks criticos estiverem aprovados.

# %%
vendas_nulos_criticos = int(df_vendas[['id_client', 'id_product', 'sale_date', 'total']].isna().sum().sum())
produtos_nulos_criticos = int(df_produtos[['code', 'price']].isna().sum().sum())
clientes_nulos_criticos = int(df_clientes[['code', 'email']].isna().sum().sum())
custos_nulos_criticos = int(df_custos[['product_id', 'historic_data']].isna().sum().sum())

emails_invalidos_qtd = int((~df_clientes['email'].str.contains('@', na=False)).sum())
tem_orfaos = len(orfaos_cli) > 0 or len(orfaos_prod) > 0 or len(sem_custo) > 0

checklist = pd.DataFrame([
    {'check': 'Tabelas Delta Lake carregadas', 'tipo': 'go_no_go', 'status': len(df_vendas) > 0},
    {'check': 'Sem orfaos referenciais criticos', 'tipo': 'go_no_go', 'status': not tem_orfaos},
    {'check': 'Nulos criticos em vendas = 0', 'tipo': 'go_no_go', 'status': vendas_nulos_criticos == 0},
    {'check': 'Nulos criticos em produtos = 0', 'tipo': 'go_no_go', 'status': produtos_nulos_criticos == 0},
    {'check': 'Nulos criticos em clientes = 0', 'tipo': 'go_no_go', 'status': clientes_nulos_criticos == 0},
    {'check': 'Nulos criticos em custos = 0', 'tipo': 'go_no_go', 'status': custos_nulos_criticos == 0},
    {'check': 'Custo historico em estrutura aninhada mapeado', 'tipo': 'go_no_go', 'status': 'historic_data' in df_custos.columns},

    # Diagnostico do estado bruto (esperado encontrar problema para tratar na Etapa 2)
    {'check': 'Diagnostico: existem emails invalidos no bruto', 'tipo': 'diagnostico', 'status': emails_invalidos_qtd > 0},
    {'check': 'Diagnostico: sale_date esta em formato misto', 'tipo': 'diagnostico', 'status': formatos.nunique() > 1},
])

checklist['resultado'] = checklist['status'].map({True: 'PASS', False: 'FAIL'})
display(checklist[['tipo', 'check', 'resultado']])

check_go_no_go = checklist[checklist['tipo'] == 'go_no_go']['status'].all()
if check_go_no_go:
    print('\nGO: EDA pronta para avancar para a Etapa 2.')
else:
    print('\nNO-GO: revise os itens críticos com FAIL antes de seguir.')

print(f"\nDiagnóstico bruto: emails inválidos identificados = {emails_invalidos_qtd} (esperado > 0 nesta etapa).")


# %% [markdown]
# ## 8. Resumo Executivo (Etapa 1)
#
# - A EDA confirmou problemas de qualidade reais em vendas, produtos, clientes e custos.
# - Não há quebra de integridade referencial entre vendas, clientes, produtos e custos.
# - O maior risco de análise de margem no bruto é a ausência de câmbio histórico.
# - As decisões de limpeza foram documentadas com risco e validação esperada na Etapa 2.
# - O Go/No-Go agora separa critérios críticos de execução e diagnóstico do estado bruto.

# %%
def run_all_quality_checks_eda(df_vendas_local, df_produtos_local, df_clientes_local, df_custos_local):
    """Executa checks críticos da Etapa 1 para facilitar reexecução controlada."""
    checks = {
        'vendas_colunas_criticas_ok': set(['id_client', 'id_product', 'sale_date', 'total']).issubset(df_vendas_local.columns),
        'produtos_colunas_criticas_ok': set(['code', 'price', 'actual_category']).issubset(df_produtos_local.columns),
        'clientes_colunas_criticas_ok': set(['code', 'email', 'location']).issubset(df_clientes_local.columns),
        'custos_colunas_criticas_ok': set(['product_id', 'historic_data']).issubset(df_custos_local.columns),
        'integridade_clientes_sem_orfaos': len(set(df_vendas_local['id_client']) - set(df_clientes_local['code'])) == 0,
        'integridade_produtos_sem_orfaos': len(set(df_vendas_local['id_product']) - set(df_produtos_local['code'])) == 0,
    }
    out = pd.DataFrame({'check': checks.keys(), 'status': checks.values()})
    out['resultado'] = out['status'].map({True: 'PASS', False: 'FAIL'})
    return out

qa_eda = run_all_quality_checks_eda(df_vendas, df_produtos, df_clientes, df_custos)
display(qa_eda[['check', 'resultado']])

if qa_eda['status'].all():
    print('RUN_ALL CHECKS (EDA): PASS')
else:
    print('RUN_ALL CHECKS (EDA): FAIL')
