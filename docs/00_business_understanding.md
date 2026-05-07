# Business Understanding — LH Nautical

> Documento conceitual da etapa 1 do CRISP-DM. Texto puro, sem código.
> Para detalhamento técnico, ver [README.md](../README.md) e [RELATORIO_EXECUTIVO.md](../RELATORIO_EXECUTIVO.md).

---

## 1. Contexto do negócio

A **LH Nautical** é uma empresa de varejo do setor náutico com loja física em Florianópolis e operação de e-commerce de alcance nacional. O catálogo é composto majoritariamente por produtos importados (motores de popa, eletrônicos de bordo, acessórios), com preço de compra denominado em **dólar americano (USD)** e preço de venda em **real brasileiro (BRL)**.

Cenário operacional descrito no desafio:

- Controle de estoque feito em planilhas manuais.
- Banco de dados do e-commerce desconectado do sistema financeiro.
- Decisões comerciais tomadas sem dados consolidados (sem visão unificada de margem, recorrência de cliente, sazonalidade).

## 2. Problema de negócio

A operação reporta crescimento de receita, mas o resultado financeiro real é desconhecido porque **o custo de importação não é convertido pelo câmbio do dia da venda**. Em um regime cambial volátil (USD/BRL oscilou entre R$4,72 e R$6,20 no período 2023–2024), tratar custo USD com taxa fixa ou tardia distorce a margem de forma material.

### Pergunta-chave

> Qual é a margem real da operação quando o custo de cada venda é convertido pela taxa PTAX/BCB do próprio dia?

### Pergunta secundária

> Quais produtos, clientes e categorias destroem margem? Quais devem entrar nas próximas campanhas? Como prever demanda nos próximos 6 meses?

## 3. Achado central

Aplicando a metodologia de custo real (preço de compra em USD × câmbio PTAX do dia da venda), a operação acumula prejuízo de **R$-139 milhões em 2 anos** sobre R$2,61 bilhões de receita — margem real de **-5,3%**. O número é invisível sem a integração com a API do Banco Central como quinta fonte de dados.

A receita cresceu +2,5% de 2023 para 2024, mas o câmbio subiu +8% e o custo em BRL cresceu +10,5%. **A operação está vendendo abaixo do custo de importação.**

## 4. Stakeholders e objetivos

| Stakeholder | Objetivo no projeto |
|---|---|
| Direção comercial | Decidir reajuste de preços e priorização de campanhas |
| Equipe de vendas | Saber quais produtos e clientes priorizar/evitar |
| Financeiro | Entender margem real e o impacto cambial |
| Operações | Planejar estoque com forecast confiável |
| Indicium (avaliador) | Avaliar entregas técnicas (pipeline, análises, dashboard, recomendação) |

## 5. Decisões de arquitetura

### 5.1. Arquitetura medallion (Bronze → Silver → Gold)

| Camada | Responsabilidade | Localização |
|---|---|---|
| **Bronze** | Dados como vieram da fonte, sem transformação | `data/bronze/` (local) e tabelas `bronze_*` no Unity Catalog |
| **Silver** | Dados limpos, padronizados, com tipos corretos | `data/silver/` e `silver_*` |
| **Gold** | Dados prontos para análise/modelagem, com enriquecimento (margem real, etc.) | `data/gold/` e `gold_*` |

### 5.2. Quinta fonte: API do Banco Central (BCB/PTAX)

As 4 fontes do desafio (vendas, produtos, clientes, custos de importação) **não permitem calcular margem real sozinhas**. A integração com `olinda.bcb.gov.br/olinda/servico/PTAX/versao/v1` foi tratada como uma **decisão analítica essencial**, não como bonus opcional. Sem ela, a margem reportada seria artificial.

### 5.3. As-of join para custo USD vigente

Cada venda usa o **último custo USD do produto vigente até a data da venda** (não o custo médio nem o custo atual). Isso preserva a realidade econômica do momento da transação.

### 5.4. Reconciliação financeira

O notebook gold valida que `receita - custo - lucro` reconcilia em todas as transações com tolerância < R$0,01 (proteção contra erros de arredondamento ou conversão).

## 6. Premissa crítica

> **Sem conversão cambial histórica por data de venda, o resultado parece artificialmente melhor.**

Esta é a premissa fundadora do projeto. Toda análise downstream (margem por produto/cliente/categoria, recomendação ajustada por margem, break-even cambial) depende dela. Comprometer essa premissa invalida as conclusões.

## 7. Glossário de variáveis (camada Gold)

### `gold_fct_vendas` (uma linha por transação)

| Campo | Tipo | Descrição |
|---|---|---|
| `id` | int | Chave primária da transação |
| `id_client` | int | Chave do cliente (FK para `silver_clientes`) |
| `id_product` | int | Chave do produto (FK para `silver_produtos`) |
| `qtd` | int | Quantidade vendida na transação |
| `total` | float | Receita da transação em BRL (preço × qtd) |
| `sale_date` | date | Data da venda — usada para casar com câmbio PTAX |
| `usd_price_vigente` | float | Custo unitário USD vigente na `sale_date` (as-of join) |
| `taxa_brl` | float | Taxa USD/BRL PTAX/BCB do dia da venda |
| `custo_brl` | float | Custo total em BRL = `qtd × usd_price_vigente × taxa_brl` |
| `lucro` | float | `total - custo_brl` (real, com câmbio do dia) |
| `margem_pct` | float | `lucro / total × 100` |
| `ano`, `mes`, `ano_mes` | int / str | Derivados de `sale_date` para agregações |

### `silver_produtos`

| Campo | Descrição |
|---|---|
| `code` | Código do produto (PK) |
| `name` | Nome do produto |
| `price` | Preço de tabela em BRL |
| `actual_category` | Categoria normalizada (3 canônicas após dedup de 39 variações) |

### `silver_clientes`

| Campo | Descrição |
|---|---|
| `code` | Código do cliente (PK) |
| `full_name` | Nome completo |
| `email` | Email corrigido (originalmente 30/49 com `#` no lugar de `@`) |
| `state`, `city` | Extraídos do campo `location` original |
| `nome_suspeito` | Flag de qualidade — nome com padrão atípico |

### `silver_cambio` (BCB/PTAX)

| Campo | Descrição |
|---|---|
| `data` | Data (1 linha por dia útil) |
| `taxa_brl` | Taxa USD/BRL PTAX de fechamento |

## 8. Limites do estudo

- **Recorte temporal fechado:** Jan/2023 a Dez/2024 (24 meses). Modelos de previsão têm precisão limitada com 2 ciclos sazonais.
- **Dependência cambial:** variação de R$0,50 no câmbio muda o resultado anual em ~R$132M. Análises são sensíveis à premissa cambial.
- **Bases sem chave de integração explícita:** o pipeline assume estabilidade de formatos das 4 bases brutas.
- **Densidade alta do catálogo de compras (73,6%):** sistemas de filtragem colaborativa têm Precision@5 inferior ao Random Baseline — o valor da recomendação está no perfil de margem, não no acerto preditivo puro.

## 9. Referências cruzadas

| O quê | Onde |
|---|---|
| Resumo executivo e visualizações | [`RELATORIO_EXECUTIVO.md`](../RELATORIO_EXECUTIVO.md) |
| Visão geral do projeto e dashboard | [`README.md`](../README.md) |
| Notebooks numerados | [`notebooks/`](../notebooks/) |
| Dashboard interativo | [`dashboard/streamlit_app.py`](../dashboard/streamlit_app.py) e [streamlit.app público](https://lh-nautical-dashboard.streamlit.app) |
