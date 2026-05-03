from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="LH Nautical - Dashboard Executivo",
    page_icon="📈",
    layout="wide",
)

# Fontes de dados — local-first com fallback pra raw URL do GitHub.
# Em dev e no Streamlit Cloud (repo clonado), usa o arquivo local em
# data/silver/ ou data/gold/. Cai pro raw URL quando o repo não está
# disponível (ex: app rodando isolado fora do checkout).
GITHUB_RAW = "https://raw.githubusercontent.com/ASCCJR/Indicium_LH_Nautical/main/data"
LOCAL_DATA = Path(__file__).parent.parent / "data"

_DATA_PATHS = {
    "vendas_gold": "gold/vendas_gold.csv",
    "produtos_clean": "silver/produtos_clean.csv",
    "clientes_clean": "silver/clientes_clean.csv",
}


def _resolve(rel_path: str) -> str:
    local = LOCAL_DATA / rel_path
    if local.exists():
        return str(local)
    return f"{GITHUB_RAW}/{rel_path}"


REQUIRED_FILES = {key: _resolve(rel) for key, rel in _DATA_PATHS.items()}


def _missing_local_sources() -> list[str]:
    """Lista CSVs que estão sendo lidos via fallback URL (local não encontrado)."""
    return [key for key, rel in _DATA_PATHS.items() if not (LOCAL_DATA / rel).exists()]


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    vendas = pd.read_csv(REQUIRED_FILES["vendas_gold"], parse_dates=["sale_date"])
    produtos = pd.read_csv(REQUIRED_FILES["produtos_clean"])
    clientes = pd.read_csv(REQUIRED_FILES["clientes_clean"])

    produtos = produtos.rename(columns={"code": "id_product", "name": "produto_nome"})
    clientes = clientes.rename(columns={"code": "id_client", "full_name": "cliente_nome"})

    vendas = vendas.merge(
        produtos[["id_product", "produto_nome", "actual_category"]],
        on="id_product",
        how="left",
    )
    vendas = vendas.merge(
        clientes[["id_client", "cliente_nome", "state", "city"]],
        on="id_client",
        how="left",
    )

    return vendas, produtos, clientes


def format_money(value: float) -> str:
    return f"R$ {value:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")


def format_pct(value: float) -> str:
    return f"{value:.1f}%"


def kpi_row(vendas_f: pd.DataFrame) -> None:
    receita = float(vendas_f["total"].sum())
    custo = float(vendas_f["custo_brl"].sum())
    lucro = float(vendas_f["lucro"].sum())
    margem = (lucro / receita * 100.0) if receita else 0.0
    n_clientes = int(vendas_f["id_client"].nunique())
    n_produtos = int(vendas_f["id_product"].nunique())

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Receita Total", format_money(receita))
    c2.metric("Custo Total", format_money(custo))
    c3.metric("Lucro Total", format_money(lucro))
    delta_label = "⚠️ Negativa" if margem < 0 else None
    c4.metric("Margem", format_pct(margem), delta=delta_label, delta_color="inverse")
    c5.metric("Clientes Ativos", str(n_clientes))
    c6.metric("Produtos Vendidos", str(n_produtos))


# ── App ──────────────────────────────────────────────────────────────────────

st.title("LH Nautical — Dashboard Executivo")
st.caption("Baseado nas camadas clean/gold. Margem = custo USD convertido por câmbio diário PTAX/BCB.")

_via_url = _missing_local_sources()
if _via_url:
    st.caption(
        f"ℹ️ Lendo {len(_via_url)} CSV(s) via GitHub raw URL — "
        "arquivos locais ausentes em `data/silver/` ou `data/gold/`."
    )

vendas, produtos, clientes = load_data()

if "taxa_brl" in vendas.columns:
    st.caption(
        f"Cobertura cambial: {vendas['taxa_brl'].dropna().shape[0]:,} transações com taxa PTAX/BCB."
    )

# ── Filtros globais ───────────────────────────────────────────────────────────
st.sidebar.header("Filtros")
anos = sorted(vendas["ano"].dropna().unique().tolist())
anos_sel = st.sidebar.multiselect("Ano", options=anos, default=anos)

estados = sorted([x for x in vendas["state"].dropna().unique().tolist() if str(x).strip()])
estados_sel = st.sidebar.multiselect("Estado", options=estados, default=estados)

categorias = sorted([x for x in vendas["actual_category"].dropna().unique().tolist() if str(x).strip()])
cat_sel = st.sidebar.multiselect("Categoria", options=categorias, default=categorias)

min_dt = vendas["sale_date"].min().date()
max_dt = vendas["sale_date"].max().date()
dt_ini, dt_fim = st.sidebar.date_input(
    "Intervalo de datas",
    value=(min_dt, max_dt),
    min_value=min_dt,
    max_value=max_dt,
)

vendas_f = vendas.copy()
if anos_sel:
    vendas_f = vendas_f[vendas_f["ano"].isin(anos_sel)]
if estados_sel:
    vendas_f = vendas_f[vendas_f["state"].isin(estados_sel)]
if cat_sel:
    vendas_f = vendas_f[vendas_f["actual_category"].isin(cat_sel)]
vendas_f = vendas_f[
    (vendas_f["sale_date"].dt.date >= dt_ini) &
    (vendas_f["sale_date"].dt.date <= dt_fim)
]

if vendas_f.empty:
    st.warning("Nenhum dado para os filtros selecionados. Ajuste os filtros na barra lateral.")
    st.stop()

kpi_row(vendas_f)
st.markdown("---")

tabs = st.tabs(["1) Resumo", "2) Produtos", "3) Clientes", "4) Previsão", "5) Recomendação"])

# ── Tab 1: Resumo ─────────────────────────────────────────────────────────────
with tabs[0]:
    st.subheader("Resumo de Desempenho")

    resumo_mensal = (
        vendas_f.groupby("ano_mes", as_index=False)
        .agg(
            receita=("total", "sum"),
            lucro=("lucro", "sum"),
            taxa_media=("taxa_brl", "mean"),
        )
        .sort_values("ano_mes")
    )
    resumo_mensal["margem_pct"] = (
        resumo_mensal["lucro"] / resumo_mensal["receita"] * 100
    ).round(1)
    resumo_mensal["receita_m"] = (resumo_mensal["receita"] / 1e6).round(2)

    fig = go.Figure()

    # Bars: receita mensal
    bar_colors = [
        "#e74c3c" if m < -5 else ("#f39c12" if m < 0 else "#3498db")
        for m in resumo_mensal["margem_pct"]
    ]
    fig.add_trace(go.Bar(
        x=resumo_mensal["ano_mes"],
        y=resumo_mensal["receita_m"],
        name="Receita (R$M)",
        marker_color=bar_colors,
        opacity=0.75,
        yaxis="y1",
    ))

    # Line: margem%
    fig.add_trace(go.Scatter(
        x=resumo_mensal["ano_mes"],
        y=resumo_mensal["margem_pct"],
        name="Margem (%)",
        mode="lines+markers",
        line=dict(color="#c0392b", width=2.5),
        yaxis="y2",
    ))

    # Line: câmbio (if available)
    if resumo_mensal["taxa_media"].notna().any():
        fig.add_trace(go.Scatter(
            x=resumo_mensal["ano_mes"],
            y=resumo_mensal["taxa_media"].round(3),
            name="Câmbio R$/USD",
            mode="lines+markers",
            line=dict(color="#f39c12", width=1.8, dash="dot"),
            marker=dict(size=5),
            yaxis="y3",
        ))

    fig.update_layout(
        title="Evolução Mensal: Receita, Margem% e Câmbio — o câmbio é o culpado",
        xaxis=dict(title="Mês"),
        yaxis=dict(title="Receita (R$M)", side="left"),
        yaxis2=dict(
            title="Margem (%)",
            overlaying="y",
            side="right",
            zeroline=True,
            zerolinecolor="#c0392b",
            zerolinewidth=1.5,
        ),
        yaxis3=dict(
            title="R$/USD",
            overlaying="y",
            side="right",
            anchor="free",
            position=0.98,
            showgrid=False,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified",
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

    n_neg = int((resumo_mensal["margem_pct"] < -5).sum())
    n_risco = int(((resumo_mensal["margem_pct"] >= -5) & (resumo_mensal["margem_pct"] < 0)).sum())
    if n_neg > 0 or n_risco > 0:
        st.error(
            f"⚠️ {n_neg} meses com margem negativa (<−5%) e {n_risco} em zona de risco (−5% a 0%). "
            "A depreciação do BRL frente ao USD corrói a margem bruta."
        )

# ── Tab 2: Produtos ──────────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("Produtos: lucratividade real")

    prod_lucro = (
        vendas_f.groupby(["id_product", "produto_nome", "actual_category"], as_index=False)
        .agg(receita=("total", "sum"), lucro=("lucro", "sum"))
        .assign(margem=lambda d: (d["lucro"] / d["receita"] * 100).round(1))
    )
    prod_piores = prod_lucro.nsmallest(15, "lucro")

    fig_prod = px.bar(
        prod_piores,
        x="lucro",
        y="produto_nome",
        color="actual_category",
        orientation="h",
        title="Top 15 Produtos com Maior Prejuízo Acumulado",
    )
    st.plotly_chart(fig_prod, use_container_width=True)
    st.dataframe(
        prod_piores[["produto_nome", "actual_category", "receita", "lucro", "margem"]]
        .rename(columns={"produto_nome": "Produto", "actual_category": "Categoria",
                         "receita": "Receita (R$)", "lucro": "Lucro (R$)", "margem": "Margem Média (%)"})
        .reset_index(drop=True),
        use_container_width=True,
    )

# ── Tab 3: Clientes ──────────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("Clientes: ranking de lucratividade real")

    cli_lucro = (
        vendas_f.groupby(["id_client", "cliente_nome", "state"], as_index=False)
        .agg(receita=("total", "sum"), lucro=("lucro", "sum"))
        .assign(margem=lambda d: (d["lucro"] / d["receita"] * 100).round(1))
        .sort_values("lucro", ascending=False)
    )
    fig_cli = px.bar(
        cli_lucro.head(15),
        x="lucro",
        y="cliente_nome",
        color="state",
        orientation="h",
        title="Top 15 Clientes por Lucro Acumulado (ou menor prejuízo)",
    )
    st.plotly_chart(fig_cli, use_container_width=True)
    st.dataframe(cli_lucro.head(20), use_container_width=True)

# ── Tab 4: Previsão ───────────────────────────────────────────────────────────
with tabs[3]:
    st.subheader("Previsão: Baseline Sazonal")

    serie = (
        vendas_f.assign(mes=vendas_f["sale_date"].dt.to_period("M").dt.to_timestamp())
        .groupby("mes", as_index=False)
        .agg(receita=("total", "sum"))
        .sort_values("mes")
    )

    if len(serie) < 12:
        st.info("Histórico insuficiente para o baseline sazonal (mínimo 12 meses). Amplie o filtro de datas.")
    else:
        n_test = min(12, len(serie) - 1)
        treino = serie.iloc[:-n_test]
        teste = serie.iloc[-n_test:]

        media_treino = float(treino["receita"].mean())
        fator_mensal = (
            treino.assign(mes_num=treino["mes"].dt.month)
            .groupby("mes_num")["receita"]
            .mean()
            / media_treino
        )

        serie["pred"] = serie["mes"].apply(
            lambda d: media_treino * fator_mensal.get(d.month, 1.0)
        )
        pred_teste = teste["mes"].apply(
            lambda d: media_treino * fator_mensal.get(d.month, 1.0)
        )
        mape = float(
            ((teste["receita"].values - pred_teste.values) / teste["receita"].values).mean() * 100
        )
        mape = abs(mape)

        ultimo_mes = serie["mes"].max()
        meses_fut = pd.date_range(
            ultimo_mes + pd.offsets.MonthBegin(1), periods=6, freq="MS"
        )
        forecast = pd.DataFrame({
            "mes": meses_fut,
            "receita": [
                round(media_treino * fator_mensal.get(m.month, 1.0) / 1e6, 2)
                for m in meses_fut
            ],
            "tipo": ["Forecast"] * 6,
        })

        hist = serie[["mes", "receita"]].copy()
        hist["receita"] = (hist["receita"] / 1e6).round(2)
        hist["tipo"] = "Histórico"
        prev_df = pd.concat([hist, forecast], ignore_index=True)

        st.caption(
            f"Método: Média histórica × fator sazonal mensal  |  MAPE no histórico: **{mape:.1f}%**"
        )

        fig_prev = px.line(
            prev_df,
            x="mes",
            y="receita",
            color="tipo",
            markers=True,
            color_discrete_map={"Histórico": "#3498db", "Forecast": "#e67e22"},
            title="Histórico de Receita + Forecast Jan–Jun/2025 (Baseline Sazonal)",
            labels={"receita": "Receita (R$M)", "mes": "Mês"},
        )
        fig_prev.update_traces(
            selector=dict(name="Forecast"),
            line=dict(dash="dash"),
        )
        # Annotate forecast values
        for _, row in forecast.iterrows():
            fig_prev.add_annotation(
                x=row["mes"], y=row["receita"],
                text=f"R${row['receita']}M",
                showarrow=False, yshift=14, font=dict(size=9, color="#e67e22"),
            )
        fig_prev.add_vline(
            x=pd.Timestamp("2025-01-01").timestamp() * 1000,
            line_dash="dot", line_color="gray", opacity=0.5,
        )
        st.plotly_chart(fig_prev, use_container_width=True)
        st.warning(
            "⚠️ Com margem atual de −5,3%, cada mês de receita previsto implica "
            "prejuízo proporcional. Prioridade: reajuste de preços antes de escalar vendas."
        )

# ── Tab 5: Recomendação ───────────────────────────────────────────────────────
with tabs[4]:
    st.subheader("Recomendação: Content-Based ajustado por Margem")

    # Margem real por produto no período filtrado
    margem_prod = (
        vendas_f.groupby(["id_product", "produto_nome", "actual_category"], as_index=False)
        .agg(receita=("total", "sum"), lucro=("lucro", "sum"))
        .assign(margem_pct=lambda d: (d["lucro"] / d["receita"] * 100).round(1))
        .sort_values("margem_pct", ascending=False)
    )
    margem_prod["status"] = margem_prod["margem_pct"].apply(
        lambda x: "positiva" if x >= 0 else ("risco" if x >= -5 else "negativa")
    )

    # Perfil geral + gráfico por categoria
    col_kpi, col_cat = st.columns([1, 2])

    with col_kpi:
        st.markdown("**Perfil de margem do catálogo (filtro atual)**")
        contagem = margem_prod["status"].value_counts()
        n_total = len(margem_prod)
        st.metric("Margem positiva (≥0%)", f"{contagem.get('positiva', 0)}/{n_total}")
        st.metric("Zona de risco (−5% a 0%)", f"{contagem.get('risco', 0)}/{n_total}")
        st.metric("Margem negativa (<−5%)", f"{contagem.get('negativa', 0)}/{n_total}")

    with col_cat:
        cat_margem = (
            margem_prod.groupby("actual_category", as_index=False)["margem_pct"]
            .mean()
            .sort_values("margem_pct")
        )
        fig_cat = px.bar(
            cat_margem,
            x="margem_pct",
            y="actual_category",
            orientation="h",
            color="margem_pct",
            color_continuous_scale=["#e74c3c", "#f39c12", "#27ae60"],
            title="Margem Média por Categoria",
            labels={"margem_pct": "Margem (%)", "actual_category": "Categoria"},
        )
        fig_cat.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_cat, use_container_width=True)

    st.markdown("---")
    st.markdown("**Recomendações por cliente** — produtos não comprados com melhor margem na categoria favorita")

    clientes_lista = (
        vendas_f.groupby(["id_client", "cliente_nome"], as_index=False)["total"]
        .sum()
        .sort_values("total", ascending=False)
    )
    opcoes = {row["cliente_nome"]: row["id_client"] for _, row in clientes_lista.iterrows()}
    cliente_sel = st.selectbox("Selecione o cliente", options=list(opcoes.keys()))

    if cliente_sel:
        cid = opcoes[cliente_sel]
        historico_cli = vendas_f[vendas_f["id_client"] == cid]
        ja_comprou = set(historico_cli["id_product"])
        cat_favorita = historico_cli.groupby("actual_category")["total"].sum().idxmax()

        recomendacoes = (
            margem_prod[
                (margem_prod["actual_category"] == cat_favorita) &
                (~margem_prod["id_product"].isin(ja_comprou))
            ]
            .sort_values("margem_pct", ascending=False)
            .head(5)
            .reset_index(drop=True)
        )

        receita_cli = float(historico_cli["total"].sum())
        lucro_cli = float(historico_cli["lucro"].sum())
        margem_cli = (lucro_cli / receita_cli * 100) if receita_cli else 0.0

        st.markdown(
            f"**{cliente_sel}** — Receita acumulada: {format_money(receita_cli)} | "
            f"Margem: {format_pct(margem_cli)} | Categoria favorita: **{cat_favorita}**"
        )

        if recomendacoes.empty:
            st.info(f"Este cliente já comprou todos os produtos disponíveis na categoria {cat_favorita}.")
        else:
            st.dataframe(
                recomendacoes[["produto_nome", "actual_category", "margem_pct", "status"]]
                .rename(columns={
                    "produto_nome": "Produto",
                    "actual_category": "Categoria",
                    "margem_pct": "Margem (%)",
                    "status": "Status",
                }),
                use_container_width=True,
            )
            st.caption(
                "Lógica: produtos da categoria favorita do cliente que ele ainda não comprou, "
                "ordenados por maior margem — prioriza produtos com margem positiva antes do reajuste geral de preços."
            )

st.markdown("---")
