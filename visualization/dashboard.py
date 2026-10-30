# visualization/dashboard.py
"""
Dashboard interativo do BESS Arbitrage Analyzer.

Interface Streamlit multi-página para exploração visual de:
1. Visão geral (KPIs e ranking de submercados)
2. Curtailment detalhado (timeline, padrões horários, heatmaps)
3. Spread de preços (PLD, janelas ótimas, sazonalidade)
4. Simulação de operação do BESS (backtest hora a hora)
5. Oportunidades de investimento (comparativo, sensibilidade)

Execução:
    streamlit run visualization/dashboard.py
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Garante que o diretório raiz do projeto está no path
_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from config.settings import Settings
from models.data_models import TipoFonte

from visualization.data_loader import (
    load_generation_data, load_coff_data, load_pld_data, load_aneel_profile,
    convert_registros_to_df, convert_pld_to_df, filter_by_submercado,
    get_curtailment_analyzer, get_spread_analyzer, get_opportunity_finder,
)

from visualization.charts import (
    grafico_ranking_submercados, cards_kpi_resumo,
    grafico_curtailment_timeline, grafico_curtailment_por_hora,
    grafico_heatmap_curtailment,
    grafico_pld_timeline, grafico_perfil_horario_pld,
    grafico_spread_diario_boxplot, grafico_sazonalidade_spread,
    grafico_simulacao_operacao, grafico_fluxo_caixa_acumulado,
    grafico_comparativo_oportunidades, grafico_sensibilidade_heatmap,
    formatar_moeda, formatar_numero_grande,
)


# ======================================================================
# CONFIGURAÇÃO DA PÁGINA
# ======================================================================
st.set_page_config(
    page_title="BESS Arbitrage Analyzer",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ======================================================================
# SIDEBAR: PARÂMETROS GLOBAIS
# ======================================================================
def render_sidebar() -> dict:
    """Renderiza sidebar com todos os parâmetros de configuração da análise"""
    st.sidebar.title("🔋 BESS Arbitrage Analyzer")
    st.sidebar.markdown(
        "Mapeamento de restrições de geração e oportunidades de "
        "arbitragem via armazenamento de energia."
    )
    st.sidebar.divider()

    st.sidebar.subheader("📅 Período de Análise")
    dias = st.sidebar.slider(
        "Janela retroativa (dias)", min_value=7, max_value=365,
        value=Settings.JANELA_ANALISE_DIAS, step=7
    )
    data_fim = datetime.now()
    data_inicio = data_fim - timedelta(days=dias)
    st.sidebar.caption(f"📍 {data_inicio.strftime('%d/%m/%Y')} → {data_fim.strftime('%d/%m/%Y')}")

    st.sidebar.divider()
    st.sidebar.subheader("🌍 Filtros")

    submercado = st.sidebar.selectbox(
        "Submercado", options=["TODOS"] + Settings.SUBMERCADOS, index=0
    )

    fontes = st.sidebar.multiselect(
        "Fontes de Geração",
        options=['EOLICA', 'SOLAR', 'HIDRAULICA', 'TERMICA'],
        default=['EOLICA', 'SOLAR'],
    )

    st.sidebar.divider()
    st.sidebar.subheader("⚙️ Premissas Técnico-Econômicas")

    eficiencia = st.sidebar.slider(
        "Eficiência Round-Trip do BESS", min_value=0.70, max_value=0.98,
        value=Settings.EFICIENCIA_BESS, step=0.01, format="%.2f"
    )

    capex_mwh = st.sidebar.number_input(
        "CAPEX por MWh (R$)", min_value=500_000, max_value=5_000_000,
        value=1_800_000, step=50_000, format="%d"
    )

    capex_mw = st.sidebar.number_input(
        "CAPEX por MW (R$)", min_value=100_000, max_value=1_500_000,
        value=450_000, step=25_000, format="%d"
    )

    st.sidebar.divider()
    incluir_aneel = st.sidebar.checkbox(
        "Incluir dados ANEEL (saturação de mercado)", value=True
    )

    st.sidebar.divider()
    if st.sidebar.button("🔄 Forçar atualização de dados (limpar cache)"):
        st.cache_data.clear()
        st.sidebar.success("Cache limpo! Dados serão recarregados.")

    return {
        'data_inicio': data_inicio,
        'data_fim': data_fim,
        'submercado': submercado,
        'fontes': tuple(fontes) if fontes else ('EOLICA', 'SOLAR'),
        'eficiencia': eficiencia,
        'capex_mwh': float(capex_mwh),
        'capex_mw': float(capex_mw),
        'incluir_aneel': incluir_aneel,
    }


# ======================================================================
# CARREGAMENTO CENTRAL DE DADOS (compartilhado entre abas)
# ======================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def _carregar_dados_completos(params: dict):
    """Wrapper único de carregamento, evita duplicar chamadas entre abas"""
    registros_geracao = load_generation_data(
        params['fontes'], params['data_inicio'], params['data_fim']
    )
    registros_coff_eolica = (
        load_coff_data('EOLICA', params['data_inicio'], params['data_fim'])
        if 'EOLICA' in params['fontes'] else []
    )
    registros_coff_solar = (
        load_coff_data('SOLAR', params['data_inicio'], params['data_fim'])
        if 'SOLAR' in params['fontes'] else []
    )
    registros_pld = load_pld_data(params['data_inicio'], params['data_fim'])

    df_perfil_aneel = pd.DataFrame()
    if params['incluir_aneel']:
        df_perfil_aneel = load_aneel_profile()

    return (
        registros_geracao,
        registros_coff_eolica,
        registros_coff_solar,
        registros_pld,
        df_perfil_aneel,
    )


def carregar_dados(params: dict):
    """Carrega dados com spinner visível e tratamento de erro amigável"""
    with st.spinner("🔍 Coletando e processando dados do ONS/ANEEL..."):
        try:
            (
                registros_geracao,
                registros_coff_eolica,
                registros_coff_solar,
                registros_pld,
                df_perfil_aneel,
            ) = _carregar_dados_completos(params)
        except Exception as e:
            st.error(f"❌ Erro ao carregar dados: {e}")
            st.stop()

    if not registros_pld:
        st.error(
            "❌ Nenhum dado de PLD foi encontrado para o período selecionado. "
            "Tente expandir a janela de análise ou verificar a conectividade "
            "com o portal de dados abertos do ONS."
        )
        st.stop()

    registros_geracao_f, registros_pld_f = filter_by_submercado(
        registros_geracao, registros_pld, params['submercado']
    )

    registros_coff_eolica_f = [
        r for r in registros_coff_eolica
        if params['submercado'] in (None, 'TODOS')
        or r.submercado == params['submercado']
    ]
    registros_coff_solar_f = [
        r for r in registros_coff_solar
        if params['submercado'] in (None, 'TODOS')
        or r.submercado == params['submercado']
    ]

    return (
        registros_geracao_f,
        registros_coff_eolica_f,
        registros_coff_solar_f,
        registros_pld_f,
        df_perfil_aneel,
    )

# ======================================================================
# ABA 1: VISÃO GERAL
# ======================================================================
def render_tab_visao_geral(registros_geracao, registros_pld, df_perfil_aneel,
                           params: dict, registros_coff_eolica=None,
                           registros_coff_solar=None):
    """Aba de visão geral: KPIs consolidados e ranking de submercados"""
    st.header("📊 Visão Geral")

    finder = get_opportunity_finder(
        params['eficiencia'], params['capex_mwh'], params['capex_mw']
    )

    with st.spinner("Processando análise consolidada de oportunidades..."):
        oportunidades = finder.find_opportunities(
            registros_geracao, registros_pld,
            registros_coff_eolica=registros_coff_eolica,
            registros_coff_solar=registros_coff_solar,
            df_perfil_aneel=df_perfil_aneel if not df_perfil_aneel.empty else None,
        )

    if not oportunidades:
        st.warning(
            "⚠️ Nenhuma oportunidade de BESS identificada com os dados e "
            "parâmetros atuais. Tente ajustar o período de análise, "
            "incluir mais fontes de geração ou revisar o submercado filtrado."
        )
        return

    resumo = finder.generate_executive_report(oportunidades)
    df_oportunidades = finder.export_opportunities_to_dataframe(oportunidades)

    # --- KPIs em cards ---
    kpis = cards_kpi_resumo(resumo['resumo_executivo'])
    cols = st.columns(len(kpis))
    for col, kpi in zip(cols, kpis):
        col.metric(label=kpi['label'], value=kpi['value'], delta=kpi['delta'])

    st.divider()

    # --- Ranking + comparativo lado a lado ---
    col_left, col_right = st.columns([1, 1.3])

    with col_left:
        st.plotly_chart(
            grafico_ranking_submercados(df_oportunidades),
            use_container_width=True
        )

    with col_right:
        st.plotly_chart(
            grafico_comparativo_oportunidades(df_oportunidades),
            use_container_width=True
        )

    st.divider()

    # --- Tabela detalhada ---
    st.subheader("📋 Ranking Detalhado de Oportunidades")
    st.dataframe(
        df_oportunidades.style.format({
            'energia_disponivel_mwh': '{:,.0f}',
            'preco_medio_carga_rs_mwh': 'R$ {:,.2f}',
            'preco_medio_descarga_rs_mwh': 'R$ {:,.2f}',
            'spread_medio_rs_mwh': 'R$ {:,.2f}',
            'receita_estimada_anual_rs': 'R$ {:,.0f}',
            'capacidade_recomendada_mwh': '{:,.0f}',
            'potencia_recomendada_mw': '{:,.0f}',
            'payback_anos': '{:,.1f}',
            'score_atratividade': '{:,.1f}',
        }, na_rep='N/A'),
        use_container_width=True,
        hide_index=True,
    )

    # --- Download de resultados ---
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            "⬇️ Baixar Ranking (CSV)",
            data=df_oportunidades.to_csv(index=False, sep=';').encode('utf-8-sig'),
            file_name=f"oportunidades_bess_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )
    with col_dl2:
        import json
        st.download_button(
            "⬇️ Baixar Relatório Completo (JSON)",
            data=json.dumps(resumo, default=str, ensure_ascii=False, indent=2).encode('utf-8'),
            file_name=f"relatorio_bess_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json",
        )

    return oportunidades, df_oportunidades


# ======================================================================
# ABA 2: CURTAILMENT
# ======================================================================
def render_tab_curtailment(registros_geracao, registros_coff_eolica,
                           registros_coff_solar, params: dict):
    """Aba de análise detalhada de curtailment"""
    st.header("⚡ Análise de Curtailment")

    registros_coff = registros_coff_eolica + registros_coff_solar
    analyzer = get_curtailment_analyzer()

    if registros_coff:
        st.subheader("Dados oficiais de restrição COFF do ONS")
        st.caption(
            "Cada registro representa uma usina em um intervalo de 30 minutos. "
            "A potência restringida é calculada pela diferença positiva entre "
            "geração estimada e verificada, apenas quando a flag COFF está ativa."
        )
        df_coff = analyzer.analyze_curtailment_real(registros_coff)
        if not df_coff.empty:
            energia_total = df_coff['energia_cortada_total_mwh'].sum()
            usinas = len({registro.usina for registro in registros_coff})
            c1, c2, c3 = st.columns(3)
            c1.metric("Energia restringida estimada", f"{energia_total:,.0f} MWh")
            c2.metric("Usinas afetadas", f"{usinas:,.0f}")
            c3.metric("Registros de 30 min", f"{df_coff['num_registros'].sum():,.0f}")
            st.dataframe(
                df_coff,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info(
                "O ONS publicou dados COFF para o período, mas não há registros "
                "com flag de restrição e diferença positiva entre geração "
                "estimada e verificada."
            )
    else:
        st.warning(
            "Não há registros COFF disponíveis para o período selecionado. "
            "Sem programação de geração, os CSVs de geração verificada não "
            "permitem inferir curtailment."
        )

    tem_curtailment_inferivel = any(
        registro.curtailment_mw > 0 for registro in registros_geracao
    )
    if not registros_geracao:
        st.info(
            "ℹ️ Nenhum dado de geração disponível para o período/fontes "
            "selecionados."
        )
        return

    if not tem_curtailment_inferivel:
        st.info(
            "A geração programada não está publicada nos registros usados "
            "nesta análise. Os gráficos inferidos de geração programada versus "
            "verificada foram omitidos para não apresentar zeros enganosos; "
            "consulte os dados COFF acima."
        )
        return

    df_geracao = convert_registros_to_df(registros_geracao)

    submercados_disponiveis = sorted(df_geracao['submercado'].unique())
    fontes_disponiveis = sorted(df_geracao['fonte'].unique())

    col1, col2 = st.columns(2)
    with col1:
        submercado_sel = st.selectbox(
            "Submercado para análise", options=submercados_disponiveis, key="curt_sub"
        )
    with col2:
        fonte_sel = st.selectbox(
            "Fonte", options=['TODAS'] + fontes_disponiveis, key="curt_fonte"
        )

    # --- Métricas agregadas ---
    with st.spinner("Calculando métricas de curtailment..."):
        df_metricas = analyzer.analyze_curtailment_by_source(registros_geracao)

    if not df_metricas.empty:
        df_filtrado = df_metricas[df_metricas['submercado'] == submercado_sel]
        if fonte_sel != 'TODAS':
            df_filtrado = df_filtrado[df_filtrado['fonte'] == fonte_sel]

        if not df_filtrado.empty:
            linha = df_filtrado.iloc[0] if len(df_filtrado) == 1 else df_filtrado.agg({
                'energia_cortada_total_mwh': 'sum',
                'pct_tempo_com_curtailment': 'mean',
                'pct_energia_perdida': 'mean',
                'capacidade_instalada_mw': 'sum',
            })

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Energia Cortada Total", f"{linha['energia_cortada_total_mwh']:,.0f} MWh")
            c2.metric("% Tempo com Curtailment", f"{linha['pct_tempo_com_curtailment']*100:.1f}%")
            c3.metric("% Energia Perdida", f"{linha['pct_energia_perdida']*100:.1f}%")
            c4.metric("Capacidade Instalada", f"{linha['capacidade_instalada_mw']:,.0f} MW")

    st.divider()

    # --- Timeline de curtailment ---
    st.plotly_chart(
        grafico_curtailment_timeline(df_geracao, submercado_sel, fonte_sel),
        use_container_width=True
    )

    st.divider()

    # --- Padrões temporais ---
    col_a, col_b = st.columns(2)

    with st.spinner("Identificando padrões temporais..."):
        padroes = analyzer.analyze_temporal_patterns(registros_geracao)

    with col_a:
        if fonte_sel != 'TODAS':
            st.plotly_chart(
                grafico_curtailment_por_hora(
                    padroes.get('por_hora', pd.DataFrame()), submercado_sel, fonte_sel
                ),
                use_container_width=True
            )
        else:
            st.info("Selecione uma fonte específica para ver o padrão horário detalhado")

    with col_b:
        fonte_heatmap = fonte_sel if fonte_sel != 'TODAS' else (
            fontes_disponiveis[0] if fontes_disponiveis else None
        )
        if fonte_heatmap:
            st.plotly_chart(
                grafico_heatmap_curtailment(padroes.get('por_hora', pd.DataFrame()), fonte_heatmap),
                use_container_width=True
            )

    st.divider()

    # --- Eventos discretos (tabela) ---
    st.subheader("🔍 Principais Eventos de Curtailment Detectados")
    with st.spinner("Detectando eventos discretos..."):
        eventos = analyzer.detect_curtailment_events(registros_geracao, min_duracao_horas=1)

    eventos_filtrados = [e for e in eventos if e.submercado == submercado_sel]

    if eventos_filtrados:
        df_eventos = pd.DataFrame([{
            'Início': e.timestamp.strftime('%Y-%m-%d %H:%M'),
            'Duração (h)': e.duracao_horas,
            'Magnitude Média (MW)': round(e.magnitude_mw, 1),
            'Energia Estimada (MWh)': round(e.magnitude_mw * e.duracao_horas, 1),
            'Tipo': e.tipo.value,
            'Descrição': e.descricao,
        } for e in eventos_filtrados])

        df_eventos = df_eventos.sort_values('Energia Estimada (MWh)', ascending=False)
        st.dataframe(df_eventos.head(20), use_container_width=True, hide_index=True)
    else:
        st.info("Nenhum evento discreto de curtailment detectado para este submercado.")

# ======================================================================
# ABA 3: SPREAD DE PREÇOS (PLD)
# ======================================================================
def render_tab_spread(registros_pld, params: dict):
    """Aba de análise detalhada de spread de preços"""
    st.header("💹 Análise de Spread de Preços (PLD)")

    if not registros_pld:
        st.info("ℹ️ Nenhum dado de PLD disponível para o período selecionado.")
        return

    analyzer = get_spread_analyzer(params['eficiencia'])
    df_pld = convert_pld_to_df(registros_pld)

    submercados_disponiveis = sorted(df_pld['submercado'].unique())

    submercado_sel = st.selectbox(
        "Submercado para análise", options=submercados_disponiveis, key="spread_sub"
    )

    # --- Ranking de spread (todos os submercados) ---
    with st.spinner("Calculando ranking de spread..."):
        ranking_spread = analyzer.rank_submercados_por_spread(registros_pld)

    if not ranking_spread.empty:
        c1, c2, c3, c4 = st.columns(4)
        linha_sel = ranking_spread[ranking_spread['submercado'] == submercado_sel]

        if not linha_sel.empty:
            linha_sel = linha_sel.iloc[0]
            c1.metric("Spread Médio", f"R$ {linha_sel['spread_medio_rs_mwh']:,.2f}/MWh")
            c2.metric("Spread P90", f"R$ {linha_sel['spread_p90_rs_mwh']:,.2f}/MWh")
            c3.metric("Consistência Horário", f"{linha_sel['consistencia_horario_pct']:.0f}%")
            c4.metric("Score de Spread", f"{linha_sel['score_spread']:.1f}/100")

    st.divider()

    # --- Janelas ótimas ---
    with st.spinner("Identificando janelas ótimas de carga/descarga..."):
        janelas = analyzer.identify_optimal_windows(registros_pld, submercado_sel)

    col_left, col_right = st.columns([1.5, 1])

    with col_left:
        if janelas:
            st.plotly_chart(
                grafico_perfil_horario_pld(
                    janelas.get('perfil_horario_completo', []),
                    submercado_sel,
                    hora_carga_ini=janelas['janela_carga']['hora_inicio'],
                    hora_carga_fim=janelas['janela_carga']['hora_fim'],
                    hora_descarga_ini=janelas['janela_descarga']['hora_inicio'],
                    hora_descarga_fim=janelas['janela_descarga']['hora_fim'],
                ),
                use_container_width=True
            )
        else:
            st.info("Não foi possível identificar janelas ótimas para este submercado.")

    with col_right:
        if janelas:
            st.markdown("#### 🟢 Janela de Carga Recomendada")
            st.write(
                f"**{janelas['janela_carga']['hora_inicio']:02d}h "
                f"- {janelas['janela_carga']['hora_fim']:02d}h**"
            )
            st.caption(f"PLD médio: R$ {janelas['janela_carga']['pld_medio']:,.2f}/MWh")

            st.markdown("#### 🔴 Janela de Descarga Recomendada")
            st.write(
                f"**{janelas['janela_descarga']['hora_inicio']:02d}h "
                f"- {janelas['janela_descarga']['hora_fim']:02d}h**"
            )
            st.caption(f"PLD médio: R$ {janelas['janela_descarga']['pld_medio']:,.2f}/MWh")

            st.markdown("#### 💰 Spread Estimado")
            st.write(f"**Bruto:** R$ {janelas['spread_bruto_rs_mwh']:,.2f}/MWh")
            st.write(f"**Líquido (c/ eficiência):** R$ {janelas['spread_liquido_rs_mwh']:,.2f}/MWh")

    st.divider()

    # --- Série temporal completa do PLD ---
    st.plotly_chart(
        grafico_pld_timeline(
            df_pld, submercado_sel,
            janela_carga=janelas.get('janela_carga') if janelas else None,
            janela_descarga=janelas.get('janela_descarga') if janelas else None,
        ),
        use_container_width=True
    )

    st.divider()

    # --- Distribuição e sazonalidade ---
    col_a, col_b = st.columns(2)

    with st.spinner("Calculando distribuição e sazonalidade..."):
        df_spread_diario = analyzer.calculate_daily_spread(registros_pld)
        df_sazonalidade = analyzer.analyze_seasonal_spread(registros_pld)

    with col_a:
        st.plotly_chart(
            grafico_spread_diario_boxplot(df_spread_diario),
            use_container_width=True
        )

    with col_b:
        st.plotly_chart(
            grafico_sazonalidade_spread(df_sazonalidade),
            use_container_width=True
        )

    return janelas


# ======================================================================
# ABA 4: SIMULAÇÃO DE OPERAÇÃO DO BESS
# ======================================================================
def render_tab_simulacao(registros_pld, params: dict):
    """Aba de simulação interativa de operação de arbitragem"""
    st.header("🔄 Simulação de Operação do BESS")

    if not registros_pld:
        st.info("ℹ️ Nenhum dado de PLD disponível para simulação.")
        return

    analyzer = get_spread_analyzer(params['eficiencia'])
    df_pld = convert_pld_to_df(registros_pld)
    submercados_disponiveis = sorted(df_pld['submercado'].unique())

    st.subheader("⚙️ Parâmetros da Simulação")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        submercado_sel = st.selectbox(
            "Submercado", options=submercados_disponiveis, key="sim_sub"
        )
    with col2:
        capacidade_mwh = st.number_input(
            "Capacidade (MWh)", min_value=1.0, max_value=1000.0,
            value=100.0, step=10.0, key="sim_cap"
        )
    with col3:
        potencia_mw = st.number_input(
            "Potência (MW)", min_value=1.0, max_value=500.0,
            value=25.0, step=5.0, key="sim_pot"
        )
    with col4:
        estrategia = st.selectbox(
            "Estratégia", options=['janela_fixa', 'percentil_diario'],
            format_func=lambda x: {
                'janela_fixa': 'Janela Fixa (recorrente)',
                'percentil_diario': 'Percentil Diário (adaptativa)'
            }[x],
            key="sim_estrategia"
        )

    if st.button("▶️ Executar Simulação", type="primary"):
        with st.spinner("Executando backtest hora a hora..."):
            df_simulacao = analyzer.simulate_arbitrage_operation(
                registros_pld, submercado_sel, capacidade_mwh, potencia_mw,
                estrategia=estrategia
            )
            metricas = analyzer.calculate_simulation_metrics(df_simulacao, capacidade_mwh)

        if df_simulacao.empty:
            st.warning("Simulação não produziu resultados. Verifique os dados de entrada.")
            return

        st.session_state['df_simulacao'] = df_simulacao
        st.session_state['metricas_simulacao'] = metricas
        st.session_state['capacidade_simulacao'] = capacidade_mwh

    if 'df_simulacao' in st.session_state:
        df_simulacao = st.session_state['df_simulacao']
        metricas = st.session_state['metricas_simulacao']
        capacidade_mwh_sim = st.session_state['capacidade_simulacao']

        st.divider()
        st.subheader("📈 Resultados da Simulação")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Receita Total", formatar_moeda(metricas.get('receita_total_periodo_rs', 0)))
        c2.metric("Receita Anualizada", formatar_moeda(metricas.get('receita_anualizada_rs', 0)))
        c3.metric("Ciclos Equivalentes", f"{metricas.get('ciclos_equivalentes_periodo', 0):.1f}")
        c4.metric("Ciclos/Ano Estimados", f"{metricas.get('ciclos_equivalentes_anualizados', 0):.0f}")
        c5.metric("Eficiência Efetiva", f"{metricas.get('eficiencia_efetiva_pct', 0):.1f}%")

        st.divider()

        st.plotly_chart(
            grafico_simulacao_operacao(df_simulacao, capacidade_mwh_sim),
            use_container_width=True
        )

        st.plotly_chart(
            grafico_fluxo_caixa_acumulado(df_simulacao),
            use_container_width=True
        )

        with st.expander("📋 Ver dados detalhados da simulação (hora a hora)"):
            st.dataframe(df_simulacao, use_container_width=True, hide_index=True)

        st.download_button(
            "⬇️ Baixar Simulação Completa (CSV)",
            data=df_simulacao.to_csv(index=False, sep=';').encode('utf-8-sig'),
            file_name=f"simulacao_bess_{submercado_sel}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )
    else:
        st.info("👆 Configure os parâmetros e clique em 'Executar Simulação' para ver os resultados.")


# ======================================================================
# ABA 5: OPORTUNIDADES E SENSIBILIDADE
# ======================================================================
def render_tab_oportunidades(oportunidades, df_oportunidades, params: dict):
    """Aba de detalhamento financeiro das oportunidades e análise de sensibilidade"""
    st.header("💡 Oportunidades de Investimento")

    if not oportunidades:
        st.info("ℹ️ Nenhuma oportunidade disponível. Veja a aba 'Visão Geral' primeiro.")
        return

    finder = get_opportunity_finder(
        params['eficiencia'], params['capex_mwh'], params['capex_mw']
    )

    nomes_oportunidades = [
        f"{o.submercado} (Score: {o.score_atratividade:.1f})" for o in oportunidades
    ]
    idx_sel = st.selectbox(
        "Selecione uma oportunidade para análise detalhada",
        options=range(len(oportunidades)),
        format_func=lambda i: nomes_oportunidades[i],
    )

    oportunidade_sel = oportunidades[idx_sel]

    # --- Cards de detalhamento ---
    st.subheader(f"📍 {oportunidade_sel.submercado} — Detalhamento")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Capacidade Recomendada", f"{oportunidade_sel.capacidade_recomendada_mwh:,.0f} MWh")
    col2.metric("Potência Recomendada", f"{oportunidade_sel.potencia_recomendada_mw:,.0f} MW")
    col3.metric("Receita Anual Estimada", formatar_moeda(oportunidade_sel.receita_estimada_anual_rs))
    col4.metric(
        "Payback Simples",
        f"{oportunidade_sel.payback_anos:.1f} anos" if oportunidade_sel.payback_anos else "N/A"
    )

    st.divider()

    # --- Análise financeira completa (VPL/TIR) ---
    st.subheader("💰 Análise Financeira (VPL/TIR)")

    capex_estimado = (
        oportunidade_sel.capacidade_recomendada_mwh * params['capex_mwh'] +
        oportunidade_sel.potencia_recomendada_mw * params['capex_mw']
    )

    with st.spinner("Calculando VPL e TIR..."):
        analise_financeira = finder.calculate_npv(
            capex_total=capex_estimado,
            receita_anual_rs=oportunidade_sel.receita_estimada_anual_rs,
        )

    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    col_f1.metric("CAPEX Total Estimado", formatar_moeda(analise_financeira['capex_total_rs']))
    col_f2.metric(
        "VPL",
        formatar_moeda(analise_financeira['vpl_rs']),
        delta="Viável ✅" if analise_financeira['viavel'] else "Inviável ❌",
        delta_color="normal" if analise_financeira['viavel'] else "inverse",
    )
    col_f3.metric(
        "TIR",
        f"{analise_financeira['tir_pct']:.1f}%" if analise_financeira.get('tir_pct') else "N/A"
    )
    col_f4.metric(
        "Payback Descontado",
        f"{analise_financeira['payback_descontado_anos']:.1f} anos"
        if analise_financeira.get('payback_descontado_anos') else "N/A"
    )

    with st.expander("ℹ️ Ver premissas econômicas utilizadas"):
        st.json({
            'capex_rs_por_mwh': params['capex_mwh'],
            'capex_rs_por_mw': params['capex_mw'],
            'opex_pct_capex_ano': finder.opex_pct_capex_ano,
            'vida_util_anos': finder.vida_util_anos,
            'taxa_desconto': finder.taxa_desconto,
            'eficiencia_bess': params['eficiencia'],
        })

    st.divider()

    # --- Análise de sensibilidade ---
    st.subheader("🎯 Análise de Sensibilidade")
    st.caption(
        "Avalia a robustez da oportunidade frente a variações no CAPEX "
        "(ex.: queda no preço de baterias) e no spread de preços "
        "(ex.: mudanças regulatórias no PLD)."
    )

    if st.button("🔬 Executar Análise de Sensibilidade", key="btn_sensitivity"):
        with st.spinner("Calculando matriz de sensibilidade..."):
            df_sensitivity = finder.sensitivity_analysis(oportunidade_sel)
            st.session_state['df_sensitivity'] = df_sensitivity

    if 'df_sensitivity' in st.session_state:
        df_sensitivity = st.session_state['df_sensitivity']

        st.plotly_chart(
            grafico_sensibilidade_heatmap(df_sensitivity),
            use_container_width=True
        )

        pct_viavel = (df_sensitivity['viavel'].sum() / len(df_sensitivity)) * 100
        st.info(f"📊 **{pct_viavel:.0f}%** dos cenários testados resultam em VPL positivo.")

        with st.expander("📋 Ver matriz completa de cenários"):
            st.dataframe(df_sensitivity, use_container_width=True, hide_index=True)

    st.divider()

    # --- Descrição textual da restrição identificada ---
    st.subheader("📝 Contexto da Oportunidade")
    st.markdown(f"""
    - **Tipo de restrição predominante:** `{oportunidade_sel.tipo_restricao.value}`
    - **Período analisado:** {oportunidade_sel.periodo_inicio.strftime('%d/%m/%Y')} a
      {oportunidade_sel.periodo_fim.strftime('%d/%m/%Y')}
    - **Energia disponível identificada:** {oportunidade_sel.energia_disponivel_mwh:,.0f} MWh
    - **Preço médio de carga:** R$ {oportunidade_sel.preco_medio_carga_rs_mwh:,.2f}/MWh
    - **Preço médio de descarga:** R$ {oportunidade_sel.preco_medio_descarga_rs_mwh:,.2f}/MWh
    - **Volatilidade de preço (desvio padrão):** R$ {oportunidade_sel.volatilidade_preco:,.2f}/MWh
    - **Ciclos possíveis estimados/ano:** {oportunidade_sel.num_ciclos_possiveis:,.0f}
    """)


# ======================================================================
# FUNÇÃO PRINCIPAL (ORQUESTRAÇÃO)
# ======================================================================
def main():
    """Função principal do dashboard - orquestra sidebar e abas"""

    # --- Header ---
    st.title("🔋 BESS Arbitrage Analyzer")
    st.markdown(
        "Dashboard interativo para mapeamento de restrições de geração "
        "(curtailment) e identificação de oportunidades de arbitragem "
        "via sistemas de armazenamento de energia (BESS), com base em "
        "dados abertos do **ONS** e **ANEEL**."
    )

    # --- Sidebar com parâmetros ---
    params = render_sidebar()

    if not params['fontes']:
        st.warning("⚠️ Selecione ao menos uma fonte de geração na barra lateral.")
        st.stop()

    # --- Carregamento centralizado de dados ---
    (
        registros_geracao,
        registros_coff_eolica,
        registros_coff_solar,
        registros_pld,
        df_perfil_aneel,
    ) = carregar_dados(params)

    st.success(
        f"✅ Dados carregados: {len(registros_geracao):,} registros de geração, "
        f"{len(registros_pld):,} registros de PLD e "
        f"{len(registros_coff_eolica) + len(registros_coff_solar):,} registros COFF."
    )

    # --- Navegação por abas ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Visão Geral",
        "⚡ Curtailment",
        "💹 Spread de Preços",
        "🔄 Simulação BESS",
        "💡 Oportunidades"
    ])

    with tab1:
        resultado_visao_geral = render_tab_visao_geral(
            registros_geracao, registros_pld, df_perfil_aneel, params,
            registros_coff_eolica, registros_coff_solar,
        )

    with tab2:
        render_tab_curtailment(
            registros_geracao, registros_coff_eolica,
            registros_coff_solar, params,
        )

    with tab3:
        render_tab_spread(registros_pld, params)

    with tab4:
        render_tab_simulacao(registros_pld, params)

    with tab5:
        if resultado_visao_geral:
            oportunidades, df_oportunidades = resultado_visao_geral
            render_tab_oportunidades(oportunidades, df_oportunidades, params)
        else:
            st.info("ℹ️ Nenhuma oportunidade disponível para análise detalhada.")

    # --- Footer ---
    st.divider()
    st.caption(
        "Dados obtidos via portais de dados abertos do ONS (Operador Nacional "
        "do Sistema Elétrico) e ANEEL (Agência Nacional de Energia Elétrica). "
        "Análise e dimensionamento gerados automaticamente - recomenda-se "
        "validação técnico-econômica detalhada antes de decisões de investimento."
    )


if __name__ == "__main__":
    main()