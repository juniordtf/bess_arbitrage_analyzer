# visualization/charts.py
"""
Biblioteca de gráficos Plotly reutilizáveis para o dashboard.

Centraliza toda a lógica de formatação visual, paleta de cores e
estilo, garantindo consistência entre as diferentes páginas/abas
do dashboard.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Optional, List


# Paleta de cores consistente por submercado
CORES_SUBMERCADO = {
    'N': '#2E8B57',    # Verde (hidro-dominante)
    'NE': '#FF8C00',   # Laranja (eólico/solar-dominante)
    'S': '#4169E1',    # Azul
    'SE': '#DC143C',   # Vermelho (maior carga)
}

CORES_FONTE = {
    'EOLICA': '#00BFC4',
    'SOLAR': '#F8766D',
    'HIDRAULICA': '#619CFF',
    'TERMICA': '#7CAE00',
}


def cor_submercado(submercado: str) -> str:
    return CORES_SUBMERCADO.get(submercado, '#888888')


def cor_fonte(fonte: str) -> str:
    return CORES_FONTE.get(fonte, '#888888')


# ======================================================================
# GRÁFICOS: VISÃO GERAL / KPIs
# ======================================================================
def grafico_ranking_submercados(df_ranking: pd.DataFrame,
                                  coluna_score: str = 'score_atratividade',
                                  titulo: str = "Ranking de Submercados") -> go.Figure:
    """Gráfico de barras horizontais do ranking de submercados por score"""
    if df_ranking.empty:
        return _figura_vazia("Sem dados para exibir ranking")

    df_plot = df_ranking.sort_values(coluna_score, ascending=True)

    fig = go.Figure(go.Bar(
        x=df_plot[coluna_score],
        y=df_plot['submercado'],
        orientation='h',
        marker_color=[cor_submercado(s) for s in df_plot['submercado']],
        text=df_plot[coluna_score].round(1),
        textposition='outside',
    ))

    fig.update_layout(
        title=titulo,
        xaxis_title="Score (0-100)",
        yaxis_title="Submercado",
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False,
        template='plotly_white',
    )
    fig.update_xaxes(range=[0, 105])

    return fig


def cards_kpi_resumo(resumo_executivo: Dict) -> List[Dict]:
    """
    Retorna lista de dicts formatados para exibição em st.metric(),
    representando os KPIs principais do resumo executivo.
    """
    return [
        {
            'label': '🏆 Submercado Mais Atrativo',
            'value': resumo_executivo.get('submercado_mais_atrativo', 'N/A'),
            'delta': f"Score: {resumo_executivo.get('score_melhor_oportunidade', 0):.1f}/100"
        },
        {
            'label': '⚡ Energia Identificada',
            'value': f"{resumo_executivo.get('energia_total_identificada_mwh', 0):,.0f} MWh",
            'delta': None
        },
        {
            'label': '🔋 Capacidade Recomendada',
            'value': f"{resumo_executivo.get('capacidade_total_recomendada_mwh', 0):,.0f} MWh",
            'delta': None
        },
        {
            'label': '💰 Receita Potencial Anual',
            'value': f"R$ {resumo_executivo.get('receita_total_potencial_anual_rs', 0):,.0f}",
            'delta': None
        },
        {
            'label': '⏱️ Payback Médio',
            'value': (
                f"{resumo_executivo.get('payback_medio_anos', 0):.1f} anos"
                if resumo_executivo.get('payback_medio_anos') else "N/A"
            ),
            'delta': None
        },
    ]


# ======================================================================
# GRÁFICOS: CURTAILMENT
# ======================================================================
def grafico_curtailment_timeline(df_geracao: pd.DataFrame,
                                   submercado: str,
                                   fonte: Optional[str] = None) -> go.Figure:
    """
    Gráfico de área empilhada mostrando geração programada vs. verificada
    ao longo do tempo, com a área de curtailment destacada em vermelho.
    """
    if df_geracao.empty:
        return _figura_vazia("Sem dados de geração disponíveis")

    df_plot = df_geracao[df_geracao['submercado'] == submercado].copy()
    if fonte and fonte != 'TODAS':
        df_plot = df_plot[df_plot['fonte'] == fonte]

    if df_plot.empty:
        return _figura_vazia(f"Sem dados para {submercado} / {fonte}")

    df_plot = df_plot.groupby('timestamp').agg({
        'geracao_programada_mw': 'sum',
        'geracao_verificada_mw': 'sum',
        'curtailment_mw': 'sum',
    }).reset_index().sort_values('timestamp')

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_plot['timestamp'], y=df_plot['geracao_programada_mw'],
        name='Geração Programada', line=dict(color='#888888', dash='dash'),
        mode='lines'
    ))

    fig.add_trace(go.Scatter(
        x=df_plot['timestamp'], y=df_plot['geracao_verificada_mw'],
        name='Geração Verificada', line=dict(color=cor_submercado(submercado)),
        fill='tonexty', fillcolor='rgba(220,20,60,0.2)',
        mode='lines'
    ))

    fig.update_layout(
        title=f"Curtailment ao Longo do Tempo - {submercado}" +
              (f" ({fonte})" if fonte and fonte != 'TODAS' else ""),
        xaxis_title="Data/Hora",
        yaxis_title="Potência (MW)",
        height=400,
        template='plotly_white',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )

    return fig


def grafico_curtailment_por_hora(df_padrao_horario: pd.DataFrame,
                                   submercado: str,
                                   fonte: str) -> go.Figure:
    """Gráfico de barras mostrando % de ocorrência de curtailment por hora do dia"""
    if df_padrao_horario.empty:
        return _figura_vazia("Sem dados de padrão horário")

    df_plot = df_padrao_horario[
        (df_padrao_horario['submercado'] == submercado) &
        (df_padrao_horario['fonte'] == fonte)
    ].sort_values('hora')

    if df_plot.empty:
        return _figura_vazia(f"Sem padrão horário para {submercado}/{fonte}")

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=df_plot['hora'], y=df_plot['pct_ocorrencia'] * 100,
            name='% Horas com Curtailment',
            marker_color=cor_fonte(fonte), opacity=0.7
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=df_plot['hora'], y=df_plot['curtailment_medio_mw'],
            name='Curtailment Médio (MW)',
            line=dict(color='#333333', width=2),
            mode='lines+markers'
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title=f"Padrão Horário de Curtailment - {submercado} ({fonte})",
        xaxis_title="Hora do Dia",
        height=400,
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    fig.update_yaxes(title_text="% de Ocorrência", secondary_y=False)
    fig.update_yaxes(title_text="Curtailment Médio (MW)", secondary_y=True)
    fig.update_xaxes(tickmode='linear', tick0=0, dtick=2)

    return fig


def grafico_heatmap_curtailment(df_padrao_horario: pd.DataFrame,
                                  fonte: str) -> go.Figure:
    """Heatmap de curtailment médio: submercado (linha) x hora (coluna)"""
    if df_padrao_horario.empty:
        return _figura_vazia("Sem dados para heatmap")

    df_fonte = df_padrao_horario[df_padrao_horario['fonte'] == fonte]
    if df_fonte.empty:
        return _figura_vazia(f"Sem dados de curtailment para {fonte}")

    pivot = df_fonte.pivot_table(
        index='submercado', columns='hora', values='curtailment_medio_mw',
        aggfunc='mean'
    ).reindex(columns=range(24))

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale='Reds',
        colorbar=dict(title="MW"),
        hovertemplate='Submercado: %{y}<br>Hora: %{x}h<br>Curtailment: %{z:.1f} MW<extra></extra>',
    ))

    fig.update_layout(
        title=f"Heatmap de Curtailment Médio por Hora - {fonte}",
        xaxis_title="Hora do Dia",
        yaxis_title="Submercado",
        height=300,
        template='plotly_white',
    )
    fig.update_xaxes(tickmode='linear', tick0=0, dtick=1)

    return fig


# ======================================================================
# GRÁFICOS: SPREAD DE PREÇOS (PLD)
# ======================================================================
def grafico_pld_timeline(df_pld: pd.DataFrame,
                           submercado: str,
                           janela_carga: Optional[Dict] = None,
                           janela_descarga: Optional[Dict] = None) -> go.Figure:
    """
    Série temporal de PLD, com destaque opcional das janelas horárias
    de carga (verde) e descarga (vermelho) identificadas pela análise.
    """
    if df_pld.empty:
        return _figura_vazia("Sem dados de PLD disponíveis")

    df_plot = df_pld[df_pld['submercado'] == submercado].sort_values('timestamp')
    if df_plot.empty:
        return _figura_vazia(f"Sem dados de PLD para {submercado}")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_plot['timestamp'], y=df_plot['pld_mwh'],
        name='PLD', line=dict(color=cor_submercado(submercado), width=1.5),
        mode='lines'
    ))

    # Destaca janelas de carga/descarga como faixas horizontais de referência
    if janela_carga:
        fig.add_hline(
            y=janela_carga.get('pld_medio'), line_dash='dot', line_color='green',
            annotation_text=f"Preço médio carga: R$ {janela_carga.get('pld_medio', 0):.0f}",
            annotation_position='bottom right'
        )
    if janela_descarga:
        fig.add_hline(
            y=janela_descarga.get('pld_medio'), line_dash='dot', line_color='red',
            annotation_text=f"Preço médio descarga: R$ {janela_descarga.get('pld_medio', 0):.0f}",
            annotation_position='top right'
        )

    fig.update_layout(
        title=f"PLD Horário - {submercado}",
        xaxis_title="Data/Hora",
        yaxis_title="PLD (R$/MWh)",
        height=400,
        template='plotly_white',
        hovermode='x unified',
    )

    return fig


def grafico_perfil_horario_pld(perfil_horario: List[Dict],
                                 submercado: str,
                                 hora_carga_ini: Optional[int] = None,
                                 hora_carga_fim: Optional[int] = None,
                                 hora_descarga_ini: Optional[int] = None,
                                 hora_descarga_fim: Optional[int] = None) -> go.Figure:
    """
    Gráfico de linha do PLD médio por hora do dia, com faixas sombreadas
    destacando visualmente as janelas ótimas de carga (verde) e
    descarga (vermelho) identificadas pela análise.
    """
    if not perfil_horario:
        return _figura_vazia("Sem perfil horário disponível")

    df_plot = pd.DataFrame(perfil_horario).sort_values('hora')

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_plot['hora'], y=df_plot['pld_medio'],
        name='PLD Médio', line=dict(color=cor_submercado(submercado), width=3),
        mode='lines+markers',
        error_y=dict(
            type='data',
            array=df_plot['pld_std'].fillna(0),
            visible=True,
            color='rgba(100,100,100,0.3)'
        )
    ))

    if hora_carga_ini is not None:
        _adicionar_faixa_horaria(fig, hora_carga_ini, hora_carga_fim,
                                   'rgba(0,150,0,0.15)', 'Janela de Carga')

    if hora_descarga_ini is not None:
        _adicionar_faixa_horaria(fig, hora_descarga_ini, hora_descarga_fim,
                                   'rgba(200,0,0,0.15)', 'Janela de Descarga')

    fig.update_layout(
        title=f"Perfil Horário Médio de PLD - {submercado}",
        xaxis_title="Hora do Dia",
        yaxis_title="PLD Médio (R$/MWh)",
        height=400,
        template='plotly_white',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    fig.update_xaxes(tickmode='linear', tick0=0, dtick=1, range=[-0.5, 23.5])

    return fig


def _adicionar_faixa_horaria(fig: go.Figure,
                                hora_ini: int,
                                hora_fim: int,
                                cor: str,
                                label: str):
    """
    Adiciona uma faixa sombreada vertical ao gráfico, tratando o caso
    de janela que cruza a meia-noite (ex.: 22h-02h), que precisa ser
    desenhada em duas partes.
    """
    if hora_ini <= hora_fim:
        fig.add_vrect(
            x0=hora_ini - 0.5, x1=hora_fim + 0.5,
            fillcolor=cor, layer='below', line_width=0,
            annotation_text=label, annotation_position='top left',
        )
    else:
        # Janela cruza meia-noite: desenha de hora_ini até 23h e de 0h até hora_fim
        fig.add_vrect(
            x0=hora_ini - 0.5, x1=23.5,
            fillcolor=cor, layer='below', line_width=0,
        )
        fig.add_vrect(
            x0=-0.5, x1=hora_fim + 0.5,
            fillcolor=cor, layer='below', line_width=0,
            annotation_text=label, annotation_position='top left',
        )


def grafico_spread_diario_boxplot(df_spread_diario: pd.DataFrame) -> go.Figure:
    """
    Boxplot comparativo do spread diário de PLD entre submercados -
    visualiza dispersão e outliers de forma mais rica que apenas a média.
    """
    if df_spread_diario.empty:
        return _figura_vazia("Sem dados de spread diário")

    fig = go.Figure()

    for submercado in sorted(df_spread_diario['submercado'].unique()):
        df_sub = df_spread_diario[df_spread_diario['submercado'] == submercado]
        fig.add_trace(go.Box(
            y=df_sub['spread_absoluto'],
            name=submercado,
            marker_color=cor_submercado(submercado),
            boxmean='sd',
        ))

    fig.update_layout(
        title="Distribuição do Spread Diário de PLD por Submercado",
        yaxis_title="Spread Diário (R$/MWh)",
        xaxis_title="Submercado",
        height=400,
        template='plotly_white',
        showlegend=False,
    )

    return fig


def grafico_sazonalidade_spread(df_sazonalidade: pd.DataFrame) -> go.Figure:
    """Linha de evolução mensal do spread médio, uma linha por submercado"""
    if df_sazonalidade.empty:
        return _figura_vazia("Sem dados de sazonalidade")

    fig = go.Figure()

    nomes_mes = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun',
                 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']

    for submercado in sorted(df_sazonalidade['submercado'].unique()):
        df_sub = df_sazonalidade[df_sazonalidade['submercado'] == submercado].sort_values('mes')
        fig.add_trace(go.Scatter(
            x=[nomes_mes[m - 1] for m in df_sub['mes']],
            y=df_sub['spread_medio_rs_mwh'],
            name=submercado,
            mode='lines+markers',
            line=dict(color=cor_submercado(submercado), width=2.5),
        ))

    fig.update_layout(
        title="Sazonalidade do Spread Médio de PLD por Mês",
        xaxis_title="Mês",
        yaxis_title="Spread Médio (R$/MWh)",
        height=400,
        template='plotly_white',
        hovermode='x unified',
    )

    return fig


# ======================================================================
# GRÁFICOS: SIMULAÇÃO DE OPERAÇÃO DO BESS
# ======================================================================
def grafico_simulacao_operacao(df_simulacao: pd.DataFrame,
                                 capacidade_mwh: float) -> go.Figure:
    """
    Gráfico duplo-eixo mostrando a operação simulada do BESS: PLD no
    eixo esquerdo (com marcadores coloridos indicando carga/descarga)
    e SOC (%) no eixo direito.
    """
    if df_simulacao.empty:
        return _figura_vazia("Sem dados de simulação disponíveis")

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=df_simulacao['timestamp'], y=df_simulacao['pld_mwh'],
            name='PLD', line=dict(color='#444444', width=1),
            mode='lines', opacity=0.6
        ),
        secondary_y=False,
    )

    cores_acao = {'carga': 'green', 'descarga': 'red', 'ocioso': 'rgba(0,0,0,0)'}
    for acao, cor in cores_acao.items():
        if acao == 'ocioso':
            continue
        df_acao = df_simulacao[df_simulacao['acao'] == acao]
        fig.add_trace(
            go.Scatter(
                x=df_acao['timestamp'], y=df_acao['pld_mwh'],
                name=f'{acao.capitalize()}',
                mode='markers',
                marker=dict(color=cor, size=6, symbol='circle'),
            ),
            secondary_y=False,
        )

    fig.add_trace(
        go.Scatter(
            x=df_simulacao['timestamp'], y=df_simulacao['soc_pct'],
            name='SOC (%)', line=dict(color='#1f77b4', width=2, dash='dot'),
            mode='lines'
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title=f"Simulação de Operação do BESS ({capacidade_mwh:.0f} MWh)",
        xaxis_title="Data/Hora",
        height=450,
        template='plotly_white',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    fig.update_yaxes(title_text="PLD (R$/MWh)", secondary_y=False)
    fig.update_yaxes(title_text="SOC (%)", secondary_y=True, range=[0, 100])

    return fig


def grafico_fluxo_caixa_acumulado(df_simulacao: pd.DataFrame) -> go.Figure:
    """Gráfico de fluxo de caixa acumulado da simulação (receita líquida ao longo do tempo)"""
    if df_simulacao.empty:
        return _figura_vazia("Sem dados de simulação")

    df_plot = df_simulacao.copy().sort_values('timestamp')
    df_plot['fluxo_acumulado'] = df_plot['fluxo_caixa_rs'].cumsum()

    fig = go.Figure(go.Scatter(
        x=df_plot['timestamp'], y=df_plot['fluxo_acumulado'],
        fill='tozeroy',
        line=dict(color='#2E8B57', width=2),
        fillcolor='rgba(46,139,87,0.2)',
    ))

    fig.update_layout(
        title="Fluxo de Caixa Acumulado (Simulação)",
        xaxis_title="Data/Hora",
        yaxis_title="R$ Acumulado",
        height=350,
        template='plotly_white',
    )

    return fig


# ======================================================================
# GRÁFICOS: OPORTUNIDADES BESS (CARDS/COMPARATIVO)
# ======================================================================
def grafico_comparativo_oportunidades(df_oportunidades: pd.DataFrame) -> go.Figure:
    """
    Scatter plot bolha: eixo X = payback, eixo Y = receita anual,
    tamanho da bolha = capacidade recomendada, cor = submercado.
    Permite visualizar rapidamente o trade-off risco x retorno.
    """
    if df_oportunidades.empty:
        return _figura_vazia("Sem oportunidades para comparar")

    df_plot = df_oportunidades.dropna(subset=['payback_anos']).copy()
    if df_plot.empty:
        return _figura_vazia("Nenhuma oportunidade com payback calculável")

    fig = px.scatter(
        df_plot,
        x='payback_anos',
        y='receita_estimada_anual_rs',
        size='capacidade_recomendada_mwh',
        color='submercado',
        color_discrete_map=CORES_SUBMERCADO,
        hover_data=['score_atratividade', 'spread_medio_rs_mwh', 'tipo_restricao'],
        size_max=45,
    )

    fig.update_layout(
        title="Comparativo de Oportunidades: Payback x Receita x Tamanho",
        xaxis_title="Payback (anos)",
        yaxis_title="Receita Anual Estimada (R$)",
        height=450,
        template='plotly_white',
    )

    return fig


def grafico_sensibilidade_heatmap(df_sensitivity: pd.DataFrame) -> go.Figure:
    """Heatmap de VPL para análise de sensibilidade CAPEX x Spread"""
    if df_sensitivity.empty:
        return _figura_vazia("Sem dados de sensibilidade")

    pivot = df_sensitivity.pivot(
        index='variacao_capex_pct',
        columns='variacao_spread_pct',
        values='vpl_rs'
    )

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[f"{c:+.0f}%" for c in pivot.columns],
        y=[f"{i:+.0f}%" for i in pivot.index],
        colorscale='RdYlGn',
        zmid=0,
        colorbar=dict(title="VPL (R$)"),
        text=pivot.values,
        texttemplate="%{text:,.0f}",
        textfont={"size": 10},
    ))

    fig.update_layout(
        title="Análise de Sensibilidade: VPL (R$) por Variação de CAPEX x Spread",
        xaxis_title="Variação do Spread de Preço (%)",
        yaxis_title="Variação do CAPEX (%)",
        height=400,
        template='plotly_white',
    )

    return fig

# ======================================================================
# UTILITÁRIOS
# ======================================================================
def _figura_vazia(mensagem: str) -> go.Figure:
    """Retorna uma figura placeholder com mensagem informativa, usada
    quando não há dados suficientes para renderizar o gráfico real -
    evita quebrar o layout do dashboard com exceções visuais."""
    fig = go.Figure()
    fig.add_annotation(
        text=f"📊 {mensagem}",
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16, color="#888888"),
    )
    fig.update_layout(
        height=300,
        template='plotly_white',
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def formatar_moeda(valor: float) -> str:
    """Formata valor numérico como moeda brasileira (R$ 1.234.567,89)"""
    if valor is None:
        return "N/A"
    return f"R$ {valor:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")


def formatar_numero_grande(valor: float, sufixo: str = "") -> str:
    """Formata números grandes com separador de milhar no padrão BR"""
    if valor is None:
        return "N/A"
    return f"{valor:,.0f}".replace(",", ".") + (f" {sufixo}" if sufixo else "")