# analysis/price_spread_analyzer.py
"""
Analisador de spread de preços (PLD) para dimensionamento de estratégias
de arbitragem via BESS.

Conceito central de arbitragem:
- CARREGAR a bateria nos horários de menor PLD (geralmente madrugada/
  horários de alta geração renovável/baixa demanda)
- DESCARREGAR nos horários de maior PLD (geralmente ponta/início da noite)

A viabilidade econômica depende de:
1. Magnitude do spread (diferença entre preço de descarga e carga)
2. Recorrência/previsibilidade do padrão diário
3. Eficiência round-trip do BESS (perdas de conversão)
4. Número de ciclos viáveis por dia/ano (vida útil vs. receita)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import logging

from config.settings import Settings
from models.data_models import RegistroPLD

logger = logging.getLogger(__name__)


class PriceSpreadAnalyzer:
    """
    Analisa séries de PLD horário para identificar e dimensionar
    oportunidades de arbitragem por submercado.
    """

    def __init__(self,
                 eficiencia_bess: float = None,
                 horas_carga_padrao: int = 4,
                 horas_descarga_padrao: int = 4):
        """
        Args:
            eficiencia_bess: eficiência round-trip (0-1). Ex.: 0.85 = 85%.
            horas_carga_padrao: duração padrão da janela de carga (horas)
                para dimensionamento quando C-rate não é especificado.
            horas_descarga_padrao: duração padrão da janela de descarga.
        """
        self.eficiencia_bess = eficiencia_bess or Settings.EFICIENCIA_BESS
        self.horas_carga_padrao = horas_carga_padrao
        self.horas_descarga_padrao = horas_descarga_padrao

    # ------------------------------------------------------------------
    # Construção do DataFrame base
    # ------------------------------------------------------------------
    def _registros_to_df(self, registros: List[RegistroPLD]) -> pd.DataFrame:
        if not registros:
            return pd.DataFrame()

        data = [{
            'timestamp': r.timestamp,
            'submercado': r.submercado,
            'pld_mwh': r.pld_mwh,
            'carga_mw': r.carga_mw,
        } for r in registros]

        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['data'] = df['timestamp'].dt.date
        df['hora'] = df['timestamp'].dt.hour
        df['mes'] = df['timestamp'].dt.month
        df['dia_semana'] = df['timestamp'].dt.dayofweek
        df['fim_de_semana'] = df['dia_semana'].isin([5, 6])

        return df.sort_values(['submercado', 'timestamp']).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Spread diário (métrica fundamental)
    # ------------------------------------------------------------------
    def calculate_daily_spread(self, registros: List[RegistroPLD]) -> pd.DataFrame:
        """
        Calcula o spread diário (max - min) de PLD por submercado,
        junto com os horários em que ocorrem o pico e o vale.
        """
        df = self._registros_to_df(registros)
        if df.empty:
            return pd.DataFrame()

        resultados = []

        for (submercado, data), grupo in df.groupby(['submercado', 'data']):
            if len(grupo) < 2:
                continue

            idx_min = grupo['pld_mwh'].idxmin()
            idx_max = grupo['pld_mwh'].idxmax()

            resultados.append({
                'submercado': submercado,
                'data': data,
                'pld_min': grupo['pld_mwh'].min(),
                'pld_max': grupo['pld_mwh'].max(),
                'pld_medio': grupo['pld_mwh'].mean(),
                'spread_absoluto': grupo['pld_mwh'].max() - grupo['pld_mwh'].min(),
                'spread_relativo_pct': (
                    (grupo['pld_mwh'].max() - grupo['pld_mwh'].min()) /
                    grupo['pld_mwh'].mean()
                    if grupo['pld_mwh'].mean() > 0 else 0.0
                ),
                'hora_vale': grupo.loc[idx_min, 'hora'],
                'hora_pico': grupo.loc[idx_max, 'hora'],
                'dia_semana': grupo['dia_semana'].iloc[0],
                'fim_de_semana': grupo['fim_de_semana'].iloc[0],
                'mes': grupo['mes'].iloc[0],
            })

        return pd.DataFrame(resultados)

    # ------------------------------------------------------------------
    # Janelas horárias ótimas (agregado, não apenas dia a dia)
    # ------------------------------------------------------------------
    def identify_optimal_windows(self,
                                   registros: List[RegistroPLD],
                                   submercado: str,
                                   duracao_carga_horas: int = None,
                                   duracao_descarga_horas: int = None) -> Dict:
        """
        Identifica as janelas horárias mais consistentes (ao longo de
        todo o período analisado) para carga e descarga do BESS,
        baseando-se no PLD médio por hora do dia.

        Diferente do spread diário (que varia dia a dia), esta análise
        busca o PADRÃO RECORRENTE mais robusto - essencial para
        estratégias de arbitragem programáticas (não oportunísticas).
        """
        duracao_carga_horas = duracao_carga_horas or self.horas_carga_padrao
        duracao_descarga_horas = duracao_descarga_horas or self.horas_descarga_padrao

        df = self._registros_to_df(registros)
        df_sub = df[df['submercado'] == submercado]

        if df_sub.empty:
            logger.warning(f"Sem dados de PLD para submercado {submercado}")
            return {}

        perfil_horario = df_sub.groupby('hora').agg(
            pld_medio=('pld_mwh', 'mean'),
            pld_std=('pld_mwh', 'std'),
            pld_mediana=('pld_mwh', 'median'),
        ).reset_index()

        perfil_horario = perfil_horario.sort_values('hora').reset_index(drop=True)

        # Busca a janela contígua de N horas com menor PLD médio (carga)
        janela_carga = self._encontrar_janela_otima(
            perfil_horario, duracao_carga_horas, minimizar=True
        )

        # Busca a janela contígua de N horas com maior PLD médio (descarga)
        janela_descarga = self._encontrar_janela_otima(
            perfil_horario, duracao_descarga_horas, minimizar=False
        )

        preco_medio_carga = janela_carga['pld_medio']
        preco_medio_descarga = janela_descarga['pld_medio']
        spread_estimado = preco_medio_descarga - preco_medio_carga

        return {
            'submercado': submercado,
            'janela_carga': {
                'hora_inicio': janela_carga['hora_inicio'],
                'hora_fim': janela_carga['hora_fim'],
                'duracao_horas': duracao_carga_horas,
                'pld_medio': round(preco_medio_carga, 2),
            },
            'janela_descarga': {
                'hora_inicio': janela_descarga['hora_inicio'],
                'hora_fim': janela_descarga['hora_fim'],
                'duracao_horas': duracao_descarga_horas,
                'pld_medio': round(preco_medio_descarga, 2),
            },
            'spread_bruto_rs_mwh': round(spread_estimado, 2),
            'spread_liquido_rs_mwh': round(
                self._calcular_spread_liquido(preco_medio_carga, preco_medio_descarga), 2
            ),
            'perfil_horario_completo': perfil_horario.to_dict('records'),
        }

    def _encontrar_janela_otima(self,
                                  perfil_horario: pd.DataFrame,
                                  duracao_horas: int,
                                  minimizar: bool) -> Dict:
        """
        Busca a janela circular (considera virada de dia, ex.: 22h-02h)
        de N horas consecutivas com menor ou maior PLD médio agregado.
        """
        n_horas_dia = 24
        valores = perfil_horario.set_index('hora')['pld_medio'].reindex(
            range(n_horas_dia)
        ).values

        melhor_valor = None
        melhor_inicio = 0

        for inicio in range(n_horas_dia):
            indices = [(inicio + i) % n_horas_dia for i in range(duracao_horas)]
            valor_agregado = np.nanmean([valores[i] for i in indices])

            if melhor_valor is None:
                melhor_valor = valor_agregado
                melhor_inicio = inicio
            elif minimizar and valor_agregado < melhor_valor:
                melhor_valor = valor_agregado
                melhor_inicio = inicio
            elif not minimizar and valor_agregado > melhor_valor:
                melhor_valor = valor_agregado
                melhor_inicio = inicio

        hora_fim = (melhor_inicio + duracao_horas - 1) % n_horas_dia

        return {
            'hora_inicio': melhor_inicio,
            'hora_fim': hora_fim,
            'pld_medio': melhor_valor,
        }

    # ------------------------------------------------------------------
    # Spread líquido (considerando eficiência do BESS)
    # ------------------------------------------------------------------
    def _calcular_spread_liquido(self,
                                   preco_carga: float,
                                   preco_descarga: float) -> float:
        """
        Calcula o spread líquido considerando a eficiência round-trip
        do BESS. A energia descarregada é sempre menor que a carregada
        devido às perdas de conversão (AC-DC-AC, auto-consumo, etc.)

        Receita líquida por MWh carregado:
        receita = (preco_descarga * eficiencia) - preco_carga
        """
        return (preco_descarga * self.eficiencia_bess) - preco_carga

    # ------------------------------------------------------------------
    # Simulação de operação histórica (backtest simplificado)
    # ------------------------------------------------------------------
    def simulate_arbitrage_operation(self,
                                       registros: List[RegistroPLD],
                                       submercado: str,
                                       capacidade_mwh: float,
                                       potencia_mw: float,
                                       estrategia: str = 'janela_fixa',
                                       soc_inicial_pct: float = 0.5,
                                       soc_min_pct: float = 0.1,
                                       soc_max_pct: float = 0.95) -> pd.DataFrame:
        """
        Simula a operação histórica de um BESS fazendo arbitragem,
        hora a hora, baseado no PLD real do período.

        Args:
            capacidade_mwh: capacidade energética do BESS
            potencia_mw: potência (define quanto pode carregar/descarregar por hora)
            estrategia: 'janela_fixa' (usa janelas horárias pré-definidas)
                        ou 'percentil_diario' (decide dinamicamente por dia,
                        carregando nas N horas mais baratas e descarregando
                        nas N horas mais caras de cada dia)
            soc_inicial_pct: estado de carga inicial (% da capacidade)
            soc_min_pct: SOC mínimo operacional (proteção de degradação)
            soc_max_pct: SOC máximo operacional

        Returns:
            DataFrame hora a hora com SOC, ação (carga/descarga/ocioso),
            energia movimentada e receita/custo de cada hora.
        """
        df = self._registros_to_df(registros)
        df_sub = df[df['submercado'] == submercado].copy()

        if df_sub.empty:
            logger.warning(f"Sem dados de PLD para simular submercado {submercado}")
            return pd.DataFrame()

        df_sub = df_sub.sort_values('timestamp').reset_index(drop=True)

        if estrategia == 'janela_fixa':
            df_sub = self._aplicar_estrategia_janela_fixa(df_sub, submercado)
        elif estrategia == 'percentil_diario':
            df_sub = self._aplicar_estrategia_percentil_diario(df_sub)
        else:
            raise ValueError(f"Estratégia desconhecida: {estrategia}")

        return self._executar_simulacao(
            df_sub, capacidade_mwh, potencia_mw,
            soc_inicial_pct, soc_min_pct, soc_max_pct
        )

    def _aplicar_estrategia_janela_fixa(self,
                                          df_sub: pd.DataFrame,
                                          submercado: str) -> pd.DataFrame:
        """Marca horas de carga/descarga baseado em janela fixa otimizada"""
        janelas = self.identify_optimal_windows(
            [RegistroPLD(timestamp=row['timestamp'], submercado=row['submercado'],
                         pld_mwh=row['pld_mwh'], carga_mw=row.get('carga_mw', 0))
             for _, row in df_sub.iterrows()],
            submercado
        )

        if not janelas:
            df_sub['sinal'] = 'ocioso'
            return df_sub

        h_carga_ini = janelas['janela_carga']['hora_inicio']
        h_carga_fim = janelas['janela_carga']['hora_fim']
        h_descarga_ini = janelas['janela_descarga']['hora_inicio']
        h_descarga_fim = janelas['janela_descarga']['hora_fim']

        def classificar_hora(hora):
            if self._hora_na_janela(hora, h_carga_ini, h_carga_fim):
                return 'carga'
            elif self._hora_na_janela(hora, h_descarga_ini, h_descarga_fim):
                return 'descarga'
            return 'ocioso'

        df_sub['sinal'] = df_sub['hora'].apply(classificar_hora)
        return df_sub

    @staticmethod
    def _hora_na_janela(hora: int, inicio: int, fim: int) -> bool:
        """Verifica se uma hora está dentro de uma janela, tratando virada de dia"""
        if inicio <= fim:
            return inicio <= hora <= fim
        else:  # janela cruza meia-noite (ex.: 22h-02h)
            return hora >= inicio or hora <= fim

    def _aplicar_estrategia_percentil_diario(self,
                                                df_sub: pd.DataFrame,
                                                pct_horas_carga: float = 0.25,
                                                pct_horas_descarga: float = 0.25) -> pd.DataFrame:
        """
        Estratégia dinâmica: a cada dia, identifica as N horas mais
        baratas (carga) e as N horas mais caras (descarga) daquele
        dia específico - mais adaptativa que janela fixa, mas requer
        operação mais sofisticada (previsão de PLD D-1).
        """
        df_sub = df_sub.copy()
        df_sub['sinal'] = 'ocioso'

        for data, grupo in df_sub.groupby('data'):
            n_horas = len(grupo)
            n_carga = max(1, int(n_horas * pct_horas_carga))
            n_descarga = max(1, int(n_horas * pct_horas_descarga))

            idx_ordenado = grupo.sort_values('pld_mwh').index

            idx_carga = idx_ordenado[:n_carga]
            idx_descarga = idx_ordenado[-n_descarga:]

            df_sub.loc[idx_carga, 'sinal'] = 'carga'
            df_sub.loc[idx_descarga, 'sinal'] = 'descarga'

        return df_sub

    def _executar_simulacao(self,
                              df_sub: pd.DataFrame,
                              capacidade_mwh: float,
                              potencia_mw: float,
                              soc_inicial_pct: float,
                              soc_min_pct: float,
                              soc_max_pct: float) -> pd.DataFrame:
        """Motor de simulação hora a hora com restrições físicas do BESS"""
        soc_mwh = capacidade_mwh * soc_inicial_pct
        soc_min_mwh = capacidade_mwh * soc_min_pct
        soc_max_mwh = capacidade_mwh * soc_max_pct

        # Raiz quadrada da eficiência aplicada em cada direção
        # (convenção comum: perdas distribuídas entre carga e descarga)
        eficiencia_unidirecional = np.sqrt(self.eficiencia_bess)

        resultados = []

        for _, row in df_sub.iterrows():
            acao = row['sinal']
            pld = row['pld_mwh']

            energia_mw_hora = 0.0
            custo_receita_rs = 0.0
            soc_antes = soc_mwh

            if acao == 'carga':
                espaco_disponivel = soc_max_mwh - soc_mwh
                energia_carregada_bruta = min(potencia_mw, espaco_disponivel / eficiencia_unidirecional)
                energia_carregada_bruta = max(0.0, energia_carregada_bruta)

                soc_mwh += energia_carregada_bruta * eficiencia_unidirecional
                energia_mw_hora = energia_carregada_bruta
                custo_receita_rs = -(energia_carregada_bruta * pld)  # custo (saída de caixa)

            elif acao == 'descarga':
                energia_disponivel = soc_mwh - soc_min_mwh
                energia_descarregada_liquida = min(
                    potencia_mw, energia_disponivel * eficiencia_unidirecional
                )
                energia_descarregada_liquida = max(0.0, energia_descarregada_liquida)

                soc_mwh -= energia_descarregada_liquida / eficiencia_unidirecional
                energia_mw_hora = -energia_descarregada_liquida
                custo_receita_rs = energia_descarregada_liquida * pld  # receita (entrada de caixa)

            resultados.append({
                'timestamp': row['timestamp'],
                'submercado': row['submercado'],
                'pld_mwh': pld,
                'acao': acao,
                'soc_antes_mwh': round(soc_antes, 3),
                'soc_depois_mwh': round(soc_mwh, 3),
                'soc_pct': round((soc_mwh / capacidade_mwh) * 100, 1),
                'energia_mw': round(energia_mw_hora, 3),
                'fluxo_caixa_rs': round(custo_receita_rs, 2),
            })

        df_resultado = pd.DataFrame(resultados)
        return df_resultado

    # ------------------------------------------------------------------
    # Métricas econômicas consolidadas da simulação
    # ------------------------------------------------------------------
    def calculate_simulation_metrics(self,
                                       df_simulacao: pd.DataFrame,
                                       capacidade_mwh: float) -> Dict:
        """
        Extrai métricas econômicas e operacionais consolidadas a partir
        do resultado da simulação (receita total, ciclos equivalentes,
        receita por MWh de capacidade, etc.)
        """
        if df_simulacao.empty:
            return {}

        receita_total_rs = df_simulacao['fluxo_caixa_rs'].sum()
        custo_total_carga_rs = df_simulacao.loc[
            df_simulacao['acao'] == 'carga', 'fluxo_caixa_rs'
        ].sum()
        receita_total_descarga_rs = df_simulacao.loc[
            df_simulacao['acao'] == 'descarga', 'fluxo_caixa_rs'
        ].sum()

        energia_total_carregada_mwh = df_simulacao.loc[
            df_simulacao['acao'] == 'carga', 'energia_mw'
        ].sum()
        energia_total_descarregada_mwh = abs(df_simulacao.loc[
            df_simulacao['acao'] == 'descarga', 'energia_mw'
        ].sum())

        # Ciclo equivalente = energia descarregada / capacidade nominal
        ciclos_equivalentes = (
            energia_total_descarregada_mwh / capacidade_mwh
            if capacidade_mwh > 0 else 0.0
        )

        periodo_dias = (
            df_simulacao['timestamp'].max() - df_simulacao['timestamp'].min()
        ).days or 1

        ciclos_por_dia = ciclos_equivalentes / periodo_dias
        receita_anualizada_rs = receita_total_rs * (365 / periodo_dias)
        ciclos_anualizados = ciclos_por_dia * 365

        receita_por_mwh_capacidade_rs = (
            receita_anualizada_rs / capacidade_mwh if capacidade_mwh > 0 else 0.0
        )

        spread_medio_capturado_rs_mwh = (
            receita_total_rs / energia_total_descarregada_mwh
            if energia_total_descarregada_mwh > 0 else 0.0
        )

        return {
            'periodo_analisado_dias': periodo_dias,
            'receita_total_periodo_rs': round(receita_total_rs, 2),
            'custo_total_carga_rs': round(custo_total_carga_rs, 2),
            'receita_total_descarga_rs': round(receita_total_descarga_rs, 2),
            'energia_total_carregada_mwh': round(energia_total_carregada_mwh, 2),
            'energia_total_descarregada_mwh': round(energia_total_descarregada_mwh, 2),
            'ciclos_equivalentes_periodo': round(ciclos_equivalentes, 2),
            'ciclos_por_dia_medio': round(ciclos_por_dia, 3),
            'ciclos_equivalentes_anualizados': round(ciclos_anualizados, 1),
            'receita_anualizada_rs': round(receita_anualizada_rs, 2),
            'receita_por_mwh_capacidade_rs_ano': round(receita_por_mwh_capacidade_rs, 2),
            'spread_medio_capturado_rs_mwh': round(spread_medio_capturado_rs_mwh, 2),
            'eficiencia_efetiva_pct': round(
                (energia_total_descarregada_mwh / energia_total_carregada_mwh * 100)
                if energia_total_carregada_mwh > 0 else 0.0, 1
            ),
        }

    # ------------------------------------------------------------------
    # Comparação de estratégias (janela fixa vs. percentil diário)
    # ------------------------------------------------------------------
    def compare_strategies(self,
                             registros: List[RegistroPLD],
                             submercado: str,
                             capacidade_mwh: float,
                             potencia_mw: float) -> pd.DataFrame:
        """
        Executa e compara múltiplas estratégias de operação para o
        mesmo BESS hipotético, permitindo escolher a mais adequada
        antes de investir em sistemas de previsão/otimização mais
        sofisticados (MILP, RL, etc.)
        """
        estrategias = ['janela_fixa', 'percentil_diario']
        resultados_comparacao = []

        for estrategia in estrategias:
            try:
                df_sim = self.simulate_arbitrage_operation(
                    registros, submercado, capacidade_mwh, potencia_mw,
                    estrategia=estrategia
                )
                metricas = self.calculate_simulation_metrics(df_sim, capacidade_mwh)
                metricas['estrategia'] = estrategia
                resultados_comparacao.append(metricas)
            except Exception as e:
                logger.error(f"Erro ao simular estratégia {estrategia}: {e}")
                continue

        if not resultados_comparacao:
            return pd.DataFrame()

        df_comparacao = pd.DataFrame(resultados_comparacao)
        colunas_ordenadas = ['estrategia'] + [
            c for c in df_comparacao.columns if c != 'estrategia'
        ]
        return df_comparacao[colunas_ordenadas].sort_values(
            'receita_anualizada_rs', ascending=False
        ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Análise de sazonalidade do spread (mensal)
    # ------------------------------------------------------------------
    def analyze_seasonal_spread(self, registros: List[RegistroPLD]) -> pd.DataFrame:
        """
        Analisa como o spread de PLD varia ao longo dos meses - útil
        para identificar sazonalidade hidrológica (período seco vs.
        período úmido) que impacta diretamente a atratividade da
        arbitragem ao longo do ano.
        """
        df_spread = self.calculate_daily_spread(registros)
        if df_spread.empty:
            return pd.DataFrame()

        sazonalidade = df_spread.groupby(['submercado', 'mes']).agg(
            spread_medio_rs_mwh=('spread_absoluto', 'mean'),
            spread_std=('spread_absoluto', 'std'),
            pld_medio_mes=('pld_medio', 'mean'),
            pld_min_mes=('pld_min', 'min'),
            pld_max_mes=('pld_max', 'max'),
            dias_analisados=('data', 'count'),
        ).reset_index()

        return sazonalidade.sort_values(['submercado', 'mes']).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Ranking de submercados por atratividade de spread
    # ------------------------------------------------------------------
    def rank_submercados_por_spread(self,
                                       registros: List[RegistroPLD]) -> pd.DataFrame:
        """
        Ranking complementar ao do CurtailmentAnalyzer, focado
        exclusivamente na dinâmica de preços (independente de haver
        ou não curtailment formal registrado).
        """
        df_spread = self.calculate_daily_spread(registros)
        if df_spread.empty:
            return pd.DataFrame()

        ranking = df_spread.groupby('submercado').agg(
            spread_medio_rs_mwh=('spread_absoluto', 'mean'),
            spread_mediano_rs_mwh=('spread_absoluto', 'median'),
            spread_p90_rs_mwh=('spread_absoluto', lambda x: np.percentile(x, 90)),
            spread_relativo_medio_pct=('spread_relativo_pct', 'mean'),
            pld_medio_geral=('pld_medio', 'mean'),
            dias_analisados=('data', 'count'),
            hora_vale_moda=('hora_vale', lambda x: x.mode()[0] if not x.mode().empty else None),
            hora_pico_moda=('hora_pico', lambda x: x.mode()[0] if not x.mode().empty else None),
        ).reset_index()

        # Score simplificado de atratividade (spread absoluto + consistência)
        ranking['consistencia_horario_pct'] = ranking.apply(
            lambda row: self._calcular_consistencia_horario(
                df_spread[df_spread['submercado'] == row['submercado']]
            ), axis=1
        )

        ranking['score_spread'] = (
            self._normalizar(ranking['spread_medio_rs_mwh']) * 0.5 +
            self._normalizar(ranking['consistencia_horario_pct']) * 0.3 +
            self._normalizar(ranking['spread_relativo_medio_pct']) * 0.2
        ) * 100

        return ranking.sort_values('score_spread', ascending=False).reset_index(drop=True)

    @staticmethod
    def _calcular_consistencia_horario(df_spread_submercado: pd.DataFrame) -> float:
        """
        Calcula o % de dias em que a hora de pico/vale está dentro de
        uma janela de ±2h da moda - mede previsibilidade do padrão.
        """
        if df_spread_submercado.empty:
            return 0.0

        moda_vale = df_spread_submercado['hora_vale'].mode()
        moda_pico = df_spread_submercado['hora_pico'].mode()

        if moda_vale.empty or moda_pico.empty:
            return 0.0

        moda_vale = moda_vale[0]
        moda_pico = moda_pico[0]

        dentro_janela_vale = df_spread_submercado['hora_vale'].apply(
            lambda h: abs(h - moda_vale) <= 2
        )
        dentro_janela_pico = df_spread_submercado['hora_pico'].apply(
            lambda h: abs(h - moda_pico) <= 2
        )

        consistencia = (dentro_janela_vale & dentro_janela_pico).mean() * 100
        return round(consistencia, 1)

    @staticmethod
    def _normalizar(serie: pd.Series) -> pd.Series:
        if serie.max() == serie.min():
            return pd.Series([0.5] * len(serie), index=serie.index)
        return (serie - serie.min()) / (serie.max() - serie.min())

    # ------------------------------------------------------------------
    # Relatório executivo consolidado
    # ------------------------------------------------------------------
    def generate_spread_report(self, registros: List[RegistroPLD]) -> Dict:
        """Gera relatório executivo completo de spread por submercado"""
        ranking = self.rank_submercados_por_spread(registros)
        sazonalidade = self.analyze_seasonal_spread(registros)

        janelas_por_submercado = {}
        for submercado in Settings.SUBMERCADOS:
            janela = self.identify_optimal_windows(registros, submercado)
            if janela:
                janelas_por_submercado[submercado] = janela

        return {
            'data_geracao_relatorio': datetime.now().isoformat(),
            'ranking_submercados': ranking.to_dict('records') if not ranking.empty else [],
            'sazonalidade_mensal': sazonalidade.to_dict('records') if not sazonalidade.empty else [],
            'janelas_otimas_por_submercado': janelas_por_submercado,
            'submercado_melhor_spread': (
                ranking.iloc[0]['submercado'] if not ranking.empty else None
            ),
        }