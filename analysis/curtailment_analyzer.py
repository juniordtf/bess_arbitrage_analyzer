# analysis/curtailment_analyzer.py
"""
Analisador de restrições de geração (curtailment) e excessos de geração
por submercado.

Dois fenômenos complementares são mapeados:

1. CURTAILMENT (restrição): geração programada > geração verificada.
   Típico em eólica/solar quando há restrição de transmissão, excesso
   de oferta no horário ou segurança do sistema (COFF/CMSE).

2. EXCESSO DE GERAÇÃO (oversupply): situação em que a oferta agregada
   do submercado supera a demanda a ponto de pressionar PLD para
   valores muito baixos (ou mínimo regulatório) - comum em períodos
   de alta hidrologia + alta geração renovável simultânea.

Ambos os fenômenos geram sinais complementares e essenciais para BESS:
- Curtailment -> energia "perdida" que poderia carregar o BESS a custo ~0
- Excesso de geração -> janelas de preço baixo para carga, seguidas de
  janelas de preço alto para descarga (arbitragem clássica)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import asdict
import logging

from config.settings import Settings
from models.data_models import (
    RegistroGeracao, RegistroPLD, RestricaoOperativa,
    TipoFonte, TipoRestricao, RazaoRestricaoCOFF, RestricaoCOFF
)

logger = logging.getLogger(__name__)


class CurtailmentAnalyzer:
    """
    Motor de análise de curtailment com hierarquia explícita de
    confiabilidade de fonte:

    1ª PRIORIDADE (dado real/confirmado):
       RestricaoCOFF - dataset oficial do ONS com razão de corte
       classificada por usina/hora. Sempre que disponível, é a
       ÚNICA fonte usada para reportar "energia de curtailment
       confirmada".

    2ª PRIORIDADE (inferência física):
       Comparação geração programada vs. verificada (RegistroGeracao)
       - usado quando não há dado COFF disponível para o período/fonte.

    3ª PRIORIDADE (proxy econômico, apenas contextual):
       PLD baixo/zero (RegistroPLD) - NUNCA deve ser reportado como
       "curtailment confirmado". Serve apenas como indicador de
       contexto de excesso de oferta no sistema, sinalizado
       explicitamente como tal em todos os relatórios/gráficos.
    """

    # Threshold mínimo de potência restringida para considerar um
    # evento relevante (evita ruído de registros com cortes triviais)
    MIN_POTENCIA_RESTRINGIDA_MW = 1.0

    def __init__(self,
                 limiar_curtailment_mw: float = None,
                 limiar_percentual: float = 0.05,
                 limiar_excesso_percentil: float = 10.0):
        self.limiar_curtailment_mw = (
            Settings.LIMIAR_CURTAILMENT_MW
            if limiar_curtailment_mw is None
            else limiar_curtailment_mw
        )
        self.limiar_percentual = limiar_percentual
        self.limiar_excesso_percentil = limiar_excesso_percentil

    # ==================================================================
    # MÉTODO PRINCIPAL - ANÁLISE COM FONTE REAL (COFF)
    # ==================================================================
    def analyze_curtailment_real(self,
                                    registros_coff: List[RestricaoCOFF]) -> pd.DataFrame:
        """
        Analisa curtailment REAL a partir do dataset oficial COFF do
        ONS, agregando por submercado, fonte e razão de restrição.

        Este é o método que deve alimentar qualquer relatório que
        afirme "houve X MWh de curtailment" - diferente do proxy de
        PLD, que deve ser sempre rotulado como estimativa indicativa.
        """
        if not registros_coff:
            return pd.DataFrame()

        data = [{
            'timestamp': r.timestamp,
            'submercado': r.submercado,
            'fonte': r.fonte.value,
            'usina': r.usina,
            'razao_restricao': r.razao_restricao.value,
            'potencia_restringida_mw': r.potencia_restringida_mw,
            'energia_cortada_mwh': r.energia_cortada_mwh,
        } for r in registros_coff]

        df = pd.DataFrame(data)

        resumo = df.groupby(['submercado', 'fonte', 'razao_restricao']).agg(
            energia_cortada_total_mwh=('energia_cortada_mwh', 'sum'),
            num_registros=('energia_cortada_mwh', 'count'),
            num_usinas_afetadas=('usina', 'nunique'),
            potencia_media_restringida_mw=('potencia_restringida_mw', 'mean'),
            potencia_maxima_restringida_mw=('potencia_restringida_mw', 'max'),
        ).reset_index()

        resumo['fonte_dado'] = 'COFF_REAL'  # flag explícita de confiabilidade

        return resumo.sort_values('energia_cortada_total_mwh', ascending=False)

    def detect_curtailment_events_real(self,
                                          registros_coff: List[RestricaoCOFF],
                                          min_duracao_horas: float = 0.5) -> List[RestricaoOperativa]:
        """
        Detecta eventos discretos de curtailment a partir do dado REAL
        COFF, consolidando registros horários consecutivos da mesma
        usina/razão em um único evento (mesma lógica de agrupamento
        temporal usada no método legado baseado em programado/verificado,
        mas agora alimentada por dado confirmado).
        """
        if not registros_coff:
            return []

        df = pd.DataFrame([{
            'timestamp': r.timestamp,
            'usina': r.usina,
            'submercado': r.submercado,
            'fonte': r.fonte,
            'potencia_restringida_mw': r.potencia_restringida_mw,
            'razao_restricao': r.razao_restricao,
            'intervalo_horas': r.intervalo_horas,
        } for r in registros_coff])

        df = df[df['potencia_restringida_mw'] >= self.MIN_POTENCIA_RESTRINGIDA_MW]
        if df.empty:
            return []

        df = df.sort_values(['usina', 'timestamp'])

        eventos = []
        for (usina, submercado, fonte, razao), grupo in df.groupby(
            ['usina', 'submercado', 'fonte', 'razao_restricao'],
            sort=False,
        ):
            grupo = grupo.sort_values('timestamp').reset_index(drop=True)
            grupo['gap_horas'] = grupo['timestamp'].diff().dt.total_seconds().div(3600).fillna(0)
            intervalo_anterior = grupo['intervalo_horas'].shift().fillna(
                grupo['intervalo_horas']
            )
            grupo['novo_evento'] = (
                grupo['gap_horas'] > intervalo_anterior * 1.5
            ).cumsum()

            for _, sub_grupo in grupo.groupby('novo_evento'):
                duracao_horas = float(sub_grupo['intervalo_horas'].sum())
                if duracao_horas < min_duracao_horas:
                    continue

                razao = RazaoRestricaoCOFF(razao)
                tipo = self._mapear_razao_para_tipo_restricao(
                    razao, sub_grupo['fonte'].iloc[0]
                )
                energia_cortada_mwh = float(
                    (
                        sub_grupo['potencia_restringida_mw']
                        * sub_grupo['intervalo_horas']
                    ).sum()
                )
                magnitude_media_mw = energia_cortada_mwh / duracao_horas

                eventos.append(RestricaoOperativa(
                    timestamp=sub_grupo['timestamp'].iloc[0],
                    submercado=sub_grupo['submercado'].iloc[0],
                    tipo=tipo,
                    magnitude_mw=magnitude_media_mw,
                    duracao_horas=duracao_horas,
                    descricao=(
                        f"Curtailment REAL (COFF) confirmado na usina {usina} "
                        f"por razão {razao.value}, {duracao_horas:g}h consecutivas"
                    ),
                ))

        logger.info(f"Detectados {len(eventos)} eventos REAIS de curtailment (fonte COFF)")
        return eventos

    @staticmethod
    def _mapear_razao_para_tipo_restricao(razao: RazaoRestricaoCOFF,
                                              fonte: TipoFonte) -> TipoRestricao:
        """Mapeia razão oficial COFF + fonte para o enum de tipo de restrição consolidado"""
        if razao == RazaoRestricaoCOFF.ELETRICA:
            return TipoRestricao.CURTAILMENT_ELETRICO_COFF
        if razao == RazaoRestricaoCOFF.ENERGETICA:
            return TipoRestricao.CURTAILMENT_ENERGETICO_COFF
        if fonte == TipoFonte.EOLICA:
            return TipoRestricao.CURTAILMENT_EOLICO_COFF
        if fonte == TipoFonte.SOLAR:
            return TipoRestricao.CURTAILMENT_SOLAR_COFF
        return TipoRestricao.RESTRICAO_TRANSMISSAO

    # ==================================================================
    # MÉTODO LEGADO - PROXY VIA PLD (APENAS CONTEXTUAL, NUNCA PRIMÁRIO)
    # ==================================================================
    def detect_oversupply_events(self,
                                    registros_pld: List[RegistroPLD],
                                    percentil_threshold: float = 0.10,
                                    min_duracao_horas: int = 1) -> List[RestricaoOperativa]:
        """
        ⚠️ MÉTODO DE PROXY ECONÔMICO - NÃO CONFUNDIR COM DADO CONFIRMADO.

        Detecta janelas de PLD anormalmente baixo como um INDICADOR
        INDIRETO de possível excesso de oferta no sistema. Este método
        NUNCA deve ser usado isoladamente para afirmar "houve
        curtailment" em relatórios - ele serve apenas como sinal de
        contexto econômico complementar ao dado real (COFF).

        Motivo técnico: o PLD no Brasil raramente reflete eventos de
        curtailment por restrição ELÉTRICA (a maioria dos casos reais
        no NE/SE), pois é um preço de equilíbrio nacional/submercado,
        não um indicador de restrição local de rede. Mesmo em casos de
        excesso ENERGÉTICO real, o PLD tem piso regulatório que pode
        mascarar o evento.

        Todos os eventos retornados por este método carregam o tipo
        TipoRestricao.EXCESSO_OFERTA_PROXY_PLD, que é tratado de forma
        visualmente distinta (e com ressalva textual) em todas as
        camadas de relatório/dashboard.
        """
        if not registros_pld:
            return []

        df = pd.DataFrame([{
            'timestamp': r.timestamp,
            'submercado': r.submercado,
            'pld_mwh': r.pld_mwh,
        } for r in registros_pld])

        eventos = []

        for submercado, grupo in df.groupby('submercado'):
            grupo = grupo.sort_values('timestamp').reset_index(drop=True)
            threshold = grupo['pld_mwh'].quantile(percentil_threshold)

            grupo['abaixo_threshold'] = grupo['pld_mwh'] <= threshold
            grupo['gap_horas'] = grupo['timestamp'].diff().dt.total_seconds().div(3600).fillna(0)
            grupo['novo_evento'] = (
                (~grupo['abaixo_threshold']) | (grupo['gap_horas'] > 1.5)
            ).cumsum()

            for _, sub_grupo in grupo[grupo['abaixo_threshold']].groupby('novo_evento'):
                duracao_horas = len(sub_grupo)
                if duracao_horas < min_duracao_horas:
                    continue

                eventos.append(RestricaoOperativa(
                    timestamp=sub_grupo['timestamp'].iloc[0],
                    submercado=submercado,
                    tipo=TipoRestricao.EXCESSO_OFERTA_PROXY_PLD,
                    magnitude_mw=0.0,  # proxy de preço não tem magnitude física confiável
                    duracao_horas=duracao_horas,
                    descricao=(
                        f"⚠️ INDICATIVO (proxy PLD): PLD médio de "
                        f"R$ {sub_grupo['pld_mwh'].mean():.2f}/MWh "
                        f"(abaixo do percentil {percentil_threshold*100:.0f}% do período) "
                        f"por {duracao_horas}h - possível excesso de oferta, "
                        f"SEM confirmação por dado real de restrição."
                    ),
                ))

        logger.info(
            f"Detectados {len(eventos)} eventos INDICATIVOS via proxy PLD "
            f"(⚠️ não confirmados por dado real COFF)"
        )
        return eventos

    # ==================================================================
    # CONSOLIDAÇÃO: FUNDE DADO REAL + PROXY, COM RÓTULO DE CONFIABILIDADE
    # ==================================================================
    def analyze_curtailment_consolidado(self,
                                           registros_coff_eolica: Optional[List[RestricaoCOFF]] = None,
                                           registros_coff_solar: Optional[List[RestricaoCOFF]] = None,
                                           registros_geracao: Optional[List[RegistroGeracao]] = None,
                                           registros_pld: Optional[List[RegistroPLD]] = None) -> Dict:
        """
        Método orquestrador que aplica a hierarquia de confiabilidade
        de fonte e retorna um resultado consolidado, SEMPRE explicitando
        qual fração do resultado vem de dado confirmado vs. indicativo.

        Hierarquia de uso:
        1. Se há registros_coff (eólica e/ou solar) -> usa como fonte
           PRIMÁRIA e CONFIRMADA de curtailment.
        2. Se não há COFF mas há registros_geracao (programado vs.
           verificado) -> usa como fonte SECUNDÁRIA (inferência física,
           ainda assim mais confiável que preço).
        3. registros_pld é SEMPRE processado, mas seus eventos são
           marcados como indicativos e nunca somados ao total
           "confirmado" de energia cortada.

        Returns:
            Dict com:
            - 'eventos_confirmados': List[RestricaoOperativa] (fonte COFF ou geração)
            - 'eventos_indicativos_pld': List[RestricaoOperativa] (proxy, sempre separado)
            - 'fonte_dados_primaria': str ('COFF_REAL' | 'GERACAO_PROGRAMADA_VS_VERIFICADA' | 'NENHUMA')
            - 'energia_confirmada_total_mwh': float
            - 'tem_dado_confirmado': bool
            - 'resumo_por_razao': pd.DataFrame (apenas se fonte COFF disponível)
        """
        registros_coff = (registros_coff_eolica or []) + (registros_coff_solar or [])

        eventos_confirmados: List[RestricaoOperativa] = []
        fonte_primaria = 'NENHUMA'
        resumo_por_razao = pd.DataFrame()

        if registros_coff:
            eventos_confirmados = self.detect_curtailment_events_real(registros_coff)
            resumo_por_razao = self.analyze_curtailment_real(registros_coff)
            fonte_primaria = 'COFF_REAL'

        elif registros_geracao:
            eventos_inferidos = self.detect_curtailment_events(registros_geracao)
            if eventos_inferidos:
                eventos_confirmados = eventos_inferidos
                fonte_primaria = 'GERACAO_PROGRAMADA_VS_VERIFICADA'

        if not registros_coff and fonte_primaria == 'GERACAO_PROGRAMADA_VS_VERIFICADA':
            logger.warning(
                "Dataset COFF real não disponível - usando inferência via "
                "geração programada vs. verificada (fonte secundária, "
                "menos granular que o dado oficial de razão de corte)"
            )
        elif not registros_coff and fonte_primaria == 'NENHUMA':
            logger.warning(
                "⚠️ NENHUMA fonte confirmada de curtailment disponível "
                "(nem COFF, nem geração programada/verificada). A análise "
                "ficará restrita a indicadores de PLD - resultado NÃO deve "
                "ser reportado como 'ausência de curtailment confirmado', "
                "mas sim como 'dado confirmado indisponível para o período'."
            )

        eventos_indicativos_pld = []
        if registros_pld:
            eventos_indicativos_pld = self.detect_oversupply_events(registros_pld)

        energia_confirmada_total = sum(
            e.magnitude_mw * e.duracao_horas for e in eventos_confirmados
        )

        return {
            'eventos_confirmados': eventos_confirmados,
            'eventos_indicativos_pld': eventos_indicativos_pld,
            'fonte_dados_primaria': fonte_primaria,
            'energia_confirmada_total_mwh': round(energia_confirmada_total, 1),
            'tem_dado_confirmado': fonte_primaria != 'NENHUMA',
            'resumo_por_razao': resumo_por_razao,
        }

    def gerar_aviso_transparencia(self, resultado_consolidado: Dict) -> str:
        """
        Gera texto de aviso padronizado para exibição em relatórios/
        dashboard, deixando explícita a origem e confiabilidade dos
        dados apresentados - requisito direto levantado na revisão
        deste módulo (evitar "estranheza" por falsos negativos).
        """
        fonte = resultado_consolidado['fonte_dados_primaria']
        tem_confirmado = resultado_consolidado['tem_dado_confirmado']
        n_indicativos = len(resultado_consolidado['eventos_indicativos_pld'])

        if fonte == 'COFF_REAL':
            return (
                "✅ Dados de curtailment baseados no dataset OFICIAL de "
                "Restrição COFF do ONS (granularidade de 30 minutos por "
                "usina). A potência restringida é estimada pela diferença "
                "entre geração estimada e verificada, e a razão é exibida "
                "quando publicada pelo ONS."
            )
        elif fonte == 'GERACAO_PROGRAMADA_VS_VERIFICADA':
            return (
                "⚠️ Dataset oficial de Restrição COFF indisponível para o "
                "período. Curtailment estimado por inferência física "
                "(geração programada vs. verificada) - método mais "
                "confiável que proxy de preço, porém sem classificação "
                "oficial de razão de corte."
            )
        else:
            aviso = (
                "❌ ATENÇÃO: nenhuma fonte confirmada de curtailment "
                "(COFF ou geração programada/verificada) estava disponível "
                "para o período analisado. A ausência de eventos "
                "confirmados NÃO significa ausência de curtailment real - "
                "significa apenas que o dado não pôde ser obtido."
            )
            if n_indicativos > 0:
                aviso += (
                    f" Foram identificados {n_indicativos} eventos "
                    f"INDICATIVOS via proxy de PLD baixo, que sugerem "
                    f"possível excesso de oferta, mas que requerem "
                    f"validação com dado oficial antes de qualquer "
                    f"conclusão definitiva."
                )
            return aviso

    # ------------------------------------------------------------------
    # Construção do DataFrame base
    # ------------------------------------------------------------------
    def _registros_to_df(self, registros: List[RegistroGeracao]) -> pd.DataFrame:
        """Converte lista de RegistroGeracao em DataFrame analítico"""
        if not registros:
            return pd.DataFrame()

        data = []
        for r in registros:
            data.append({
                'timestamp': r.timestamp,
                'submercado': r.submercado,
                'fonte': r.fonte.value,
                'geracao_programada_mw': r.geracao_programada_mw,
                'geracao_verificada_mw': r.geracao_verificada_mw,
                'capacidade_instalada_mw': r.capacidade_instalada_mw,
                'curtailment_mw': r.curtailment_mw,
                'fator_capacidade': r.fator_capacidade,
            })

        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hora'] = df['timestamp'].dt.hour
        df['mes'] = df['timestamp'].dt.month
        df['dia_semana'] = df['timestamp'].dt.dayofweek
        df['data'] = df['timestamp'].dt.date

        # % de curtailment em relação ao programado
        df['curtailment_pct'] = np.where(
            df['geracao_programada_mw'] > 0,
            df['curtailment_mw'] / df['geracao_programada_mw'],
            0.0
        )

        return df

    # ------------------------------------------------------------------
    # Análise principal de curtailment
    # ------------------------------------------------------------------
    def analyze_curtailment_by_source(self,
                                       registros: List[RegistroGeracao]) -> pd.DataFrame:
        """
        Agrega curtailment por submercado e fonte, retornando métricas
        de severidade, frequência e padrão temporal.
        """
        df = self._registros_to_df(registros)
        if df.empty:
            logger.warning("Nenhum registro de geração para analisar")
            return pd.DataFrame()

        # Marca eventos relevantes (acima dos limiares)
        df['evento_relevante'] = (
            (df['curtailment_mw'] >= self.limiar_curtailment_mw) &
            (df['curtailment_pct'] >= self.limiar_percentual)
        )

        agregado = df.groupby(['submercado', 'fonte']).apply(
            self._calcular_metricas_grupo, include_groups=False
        ).reset_index()

        return agregado

    def _calcular_metricas_grupo(self, grupo: pd.DataFrame) -> pd.Series:
        """Calcula métricas de severidade para um grupo (submercado x fonte)"""
        total_horas = len(grupo)
        horas_com_evento = grupo['evento_relevante'].sum()

        energia_cortada_total_mwh = grupo.loc[
            grupo['evento_relevante'], 'curtailment_mw'
        ].sum()  # assumindo granularidade horária -> MW == MWh

        energia_gerada_total_mwh = grupo['geracao_verificada_mw'].sum()
        energia_programada_total_mwh = grupo['geracao_programada_mw'].sum()

        curtailment_medio_mw = grupo.loc[
            grupo['evento_relevante'], 'curtailment_mw'
        ].mean() if horas_com_evento > 0 else 0.0

        curtailment_maximo_mw = grupo['curtailment_mw'].max()

        fator_capacidade_medio = grupo['fator_capacidade'].mean()

        pct_tempo_com_curtailment = (
            horas_com_evento / total_horas if total_horas > 0 else 0.0
        )

        pct_energia_perdida = (
            energia_cortada_total_mwh / energia_programada_total_mwh
            if energia_programada_total_mwh > 0 else 0.0
        )

        return pd.Series({
            'total_horas_analisadas': total_horas,
            'horas_com_curtailment': int(horas_com_evento),
            'pct_tempo_com_curtailment': pct_tempo_com_curtailment,
            'energia_cortada_total_mwh': energia_cortada_total_mwh,
            'energia_gerada_total_mwh': energia_gerada_total_mwh,
            'pct_energia_perdida': pct_energia_perdida,
            'curtailment_medio_evento_mw': curtailment_medio_mw,
            'curtailment_maximo_mw': curtailment_maximo_mw,
            'fator_capacidade_medio': fator_capacidade_medio,
            'capacidade_instalada_mw': grupo['capacidade_instalada_mw'].iloc[0]
                if len(grupo) > 0 else 0.0,
        })

    # ------------------------------------------------------------------
    # Detecção de eventos discretos (para timeline/BESS sizing)
    # ------------------------------------------------------------------
    def detect_curtailment_events(self,
                                    registros: List[RegistroGeracao],
                                    min_duracao_horas: int = 1) -> List[RestricaoOperativa]:
        """
        Detecta eventos discretos de curtailment (janelas contínuas),
        consolidando horas consecutivas em um único evento.

        Essencial para dimensionar BESS: um evento de 4h consecutivas
        de curtailment de 100MW representa 400MWh de energia potencial
        para absorção - isso define capacidade mínima recomendada.
        """
        df = self._registros_to_df(registros)
        if df.empty:
            return []

        df = df.sort_values(['submercado', 'fonte', 'timestamp'])

        df['evento_relevante'] = (
            (df['curtailment_mw'] >= self.limiar_curtailment_mw) &
            (df['curtailment_pct'] >= self.limiar_percentual)
        )

        eventos = []

        for (submercado, fonte), grupo in df.groupby(['submercado', 'fonte']):
            eventos.extend(
                self._extrair_eventos_continuos(
                    grupo, submercado, fonte, min_duracao_horas
                )
            )

        logger.info(f"Detectados {len(eventos)} eventos de curtailment")
        return eventos

    def _extrair_eventos_continuos(self,
                                     grupo: pd.DataFrame,
                                     submercado: str,
                                     fonte: str,
                                     min_duracao_horas: int) -> List[RestricaoOperativa]:
        """Identifica blocos contínuos de curtailment em uma série temporal"""
        grupo = grupo.reset_index(drop=True)

        # Identifica mudanças de estado (início/fim de evento)
        grupo['bloco'] = (
            grupo['evento_relevante'] != grupo['evento_relevante'].shift()
        ).cumsum()

        eventos = []
        tipo_restricao = self._classificar_tipo_restricao(fonte)

        # analysis/curtailment_analyzer.py (continuação)

        for bloco_id, bloco_df in grupo.groupby('bloco'):
            if not bloco_df['evento_relevante'].iloc[0]:
                continue

            duracao_horas = len(bloco_df)
            if duracao_horas < min_duracao_horas:
                continue

            timestamp_inicio = bloco_df['timestamp'].iloc[0]
            magnitude_media = bloco_df['curtailment_mw'].mean()
            magnitude_total_mwh = bloco_df['curtailment_mw'].sum()

            evento = RestricaoOperativa(
                timestamp=timestamp_inicio,
                submercado=submercado,
                tipo=tipo_restricao,
                magnitude_mw=magnitude_media,
                duracao_horas=duracao_horas,
                descricao=(
                    f"Curtailment {fonte} em {submercado}: "
                    f"{magnitude_total_mwh:.1f} MWh perdidos em "
                    f"{duracao_horas}h ({timestamp_inicio.strftime('%Y-%m-%d %H:%M')})"
                )
            )
            eventos.append(evento)

        return eventos

    @staticmethod
    def _classificar_tipo_restricao(fonte: str) -> TipoRestricao:
        """Classifica eventos inferidos sem marcá-los como dados COFF confirmados."""
        return TipoRestricao.RESTRICAO_TRANSMISSAO

    # ------------------------------------------------------------------
    # Padrões temporais (sazonalidade, horário do dia)
    # ------------------------------------------------------------------
    def analyze_temporal_patterns(self,
                                    registros: List[RegistroGeracao]) -> Dict[str, pd.DataFrame]:
        """
        Identifica padrões de curtailment por hora do dia, dia da semana
        e mês - essencial para entender a previsibilidade do fenômeno e
        dimensionar ciclos diários de carga/descarga do BESS.
        """
        df = self._registros_to_df(registros)
        if df.empty:
            return {}

        df['evento_relevante'] = (
            (df['curtailment_mw'] >= self.limiar_curtailment_mw) &
            (df['curtailment_pct'] >= self.limiar_percentual)
        )

        padrao_horario = df.groupby(['submercado', 'fonte', 'hora']).agg(
            curtailment_medio_mw=('curtailment_mw', 'mean'),
            pct_ocorrencia=('evento_relevante', 'mean'),
            energia_cortada_total_mwh=('curtailment_mw', lambda x: x[df.loc[x.index, 'evento_relevante']].sum())
        ).reset_index()

        padrao_mensal = df.groupby(['submercado', 'fonte', 'mes']).agg(
            curtailment_medio_mw=('curtailment_mw', 'mean'),
            pct_ocorrencia=('evento_relevante', 'mean'),
            energia_cortada_total_mwh=('curtailment_mw', lambda x: x[df.loc[x.index, 'evento_relevante']].sum())
        ).reset_index()

        padrao_dia_semana = df.groupby(['submercado', 'fonte', 'dia_semana']).agg(
            curtailment_medio_mw=('curtailment_mw', 'mean'),
            pct_ocorrencia=('evento_relevante', 'mean')
        ).reset_index()

        return {
            'por_hora': padrao_horario,
            'por_mes': padrao_mensal,
            'por_dia_semana': padrao_dia_semana,
        }

    def identify_recurring_window(self,
                                    registros: List[RegistroGeracao],
                                    submercado: str,
                                    fonte: TipoFonte,
                                    limiar_pct_ocorrencia: float = 0.3) -> Optional[Tuple[int, int]]:
        """
        Identifica a janela horária (hora_inicio, hora_fim) em que o
        curtailment ocorre com maior frequência/recorrência para uma
        combinação submercado+fonte específica.

        Essa janela define o período ideal para "carga" do BESS
        (absorver energia que seria cortada).
        """
        padroes = self.analyze_temporal_patterns(registros)
        df_horario = padroes.get('por_hora', pd.DataFrame())

        if df_horario.empty:
            return None

        df_filtrado = df_horario[
            (df_horario['submercado'] == submercado) &
            (df_horario['fonte'] == fonte.value) &
            (df_horario['pct_ocorrencia'] >= limiar_pct_ocorrencia)
        ].sort_values('hora')

        if df_filtrado.empty:
            return None

        horas = df_filtrado['hora'].tolist()

        # Identifica maior sequência contínua de horas
        melhor_inicio, melhor_fim, melhor_tamanho = horas[0], horas[0], 1
        inicio_atual, tamanho_atual = horas[0], 1

        for i in range(1, len(horas)):
            if horas[i] == horas[i-1] + 1:
                tamanho_atual += 1
            else:
                if tamanho_atual > melhor_tamanho:
                    melhor_tamanho = tamanho_atual
                    melhor_inicio = inicio_atual
                    melhor_fim = horas[i-1]
                inicio_atual = horas[i]
                tamanho_atual = 1

        if tamanho_atual > melhor_tamanho:
            melhor_tamanho = tamanho_atual
            melhor_inicio = inicio_atual
            melhor_fim = horas[-1]

        return (melhor_inicio, melhor_fim)

    # ------------------------------------------------------------------
    # Excesso de geração via análise de PLD
    # ------------------------------------------------------------------
    def analyze_oversupply_via_pld(self,
                                     registros_pld: List[RegistroPLD]) -> pd.DataFrame:
        """
        Identifica janelas de "excesso de geração" através de PLD muito
        baixo - mesmo sem dados explícitos de curtailment, PLD baixo é
        um proxy direto de oferta > demanda no submercado.

        Isso é crucial porque nem todo excesso de oferta gera
        curtailment reportado (às vezes é absorvido via PLD mínimo
        sem registro formal de corte).
        """
        if not registros_pld:
            return pd.DataFrame()

        data = [{
            'timestamp': r.timestamp,
            'submercado': r.submercado,
            'pld_mwh': r.pld_mwh,
            'carga_mw': r.carga_mw,
        } for r in registros_pld]

        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hora'] = df['timestamp'].dt.hour
        df['data'] = df['timestamp'].dt.date

        resultados = []

        for submercado, grupo in df.groupby('submercado'):
            percentil_baixo = np.percentile(
                grupo['pld_mwh'], self.limiar_excesso_percentil
            )
            percentil_alto = np.percentile(
                grupo['pld_mwh'], 100 - self.limiar_excesso_percentil
            )

            grupo = grupo.copy()
            grupo['excesso_geracao'] = grupo['pld_mwh'] <= percentil_baixo
            grupo['escassez_geracao'] = grupo['pld_mwh'] >= percentil_alto

            # Agrega por hora do dia para identificar padrão recorrente
            padrao_horario = grupo.groupby('hora').agg(
                pld_medio=('pld_mwh', 'mean'),
                pct_tempo_excesso=('excesso_geracao', 'mean'),
                pct_tempo_escassez=('escassez_geracao', 'mean'),
            ).reset_index()

            padrao_horario['submercado'] = submercado
            padrao_horario['percentil_baixo_referencia'] = percentil_baixo
            padrao_horario['percentil_alto_referencia'] = percentil_alto

            resultados.append(padrao_horario)

        if not resultados:
            return pd.DataFrame()

        return pd.concat(resultados, ignore_index=True)

    # ------------------------------------------------------------------
    # Ranking consolidado por submercado
    # ------------------------------------------------------------------
    def rank_submercados_por_potencial(self,
                                         registros_geracao: List[RegistroGeracao],
                                         registros_pld: List[RegistroPLD]) -> pd.DataFrame:
        """
        Consolida métricas de curtailment e excesso de geração (via PLD)
        em um ranking único por submercado, indicando onde há maior
        potencial para instalação de BESS com foco em arbitragem.

        O score combina:
        - Volume de energia cortada/perdida (oportunidade de carga "grátis")
        - Frequência/recorrência do fenômeno (previsibilidade = menor risco)
        - Volatilidade de PLD (amplitude do spread de arbitragem)
        """
        df_curtailment = self.analyze_curtailment_by_source(registros_geracao)
        df_oversupply = self.analyze_oversupply_via_pld(registros_pld)

        # --- Consolida curtailment por submercado (soma todas as fontes) ---
        if not df_curtailment.empty:
            curtailment_por_submercado = df_curtailment.groupby('submercado').agg(
                energia_cortada_total_mwh=('energia_cortada_total_mwh', 'sum'),
                pct_tempo_com_curtailment_medio=('pct_tempo_com_curtailment', 'mean'),
                capacidade_instalada_total_mw=('capacidade_instalada_mw', 'sum'),
            ).reset_index()
        else:
            curtailment_por_submercado = pd.DataFrame(
                columns=['submercado', 'energia_cortada_total_mwh',
                         'pct_tempo_com_curtailment_medio',
                         'capacidade_instalada_total_mw']
            )

        # --- Consolida excesso de geração (via PLD) por submercado ---
        if not df_oversupply.empty:
            oversupply_por_submercado = df_oversupply.groupby('submercado').agg(
                pct_tempo_excesso_medio=('pct_tempo_excesso', 'mean'),
                pld_medio_geral=('pld_medio', 'mean'),
            ).reset_index()
        else:
            oversupply_por_submercado = pd.DataFrame(
                columns=['submercado', 'pct_tempo_excesso_medio', 'pld_medio_geral']
            )

        # --- Calcula volatilidade de PLD (desvio padrão) por submercado ---
        volatilidade_pld = self._calcular_volatilidade_pld(registros_pld)

        # --- Merge de todas as dimensões ---
        ranking = pd.DataFrame({'submercado': Settings.SUBMERCADOS})

        ranking = ranking.merge(curtailment_por_submercado, on='submercado', how='left')
        ranking = ranking.merge(oversupply_por_submercado, on='submercado', how='left')
        ranking = ranking.merge(volatilidade_pld, on='submercado', how='left')

        ranking = ranking.fillna({
            'energia_cortada_total_mwh': 0.0,
            'pct_tempo_com_curtailment_medio': 0.0,
            'capacidade_instalada_total_mw': 0.0,
            'pct_tempo_excesso_medio': 0.0,
            'pld_medio_geral': 0.0,
            'volatilidade_pld': 0.0,
            'spread_medio_pld': 0.0,
        })

        ranking['score_potencial_bess'] = self._calcular_score_potencial(ranking)

        ranking = ranking.sort_values('score_potencial_bess', ascending=False).reset_index(drop=True)
        ranking['ranking_posicao'] = ranking.index + 1

        return ranking

    def _calcular_volatilidade_pld(self,
                                     registros_pld: List[RegistroPLD]) -> pd.DataFrame:
        """Calcula desvio padrão e spread (max-min) do PLD por submercado"""
        if not registros_pld:
            return pd.DataFrame(columns=['submercado', 'volatilidade_pld', 'spread_medio_pld'])

        data = [{
            'submercado': r.submercado,
            'pld_mwh': r.pld_mwh,
            'data': r.timestamp.date(),
        } for r in registros_pld]

        df = pd.DataFrame(data)

        # Volatilidade geral (desvio padrão do PLD no período)
        volatilidade = df.groupby('submercado')['pld_mwh'].std().reset_index()
        volatilidade.columns = ['submercado', 'volatilidade_pld']

        # Spread diário médio (max - min por dia, depois média dos dias)
        spread_diario = df.groupby(['submercado', 'data'])['pld_mwh'].agg(
            lambda x: x.max() - x.min()
        ).reset_index()
        spread_diario.columns = ['submercado', 'data', 'spread_dia']

        spread_medio = spread_diario.groupby('submercado')['spread_dia'].mean().reset_index()
        spread_medio.columns = ['submercado', 'spread_medio_pld']

        resultado = volatilidade.merge(spread_medio, on='submercado', how='outer')
        return resultado.fillna(0.0)

    @staticmethod
    def _calcular_score_potencial(ranking: pd.DataFrame) -> pd.Series:
        """
        Calcula score composto de 0-100 para priorização de submercados.

        Pesos (ajustáveis conforme estratégia):
        - 35%: energia cortada normalizada (oportunidade de carga gratuita)
        - 25%: spread médio de PLD (potencial de arbitragem clássica)
        - 20%: volatilidade de PLD (amplitude de oportunidades)
        - 20%: % tempo em excesso de geração (recorrência/previsibilidade)
        """
        def normalizar(serie: pd.Series) -> pd.Series:
            if serie.max() == serie.min():
                return pd.Series([0.5] * len(serie), index=serie.index)
            return (serie - serie.min()) / (serie.max() - serie.min())

        energia_norm = normalizar(ranking['energia_cortada_total_mwh'])
        spread_norm = normalizar(ranking['spread_medio_pld'])
        volatilidade_norm = normalizar(ranking['volatilidade_pld'])
        excesso_norm = normalizar(ranking['pct_tempo_excesso_medio'])

        score = (
            0.35 * energia_norm +
            0.25 * spread_norm +
            0.20 * volatilidade_norm +
            0.20 * excesso_norm
        ) * 100

        return score.round(2)

    # ------------------------------------------------------------------
    # Exportação de resumo executivo
    # ------------------------------------------------------------------
    def generate_summary_report(self,
                                  registros_geracao: List[RegistroGeracao],
                                  registros_pld: List[RegistroPLD]) -> Dict:
        """
        Gera relatório resumido consolidando todas as análises,
        pronto para serialização (JSON/dashboard).
        """
        ranking = self.rank_submercados_por_potencial(registros_geracao, registros_pld)
        eventos_curtailment = self.detect_curtailment_events(registros_geracao)
        eventos_oversupply = self.detect_oversupply_events(registros_pld)
        padroes_temporais = self.analyze_temporal_patterns(registros_geracao)

        # Top eventos por magnitude (maior oportunidade individual)
        top_eventos_curtailment = sorted(
            eventos_curtailment,
            key=lambda e: e.magnitude_mw * e.duracao_horas,
            reverse=True
        )[:10]

        resumo = {
            'data_geracao_relatorio': datetime.now().isoformat(),
            'periodo_analisado': {
                'inicio': min([r.timestamp for r in registros_geracao], default=None),
                'fim': max([r.timestamp for r in registros_geracao], default=None),
            },
            'ranking_submercados': ranking.to_dict('records'),
            'total_eventos_curtailment': len(eventos_curtailment),
            'total_eventos_oversupply': len(eventos_oversupply),
            'top_10_eventos_curtailment': [
                {
                    'submercado': e.submercado,
                    'tipo': e.tipo.value,
                    'timestamp_inicio': e.timestamp.isoformat(),
                    'magnitude_mw': round(e.magnitude_mw, 2),
                    'duracao_horas': e.duracao_horas,
                    'energia_estimada_mwh': round(e.magnitude_mw * e.duracao_horas, 2),
                    'descricao': e.descricao,
                }
                for e in top_eventos_curtailment
            ],
            'submercado_destaque': ranking.iloc[0]['submercado'] if not ranking.empty else None,
        }

        return resumo