# analysis/bess_opportunity_finder.py
"""
Consolidador de oportunidades de BESS para arbitragem de energia.

Este módulo é o ponto de convergência de toda a análise:

1. CurtailmentAnalyzer -> identifica ENERGIA disponível para carga
   "gratuita" (ou quase) via cortes de geração renovável/excesso hídrico
2. PriceSpreadAnalyzer -> identifica o SPREAD de preço (PLD) que
   remunera a descarga dessa energia armazenada
3. ANEELDataCollector (perfil por submercado) -> contextualiza
   saturação de mercado (quantos projetos de BESS já outorgados)

O resultado final é uma lista de objetos `OportunidadeBESS`, já com:
- Dimensionamento técnico recomendado (MWh / MW)
- Economia estimada (CAPEX, receita anual, payback)
- Score de atratividade para priorização de investimento
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import logging
import json
from pathlib import Path

from config.settings import Settings
from models.data_models import (
    RegistroGeracao, RegistroPLD, RestricaoCOFF, RestricaoOperativa,
    OportunidadeBESS, TipoRestricao
)
from analysis.curtailment_analyzer import CurtailmentAnalyzer
from analysis.price_spread_analyzer import PriceSpreadAnalyzer

logger = logging.getLogger(__name__)


class BESSOpportunityFinder:
    """
    Orquestra a análise completa de curtailment + spread + contexto
    de mercado para gerar oportunidades concretas de instalação de BESS.
    """

    # Premissas econômicas de referência (mercado BESS utility-scale, 2024)
    # Podem/devem ser sobrescritas via __init__ com dados atualizados.
    CAPEX_RS_POR_MWH_PADRAO = 1_800_000.0   # ~R$ 1,8MM/MWh (incl. BOP/EPC)
    CAPEX_RS_POR_MW_PADRAO = 450_000.0      # Parcela ligada à potência (PCS/inversores)
    OPEX_PCT_CAPEX_ANO_PADRAO = 0.02        # 2% do CAPEX/ano em O&M
    VIDA_UTIL_ANOS_PADRAO = 15
    TAXA_DESCONTO_PADRAO = 0.10             # WACC real aproximado

    def __init__(self,
                 curtailment_analyzer: Optional[CurtailmentAnalyzer] = None,
                 spread_analyzer: Optional[PriceSpreadAnalyzer] = None,
                 capex_rs_por_mwh: float = None,
                 capex_rs_por_mw: float = None,
                 opex_pct_capex_ano: float = None,
                 vida_util_anos: int = None,
                 taxa_desconto: float = None):

        self.curtailment_analyzer = curtailment_analyzer or CurtailmentAnalyzer()
        self.spread_analyzer = spread_analyzer or PriceSpreadAnalyzer()

        self.capex_rs_por_mwh = capex_rs_por_mwh or self.CAPEX_RS_POR_MWH_PADRAO
        self.capex_rs_por_mw = capex_rs_por_mw or self.CAPEX_RS_POR_MW_PADRAO
        self.opex_pct_capex_ano = opex_pct_capex_ano or self.OPEX_PCT_CAPEX_ANO_PADRAO
        self.vida_util_anos = vida_util_anos or self.VIDA_UTIL_ANOS_PADRAO
        self.taxa_desconto = taxa_desconto or self.TAXA_DESCONTO_PADRAO

    # ==================================================================
    # MÉTODO PRINCIPAL
    # ==================================================================
    def find_opportunities(self,
                              registros_geracao: List[RegistroGeracao],
                              registros_pld: List[RegistroPLD],
                              registros_coff_eolica: Optional[List[RestricaoCOFF]] = None,
                              registros_coff_solar: Optional[List[RestricaoCOFF]] = None,
                              df_perfil_aneel: Optional[pd.DataFrame] = None,
                              min_score_corte: float = 0.0) -> List[OportunidadeBESS]:
        """
        VERSÃO ATUALIZADA: agora recebe (opcionalmente) os dados reais
        de restrição COFF, priorizando-os sobre a inferência via
        geração programada/verificada e sobre o proxy de PLD.

        Caso registros_coff_* não sejam fornecidos, o comportamento
        legado é mantido (fallback para geração programada/verificada,
        com PLD apenas como contexto) - garantindo compatibilidade
        retroativa com chamadas existentes do pipeline/dashboard.

        Args:
            registros_geracao: dados de geração programada/verificada (fallback)
            registros_pld: dados de PLD (sempre usado para precificação do spread,
                            e como sinal indicativo complementar de curtailment)
            registros_coff_eolica: dados REAIS de restrição COFF para eólica (prioritário)
            registros_coff_solar: dados REAIS de restrição COFF para solar (prioritário)
            df_perfil_aneel: perfil de saturação de mercado (opcional)
            min_score_corte: score mínimo para incluir a oportunidade no resultado final
        """
        resultado_curtailment = self.curtailment_analyzer.analyze_curtailment_consolidado(
            registros_coff_eolica=registros_coff_eolica,
            registros_coff_solar=registros_coff_solar,
            registros_geracao=registros_geracao,
            registros_pld=registros_pld,
        )

        aviso_transparencia = self.curtailment_analyzer.gerar_aviso_transparencia(
            resultado_curtailment
        )
        logger.info(f"Fonte de dados de curtailment: {aviso_transparencia}")

        if not resultado_curtailment['tem_dado_confirmado']:
            logger.warning(
                "Nenhuma oportunidade será dimensionada com base em energia "
                "'confirmada' - apenas eventos indicativos (proxy PLD) estão "
                "disponíveis. As oportunidades geradas, se houver, carregarão "
                "a flag 'dados_confirmados=False' para transparência."
            )

        submercados_disponiveis = set(r.submercado for r in registros_pld)
        oportunidades = []

        for submercado in sorted(submercados_disponiveis):
            oportunidade = self._analisar_submercado(
                submercado=submercado,
                resultado_curtailment=resultado_curtailment,
                registros_pld=registros_pld,
                df_perfil_aneel=df_perfil_aneel,
            )

            if oportunidade and oportunidade.score_atratividade >= min_score_corte:
                oportunidades.append(oportunidade)

        oportunidades.sort(key=lambda o: o.score_atratividade, reverse=True)

        logger.info(
            f"{len(oportunidades)} oportunidades identificadas "
            f"(fonte primária de curtailment: {resultado_curtailment['fonte_dados_primaria']})"
        )

        return oportunidades

    def _analisar_submercado(self,
                                submercado: str,
                                resultado_curtailment: Dict,
                                registros_pld: List[RegistroPLD],
                                df_perfil_aneel: Optional[pd.DataFrame]) -> Optional[OportunidadeBESS]:
        """
        VERSÃO ATUALIZADA: separa energia CONFIRMADA (COFF ou geração)
        de energia INDICATIVA (proxy PLD) no dimensionamento, e marca
        explicitamente na OportunidadeBESS qual fração do dado é
        confiável - informação essencial para quem for tomar decisão
        de investimento com base no relatório.
        """
        eventos_confirmados_sub = [
            e for e in resultado_curtailment['eventos_confirmados']
            if e.submercado == submercado
        ]
        eventos_indicativos_sub = [
            e for e in resultado_curtailment['eventos_indicativos_pld']
            if e.submercado == submercado
        ]

        tem_dado_confirmado_sub = len(eventos_confirmados_sub) > 0

        # Usa dado confirmado se disponível; caso contrário, cai para
        # indicativo MAS sinaliza isso explicitamente na oportunidade
        eventos_para_dimensionamento = (
            eventos_confirmados_sub if tem_dado_confirmado_sub
            else eventos_indicativos_sub
        )

        if not eventos_para_dimensionamento:
            logger.debug(f"Nenhum evento de curtailment para {submercado} - pulando")
            return None

        registros_pld_sub = [r for r in registros_pld if r.submercado == submercado]
        if not registros_pld_sub:
            return None

        janelas = self.spread_analyzer.identify_optimal_windows(registros_pld_sub, submercado)
        if not janelas:
            return None

        energia_disponivel_mwh = sum(
            e.magnitude_mw * e.duracao_horas for e in eventos_para_dimensionamento
        )

        if energia_disponivel_mwh <= 0:
            return None

        janela_carga = janelas['janela_carga']
        duracao_janela_carga = (
            janela_carga['hora_fim'] - janela_carga['hora_inicio']
        ) % 24 + 1
        capacidade_mwh, potencia_mw = self._dimensionar_bess(
            energia_disponivel_mwh,
            eventos_confirmados_sub,
            eventos_indicativos_sub,
            duracao_janela_carga,
        )

        periodo_dias = max(
            1.0,
            (
                max(r.timestamp for r in registros_pld_sub)
                - min(r.timestamp for r in registros_pld_sub)
            ).total_seconds() / 86400 + 1,
        )
        num_ciclos_possiveis = self._estimar_ciclos_possiveis(
            eventos_para_dimensionamento, capacidade_mwh, periodo_dias
        )

        spread_liquido = janelas['spread_liquido_rs_mwh']
        receita_anual_rs = min(
            energia_disponivel_mwh, capacidade_mwh * num_ciclos_possiveis
        ) * spread_liquido

        capex_total, payback_anos = self._calcular_financeiro(
            capacidade_mwh, potencia_mw, receita_anual_rs
        )

        fator_saturacao = self._calcular_fator_saturacao(submercado, df_perfil_aneel)

        volatilidade = pd.Series(
            [r.pld_mwh for r in registros_pld_sub]
        ).std()

        score = self._calcular_score_atratividade(
            energia_disponivel_mwh, spread_liquido, volatilidade,
            payback_anos, fator_saturacao, num_ciclos_possiveis
        )

        tipo_restricao_predominante = self._identificar_tipo_predominante(
            eventos_para_dimensionamento
        )

        timestamps = [e.timestamp for e in eventos_para_dimensionamento]

        return OportunidadeBESS(
            submercado=submercado,
            periodo_inicio=min(timestamps),
            periodo_fim=max(timestamps),
            tipo_restricao=tipo_restricao_predominante,
            energia_disponivel_mwh=round(energia_disponivel_mwh, 1),
            preco_medio_carga_rs_mwh=janelas['janela_carga']['pld_medio'],
            preco_medio_descarga_rs_mwh=janelas['janela_descarga']['pld_medio'],
            spread_medio_rs_mwh=spread_liquido,
            volatilidade_preco=round(volatilidade, 2),
            num_ciclos_possiveis=round(num_ciclos_possiveis, 0),
            receita_estimada_anual_rs=round(receita_anual_rs, 2),
            capacidade_recomendada_mwh=round(capacidade_mwh, 1),
            potencia_recomendada_mw=round(potencia_mw, 1),
            payback_anos=round(payback_anos, 1) if payback_anos else None,
            score_atratividade=round(score, 1),
            dados_confirmados=tem_dado_confirmado_sub,  # NOVO CAMPO - transparência
        )

    @staticmethod
    def _identificar_tipo_predominante(eventos: List) -> TipoRestricao:
        """Identifica o tipo de restrição mais frequente entre os eventos analisados"""
        if not eventos:
            return TipoRestricao.EXCESSO_OFERTA_PROXY_PLD

        contagem = {}
        for e in eventos:
            contagem[e.tipo] = contagem.get(e.tipo, 0) + 1

        return max(contagem, key=contagem.get)

    # ==================================================================
    # DETERMINAÇÃO DO TIPO DE RESTRIÇÃO PREDOMINANTE
    # ==================================================================
    @staticmethod
    def _identificar_restricao_predominante(
            eventos_curtailment: List[RestricaoOperativa],
            eventos_oversupply: List[RestricaoOperativa]) -> TipoRestricao:
        """
        Determina qual tipo de restrição domina o submercado, baseado
        na energia total associada a cada tipo de evento.
        """
        energia_por_tipo: Dict[TipoRestricao, float] = {}

        for evento in eventos_curtailment + eventos_oversupply:
            energia_evento = evento.magnitude_mw * evento.duracao_horas
            energia_por_tipo[evento.tipo] = energia_por_tipo.get(evento.tipo, 0.0) + energia_evento

        if not energia_por_tipo:
            return TipoRestricao.RESTRICAO_TRANSMISSAO

        return max(energia_por_tipo, key=energia_por_tipo.get)

    # ==================================================================
    # CÁLCULO DE ENERGIA DISPONÍVEL
    # ==================================================================
    @staticmethod
    def _calcular_energia_disponivel(
            eventos_curtailment: List[RestricaoOperativa],
            eventos_oversupply: List[RestricaoOperativa]) -> float:
        """
        Soma a energia total (MWh) associada a eventos de curtailment
        e excesso de geração - representa o "combustível" disponível
        para carregar o BESS a custo marginal muito baixo/zero.
        """
        energia_curtailment = sum(
            e.magnitude_mw * e.duracao_horas for e in eventos_curtailment
        )

        # Eventos de oversupply via PLD não têm magnitude_mw física
        # (são proxy de preço), então usamos apenas para validar
        # recorrência/consistência, não para soma direta de energia.
        # Aqui aplicamos um peso conservador para não dobrar a contagem.
        num_eventos_oversupply = len(eventos_oversupply)
        duracao_media_oversupply = (
            np.mean([e.duracao_horas for e in eventos_oversupply])
            if eventos_oversupply else 0.0
        )

        # Energia "indicativa" de oversupply (não soma direta, é fator
        # de confirmação do padrão identificado via curtailment)
        energia_oversupply_indicativa = 0.0
        if not eventos_curtailment and eventos_oversupply:
            # Se não há curtailment reportado mas há PLD baixo recorrente,
            # usamos uma estimativa conservadora baseada apenas na
            # recorrência (sem magnitude MW confiável)
            energia_oversupply_indicativa = num_eventos_oversupply * duracao_media_oversupply * 50  # MW proxy conservador

        return energia_curtailment + energia_oversupply_indicativa

    # ==================================================================
    # VOLATILIDADE E PERÍODO
    # ==================================================================
    @staticmethod
    def _calcular_volatilidade(registros_pld: List[RegistroPLD]) -> float:
        if not registros_pld:
            return 0.0
        valores = [r.pld_mwh for r in registros_pld]
        return float(np.std(valores))

    @staticmethod
    def _extrair_periodo(registros_geracao: List[RegistroGeracao],
                          registros_pld: List[RegistroPLD]) -> Tuple[datetime, datetime]:
        timestamps = [r.timestamp for r in registros_geracao] + \
                     [r.timestamp for r in registros_pld]

        if not timestamps:
            agora = datetime.now()
            return agora, agora

        return min(timestamps), max(timestamps)

    # ==================================================================
    # DIMENSIONAMENTO TÉCNICO DO BESS
    # ==================================================================
    def _dimensionar_bess(self,
                            energia_disponivel_mwh: float,
                            eventos_curtailment: List[RestricaoOperativa],
                            eventos_oversupply: List[RestricaoOperativa],
                            duracao_janela_carga: int) -> Tuple[float, float]:
        """
        Dimensiona capacidade (MWh) e potência (MW) recomendadas,
        baseado em:

        1. Magnitude TÍPICA dos eventos (não o total acumulado, que
           sobre-dimensionaria o ativo para capturar eventos raros)
        2. Duração típica dos eventos (define a relação MWh/MW, ou
           seja, o C-rate do sistema)
        3. Arredondamento para capacidades comerciais padrão de mercado
        """
        eventos_relevantes = eventos_curtailment if eventos_curtailment else eventos_oversupply

        if not eventos_relevantes:
            # Fallback: usa duração da janela de carga identificada no spread
            magnitude_tipica_mw = 50.0  # proxy conservador
            duracao_tipica_horas = duracao_janela_carga
        else:
            magnitudes = [e.magnitude_mw for e in eventos_relevantes if e.magnitude_mw > 0]
            duracoes = [e.duracao_horas for e in eventos_relevantes]

            # Usa percentil 75 da magnitude (captura eventos relevantes
            # sem ser dominado por outliers extremos)
            magnitude_tipica_mw = (
                float(np.percentile(magnitudes, 75)) if magnitudes else 50.0
            )
            duracao_tipica_horas = (
                float(np.median(duracoes)) if duracoes else duracao_janela_carga
            )

        potencia_mw = magnitude_tipica_mw
        capacidade_mwh = potencia_mw * duracao_tipica_horas

        # Arredonda para valores comerciais padrão mais próximos
        capacidade_mwh = self._arredondar_para_padrao(
            capacidade_mwh, Settings.BESS_CAPACIDADE_MWH_PADRAO
        )
        potencia_mw = self._arredondar_para_padrao(
            potencia_mw, Settings.BESS_POTENCIA_MW_PADRAO
        )

        return capacidade_mwh, potencia_mw

    @staticmethod
    def _estimar_ciclos_possiveis(
            eventos: List[RestricaoOperativa],
            capacidade_mwh: float,
            periodo_dias: float) -> float:
        """Annualizes equivalent full cycles from event energy and data coverage."""
        if capacidade_mwh <= 0 or periodo_dias <= 0:
            return 0.0

        energia_disponivel_mwh = sum(
            max(evento.magnitude_mw, 0) * max(evento.duracao_horas, 0)
            for evento in eventos
        )
        ciclos_no_periodo = energia_disponivel_mwh / capacidade_mwh
        return ciclos_no_periodo * 365 / periodo_dias

    @staticmethod
    def _arredondar_para_padrao(valor: float, opcoes_padrao: List[float]) -> float:
        """Arredonda para a opção comercial padrão mais próxima (sem ultrapassar o dobro)"""
        if valor <= 0:
            return opcoes_padrao[0]

        # Encontra a menor opção padrão que seja >= valor (conservador,
        # evita subdimensionamento), mas não ultrapassa a maior opção disponível
        opcoes_validas = [o for o in opcoes_padrao if o >= valor]
        if opcoes_validas:
            return min(opcoes_validas)

        return max(opcoes_padrao)

    # ==================================================================
    # ANÁLISE FINANCEIRA
    # ==================================================================
    def _calcular_financeiro(self,
                               capacidade_mwh: float,
                               potencia_mw: float,
                               receita_anual_rs: float) -> Tuple[float, Optional[float]]:
        """
        Calcula CAPEX total estimado e payback simples (não descontado).

        CAPEX = parcela ligada à energia (MWh) + parcela ligada à
        potência (MW, que define o tamanho do PCS/inversores).

        Payback simples considera receita líquida (após OPEX) anual
        constante - uma simplificação válida para triagem inicial de
        oportunidades, mas que deve ser refinada com fluxo de caixa
        descontado (VPL/TIR) na fase de bankability do projeto.
        """
        capex_total = (
            capacidade_mwh * self.capex_rs_por_mwh +
            potencia_mw * self.capex_rs_por_mw
        )

        opex_anual = capex_total * self.opex_pct_capex_ano
        receita_liquida_anual = receita_anual_rs - opex_anual

        if receita_liquida_anual <= 0:
            return capex_total, None  # payback indefinido (projeto inviável nesse cenário)

        payback_anos = capex_total / receita_liquida_anual

        return capex_total, payback_anos

    def calculate_npv(self,
                        capex_total: float,
                        receita_anual_rs: float,
                        opex_anual_rs: Optional[float] = None) -> Dict:
        """
        Calcula VPL (Valor Presente Líquido) e indicadores financeiros
        mais robustos para avaliação detalhada de uma oportunidade
        específica (uso recomendado na fase de due diligence, após
        a triagem inicial via score_atratividade).
        """
        opex_anual_rs = opex_anual_rs or (capex_total * self.opex_pct_capex_ano)
        fluxo_liquido_anual = receita_anual_rs - opex_anual_rs

        fluxos = [-capex_total] + [fluxo_liquido_anual] * self.vida_util_anos

        vpl = sum(
            fluxo / ((1 + self.taxa_desconto) ** i)
            for i, fluxo in enumerate(fluxos)
        )

        # TIR via busca binária simples (evita dependência de numpy_financial)
        tir = self._calcular_tir(fluxos)

        payback_descontado = self._calcular_payback_descontado(
            capex_total, fluxo_liquido_anual, self.taxa_desconto
        )

        return {
            'capex_total_rs': round(capex_total, 2),
            'opex_anual_rs': round(opex_anual_rs, 2),
            'fluxo_liquido_anual_rs': round(fluxo_liquido_anual, 2),
            'vpl_rs': round(vpl, 2),
            'tir_pct': round(tir * 100, 2) if tir is not None else None,
            'payback_simples_anos': round(capex_total / fluxo_liquido_anual, 1)
                if fluxo_liquido_anual > 0 else None,
            'payback_descontado_anos': round(payback_descontado, 1)
                if payback_descontado else None,
            'viavel': vpl > 0,
        }

    @staticmethod
    def _calcular_tir(fluxos: List[float],
                        precisao: float = 0.0001,
                        max_iter: int = 1000) -> Optional[float]:
        """Calcula TIR via método da bissecção (busca a taxa que zera o VPL)"""
        def vpl_para_taxa(taxa):
            return sum(f / ((1 + taxa) ** i) for i, f in enumerate(fluxos))

        taxa_min, taxa_max = -0.99, 5.0

        if vpl_para_taxa(taxa_min) * vpl_para_taxa(taxa_max) > 0:
            return None  # sem raiz no intervalo (projeto nunca se paga ou sempre positivo)

        for _ in range(max_iter):
            taxa_media = (taxa_min + taxa_max) / 2
            vpl_medio = vpl_para_taxa(taxa_media)

            if abs(vpl_medio) < precisao:
                return taxa_media

            if vpl_para_taxa(taxa_min) * vpl_medio < 0:
                taxa_max = taxa_media
            else:
                taxa_min = taxa_media

        return taxa_media

    @staticmethod
    def _calcular_payback_descontado(capex_total: float,
                                        fluxo_anual: float,
                                        taxa_desconto: float,
                                        max_anos: int = 30) -> Optional[float]:
        """Calcula payback considerando valor do dinheiro no tempo"""
        if fluxo_anual <= 0:
            return None

        acumulado = -capex_total
        for ano in range(1, max_anos + 1):
            fluxo_descontado = fluxo_anual / ((1 + taxa_desconto) ** ano)
            acumulado += fluxo_descontado

            if acumulado >= 0:
                # Interpolação linear dentro do ano
                fracao_ano = 1 - (acumulado / fluxo_descontado)
                return ano - 1 + fracao_ano

        return None  # não paga dentro do horizonte considerado

    # ==================================================================
    # FATOR DE SATURAÇÃO DE MERCADO (ANEEL)
    # ==================================================================
    @staticmethod
    def _calcular_fator_saturacao(submercado: str,
                                     df_perfil_aneel: Optional[pd.DataFrame]) -> float:
        """
        Calcula um fator de 0 a 1 que penaliza submercados onde já
        há muitos projetos de BESS outorgados (risco de canibalização
        de spread - quanto mais BESS competindo pela mesma arbitragem,
        menor o spread capturável no longo prazo).

        Retorna 1.0 (sem penalização) se não houver dados de saturação.
        """
        if df_perfil_aneel is None or df_perfil_aneel.empty:
            return 1.0

        if 'submercado' not in df_perfil_aneel.columns:
            return 1.0

        linha_submercado = df_perfil_aneel[df_perfil_aneel['submercado'] == submercado]

        if linha_submercado.empty:
            return 1.0

        bess_outorgado_mw = linha_submercado['bess_outorgado_mw'].sum() \
            if 'bess_outorgado_mw' in linha_submercado.columns else 0.0

        capacidade_total_mw = linha_submercado['potencia_total_mw'].sum() \
            if 'potencia_total_mw' in linha_submercado.columns else 1.0

        if capacidade_total_mw <= 0:
            return 1.0

        # Razão BESS/capacidade total do submercado
        razao_saturacao = bess_outorgado_mw / capacidade_total_mw

        # Fator decai suavemente conforme saturação aumenta
        # (ex.: 5% de saturação -> fator ~0.9; 20% -> fator ~0.6)
        fator = max(0.1, 1.0 - (razao_saturacao * 4))

        return round(fator, 3)

    # ==================================================================
    # SCORE DE ATRATIVIDADE CONSOLIDADO
    # ==================================================================
    def _calcular_score_atratividade(self,
                                        energia_disponivel_mwh: float,
                                        spread_liquido: float,
                                        volatilidade_preco: float,
                                        payback_anos: Optional[float],
                                        fator_saturacao: float,
                                        num_ciclos_possiveis: float) -> float:
        """
        Score final de 0-100 combinando todas as dimensões relevantes
        para priorização de investimento em BESS.

        Pesos:
        - 25%: energia disponível para arbitragem (volume de oportunidade)
        - 25%: spread líquido (margem por MWh ciclado)
        - 20%: payback (quanto menor, melhor - invertido)
        - 15%: número de ciclos possíveis (utilização do ativo)
        - 15%: volatilidade de preço (upside potencial além do spread médio)

        O fator_saturacao atua como multiplicador final (penalização
        de mercados saturados), não como componente aditivo.
        """
        # Normalização com faixas de referência de mercado (heurísticas)
        score_energia = min(100, (energia_disponivel_mwh / 5000) * 100)  # 5000 MWh = saturação do score
        score_spread = min(100, max(0, (spread_liquido / 200) * 100))    # R\$200/MWh = spread excelente
        score_volatilidade = min(100, (volatilidade_preco / 150) * 100)   # proxy de upside
        score_ciclos = min(100, (num_ciclos_possiveis / 300) * 100)       # 300 ciclos/ano = uso intenso

        if payback_anos is None or payback_anos <= 0:
            score_payback = 0.0
        else:
            # Payback de 3 anos -> score 100; 10 anos -> score ~0
            score_payback = max(0, min(100, (10 - payback_anos) / 7 * 100))

        score_bruto = (
            0.25 * score_energia +
            0.25 * score_spread +
            0.20 * score_payback +
            0.15 * score_ciclos +
            0.15 * score_volatilidade
        )

        score_final = score_bruto * fator_saturacao

        return max(0.0, min(100.0, score_final))

    # ==================================================================
    # EXPORTAÇÃO E RELATÓRIOS
    # ==================================================================
    def export_opportunities_to_dataframe(self,
                                             oportunidades: List[OportunidadeBESS]) -> pd.DataFrame:
        """Converte lista de OportunidadeBESS em DataFrame para análise/export"""
        if not oportunidades:
            return pd.DataFrame()

        data = []
        for o in oportunidades:
            data.append({
                'submercado': o.submercado,
                'periodo_inicio': o.periodo_inicio,
                'periodo_fim': o.periodo_fim,
                'tipo_restricao': o.tipo_restricao.value,
                'energia_disponivel_mwh': o.energia_disponivel_mwh,
                'preco_medio_carga_rs_mwh': o.preco_medio_carga_rs_mwh,
                'preco_medio_descarga_rs_mwh': o.preco_medio_descarga_rs_mwh,
                'spread_medio_rs_mwh': o.spread_medio_rs_mwh,
                'volatilidade_preco': o.volatilidade_preco,
                'num_ciclos_possiveis_ano': o.num_ciclos_possiveis,
                'receita_estimada_anual_rs': o.receita_estimada_anual_rs,
                'capacidade_recomendada_mwh': o.capacidade_recomendada_mwh,
                'potencia_recomendada_mw': o.potencia_recomendada_mw,
                'c_rate_recomendado': round(
                    o.potencia_recomendada_mw / o.capacidade_recomendada_mwh, 2
                ) if o.capacidade_recomendada_mwh > 0 else None,
                'payback_anos': o.payback_anos,
                'score_atratividade': o.score_atratividade,
            })

        df = pd.DataFrame(data)
        return df.sort_values('score_atratividade', ascending=False).reset_index(drop=True)

    def generate_executive_report(self,
                                     oportunidades: List[OportunidadeBESS],
                                     output_path: Optional[str] = None) -> Dict:
        """
        Gera relatório executivo consolidado, pronto para serialização
        (JSON) ou alimentação de dashboard.

        Args:
            oportunidades: lista de OportunidadeBESS já calculadas
            output_path: se fornecido, salva o relatório em JSON neste caminho

        Returns:
            Dict com resumo executivo, ranking completo e detalhamento
            financeiro das top oportunidades.
        """
        if not oportunidades:
            logger.warning("Nenhuma oportunidade para gerar relatório")
            return {
                'data_geracao': datetime.now().isoformat(),
                'total_oportunidades': 0,
                'oportunidades': [],
            }

        df_oportunidades = self.export_opportunities_to_dataframe(oportunidades)

        # --- Estatísticas agregadas ---
        energia_total_identificada_mwh = sum(o.energia_disponivel_mwh for o in oportunidades)
        capacidade_total_recomendada_mwh = sum(o.capacidade_recomendada_mwh for o in oportunidades)
        receita_total_potencial_rs = sum(o.receita_estimada_anual_rs for o in oportunidades)

        paybacks_validos = [o.payback_anos for o in oportunidades if o.payback_anos is not None]
        payback_medio = float(np.mean(paybacks_validos)) if paybacks_validos else None

        melhor_oportunidade = oportunidades[0]  # já ordenado por score

        # --- Análise financeira detalhada (VPL/TIR) das top 5 ---
        top_5_detalhado = []
        for o in oportunidades[:5]:
            capex_estimado = (
                o.capacidade_recomendada_mwh * self.capex_rs_por_mwh +
                o.potencia_recomendada_mw * self.capex_rs_por_mw
            )
            analise_financeira = self.calculate_npv(
                capex_total=capex_estimado,
                receita_anual_rs=o.receita_estimada_anual_rs,
            )

            top_5_detalhado.append({
                'submercado': o.submercado,
                'tipo_restricao': o.tipo_restricao.value,
                'score_atratividade': o.score_atratividade,
                'dimensionamento': {
                    'capacidade_mwh': o.capacidade_recomendada_mwh,
                    'potencia_mw': o.potencia_recomendada_mw,
                    'c_rate': round(
                        o.potencia_recomendada_mw / o.capacidade_recomendada_mwh, 2
                    ) if o.capacidade_recomendada_mwh > 0 else None,
                },
                'operacional': {
                    'preco_medio_carga_rs_mwh': o.preco_medio_carga_rs_mwh,
                    'preco_medio_descarga_rs_mwh': o.preco_medio_descarga_rs_mwh,
                    'spread_medio_rs_mwh': o.spread_medio_rs_mwh,
                    'ciclos_anuais_estimados': o.num_ciclos_possiveis,
                },
                'financeiro': analise_financeira,
            })

        resumo = {
            'data_geracao': datetime.now().isoformat(),
            'premissas_economicas': {
                'capex_rs_por_mwh': self.capex_rs_por_mwh,
                'capex_rs_por_mw': self.capex_rs_por_mw,
                'opex_pct_capex_ano': self.opex_pct_capex_ano,
                'vida_util_anos': self.vida_util_anos,
                'taxa_desconto': self.taxa_desconto,
                'eficiencia_bess': self.spread_analyzer.eficiencia_bess,
            },
            'resumo_executivo': {
                'total_oportunidades_identificadas': len(oportunidades),
                'submercado_mais_atrativo': melhor_oportunidade.submercado,
                'score_melhor_oportunidade': melhor_oportunidade.score_atratividade,
                'energia_total_identificada_mwh': round(energia_total_identificada_mwh, 1),
                'capacidade_total_recomendada_mwh': round(capacidade_total_recomendada_mwh, 1),
                'receita_total_potencial_anual_rs': round(receita_total_potencial_rs, 1),
                'payback_medio_anos': round(payback_medio, 1) if payback_medio else None,
            },
            'ranking_completo': df_oportunidades.to_dict('records'),
            'top_5_analise_detalhada': top_5_detalhado,
        }

        if output_path:
            self._salvar_relatorio_json(resumo, output_path)

        return resumo

    @staticmethod
    def _salvar_relatorio_json(resumo: Dict, output_path: str):
        """Salva relatório em JSON, tratando serialização de datas/objetos"""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        def default_serializer(obj):
            if isinstance(obj, (datetime, pd.Timestamp)):
                return obj.isoformat()
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Tipo não serializável: {type(obj)}")

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(resumo, f, ensure_ascii=False, indent=2, default=default_serializer)

        logger.info(f"Relatório executivo salvo em: {path}")

    # ==================================================================
    # ANÁLISE DE SENSIBILIDADE
    # ==================================================================
    def sensitivity_analysis(self,
                                oportunidade: OportunidadeBESS,
                                variacao_capex_pct: List[float] = None,
                                variacao_spread_pct: List[float] = None) -> pd.DataFrame:
        """
        Executa análise de sensibilidade do payback/VPL de uma
        oportunidade específica, variando CAPEX e spread de preço -
        essencial para avaliar robustez da oportunidade frente a
        incertezas de mercado (ex.: queda de preços de baterias,
        mudanças regulatórias no PLD).
        """
        variacao_capex_pct = variacao_capex_pct or [-0.20, -0.10, 0.0, 0.10, 0.20]
        variacao_spread_pct = variacao_spread_pct or [-0.30, -0.15, 0.0, 0.15, 0.30]

        capex_base = (
            oportunidade.capacidade_recomendada_mwh * self.capex_rs_por_mwh +
            oportunidade.potencia_recomendada_mw * self.capex_rs_por_mw
        )

        resultados = []

        for delta_capex in variacao_capex_pct:
            for delta_spread in variacao_spread_pct:
                capex_cenario = capex_base * (1 + delta_capex)
                receita_cenario = oportunidade.receita_estimada_anual_rs * (1 + delta_spread)

                analise = self.calculate_npv(
                    capex_total=capex_cenario,
                    receita_anual_rs=receita_cenario,
                )

                resultados.append({
                    'variacao_capex_pct': delta_capex * 100,
                    'variacao_spread_pct': delta_spread * 100,
                    'capex_cenario_rs': round(capex_cenario, 0),
                    'receita_cenario_rs': round(receita_cenario, 0),
                    'vpl_rs': analise['vpl_rs'],
                    'payback_simples_anos': analise['payback_simples_anos'],
                    'viavel': analise['viavel'],
                })

        return pd.DataFrame(resultados)