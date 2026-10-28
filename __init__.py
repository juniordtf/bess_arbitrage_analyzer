# __init__.py
"""
BESS Arbitrage Analyzer
========================

Solução de análise de dados abertos do ONS e ANEEL para mapeamento de
restrições de geração por submercado e identificação de oportunidades
de instalação de BESS (Battery Energy Storage System) para arbitragem
de energia.

Módulos principais:
- config: configurações e parâmetros do sistema
- models: modelos de dados (dataclasses) usados em todo o pipeline
- data_collectors: coletores de dados ONS e ANEEL
- analysis: motores de análise (curtailment, spread, oportunidades BESS)
- visualization: dashboard e visualizações

Exemplo de uso rápido:

    from bess_arbitrage_analyzer import (
        ONSDataCollector, ANEELDataCollector,
        CurtailmentAnalyzer, PriceSpreadAnalyzer, BESSOpportunityFinder,
        TipoFonte
    )

    ons = ONSDataCollector()
    registros_pld = ons.get_pld_horario(data_inicio, data_fim)
    registros_geracao = ons.get_geracao_por_fonte(TipoFonte.EOLICA, data_inicio, data_fim)

    finder = BESSOpportunityFinder()
    oportunidades = finder.find_opportunities(registros_geracao, registros_pld)
"""

__version__ = "1.0.0"
__author__ = "BESS Arbitrage Analyzer Team"

from config.settings import Settings

from models.data_models import (
    TipoFonte,
    TipoRestricao,
    RegistroGeracao,
    RegistroPLD,
    RestricaoOperativa,
    OportunidadeBESS,
)

from data_collectors.ons_collector import ONSDataCollector, ONSDatasetUnavailableError
from data_collectors.ccee_collector import CCEEDataCollector, CCEEDatasetUnavailableError
from data_collectors.aneel_collector import ANEELDataCollector, ANEELDatasetUnavailableError

from analysis.curtailment_analyzer import CurtailmentAnalyzer
from analysis.price_spread_analyzer import PriceSpreadAnalyzer
from analysis.bess_opportunity_finder import BESSOpportunityFinder

__all__ = [
    # Config
    "Settings",
    # Models
    "TipoFonte",
    "TipoRestricao",
    "RegistroGeracao",
    "RegistroPLD",
    "RestricaoOperativa",
    "OportunidadeBESS",
    # Collectors
    "ONSDataCollector",
    "ONSDatasetUnavailableError",
    "CCEEDataCollector",
    "CCEEDatasetUnavailableError",
    "ANEELDataCollector",
    "ANEELDatasetUnavailableError",
]