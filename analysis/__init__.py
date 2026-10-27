# analysis/__init__.py
"""
Motores de análise do BESS Arbitrage Analyzer.

Pipeline de análise em três camadas:

1. CurtailmentAnalyzer
   Identifica restrições de geração (curtailment eólico/solar) e
   excesso de oferta (via PLD) por submercado, fonte e janela temporal.

2. PriceSpreadAnalyzer
   Analisa a dinâmica de preços (PLD) para identificar janelas ótimas
   de carga/descarga e simula operação histórica de arbitragem.

3. BESSOpportunityFinder
   Consolida as duas análises anteriores com dados de mercado (ANEEL)
   para gerar oportunidades de investimento dimensionadas e com
   viabilidade econômica (CAPEX, VPL, TIR, payback).

Uso típico:

    from analysis import CurtailmentAnalyzer, PriceSpreadAnalyzer, BESSOpportunityFinder

    curtailment = CurtailmentAnalyzer()
    spread = PriceSpreadAnalyzer(eficiencia_bess=0.87)
    finder = BESSOpportunityFinder(curtailment, spread)

    oportunidades = finder.find_opportunities(registros_geracao, registros_pld)
"""

from analysis.curtailment_analyzer import CurtailmentAnalyzer
from analysis.price_spread_analyzer import PriceSpreadAnalyzer
from analysis.bess_opportunity_finder import BESSOpportunityFinder

__all__ = [
    "CurtailmentAnalyzer",
    "PriceSpreadAnalyzer",
    "BESSOpportunityFinder",
]