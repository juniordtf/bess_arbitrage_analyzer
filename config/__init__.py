# config/__init__.py
"""
Módulo de configurações do BESS Arbitrage Analyzer.

Centraliza parâmetros técnicos, econômicos e de infraestrutura
(URLs de APIs, diretórios de cache, premissas de dimensionamento).
"""

from config.settings import Settings

__all__ = ["Settings"]