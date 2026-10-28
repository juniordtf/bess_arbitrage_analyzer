# data_collectors/__init__.py
"""
Coletores de dados abertos para fontes oficiais do setor elétrico
brasileiro.

- ONSDataCollector: dados de geração por fonte (eólica, solar,
  hidráulica, térmica) do Operador Nacional do Sistema Elétrico.
  Também serve como fallback secundário para PLD.
- CCEEDataCollector: fonte PRIMÁRIA de PLD horário (Preço de
  Liquidação das Diferenças) - a CCEE é a entidade oficial
  responsável pelo cálculo e publicação desse índice.
- ANEELDataCollector: dados de empreendimentos de geração e
  projetos de armazenamento (BESS) outorgados.

Todos os coletores implementam:
- Cache local em disco (TTL diferenciado para ano vigente vs. histórico)
- Fallback automático de ano (evita falhas quando o dataset do ano
  calendário atual ainda não foi publicado)
- Retry com backoff exponencial em falhas de rede
"""

from data_collectors.ons_collector import (
    ONSDataCollector,
    ONSDatasetUnavailableError,
)
from data_collectors.ccee_collector import (
    CCEEDataCollector,
    CCEEDatasetUnavailableError,
)
from data_collectors.aneel_collector import (
    ANEELDataCollector,
    ANEELDatasetUnavailableError,
)

__all__ = [
    "ONSDataCollector",
    "ONSDatasetUnavailableError",
    "CCEEDataCollector",
    "CCEEDatasetUnavailableError",
    "ANEELDataCollector",
    "ANEELDatasetUnavailableError",
]