# data_collectors/__init__.py
"""
Coletores de dados abertos para fontes oficiais do setor elétrico
brasileiro.

- ONSDataCollector: dados de geração, PLD, carga e restrições
  operativas do Operador Nacional do Sistema Elétrico (ONS)
- ANEELDataCollector: dados de empreendimentos de geração, capacidade
  instalada e projetos de armazenamento (BESS) outorgados pela ANEEL

Ambos os coletores implementam cache local em disco para evitar
downloads repetidos e tratam variações de schema/encoding comuns
nos datasets públicos dessas instituições.
"""

from data_collectors.ons_collector import (
    ONSDataCollector,
    ONSDatasetUnavailableError,
)
from data_collectors.aneel_collector import (
    ANEELDataCollector,
    ANEELDatasetUnavailableError,
)

__all__ = [
    "ONSDataCollector",
    "ONSDatasetUnavailableError",
    "ANEELDataCollector",
    "ANEELDatasetUnavailableError",
]