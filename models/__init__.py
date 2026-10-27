# models/__init__.py
"""
Modelos de dados (dataclasses) utilizados em todo o pipeline de análise.

Estes modelos representam:
- Registros brutos de geração e PLD (dados de entrada do ONS)
- Eventos de restrição operativa identificados pela análise
- Oportunidades de BESS consolidadas (saída final do pipeline)
"""

from models.data_models import (
    TipoFonte,
    TipoRestricao,
    RegistroGeracao,
    RegistroPLD,
    RestricaoOperativa,
    OportunidadeBESS,
)

__all__ = [
    "TipoFonte",
    "TipoRestricao",
    "RegistroGeracao",
    "RegistroPLD",
    "RestricaoOperativa",
    "OportunidadeBESS",
]