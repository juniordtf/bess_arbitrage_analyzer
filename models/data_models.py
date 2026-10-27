# models/data_models.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List
from enum import Enum

class TipoFonte(Enum):
    HIDRAULICA = "HIDRAULICA"
    TERMICA = "TERMICA"
    EOLICA = "EOLICA"
    SOLAR = "SOLAR"
    BIOMASSA = "BIOMASSA"

class TipoRestricao(Enum):
    CURTAILMENT_EOLICO = "CURTAILMENT_EOLICO"
    CURTAILMENT_SOLAR = "CURTAILMENT_SOLAR"
    RESTRICAO_TRANSMISSAO = "RESTRICAO_TRANSMISSAO"
    EXCESSO_HIDRICO = "EXCESSO_HIDRICO"
    SOBRA_TERMICA = "SOBRA_TERMICA"

@dataclass
class RegistroGeracao:
    timestamp: datetime
    submercado: str
    fonte: TipoFonte
    geracao_programada_mw: float
    geracao_verificada_mw: float
    capacidade_instalada_mw: float
    
    @property
    def curtailment_mw(self) -> float:
        return max(0, self.geracao_programada_mw - self.geracao_verificada_mw)
    
    @property
    def fator_capacidade(self) -> float:
        if self.capacidade_instalada_mw > 0:
            return self.geracao_verificada_mw / self.capacidade_instalada_mw
        return 0.0

@dataclass
class RegistroPLD:
    timestamp: datetime
    submercado: str
    pld_mwh: float
    carga_mw: float = 0.0
    
@dataclass
class RestricaoOperativa:
    timestamp: datetime
    submercado: str
    tipo: TipoRestricao
    magnitude_mw: float
    duracao_horas: float
    custo_oportunidade_rs: Optional[float] = None
    descricao: str = ""

@dataclass
class OportunidadeBESS:
    submercado: str
    periodo_inicio: datetime
    periodo_fim: datetime
    tipo_restricao: TipoRestricao
    energia_disponivel_mwh: float
    preco_medio_carga_rs_mwh: float
    preco_medio_descarga_rs_mwh: float
    spread_medio_rs_mwh: float
    volatilidade_preco: float
    num_ciclos_possiveis: float
    receita_estimada_anual_rs: float
    capacidade_recomendada_mwh: float
    potencia_recomendada_mw: float
    payback_anos: Optional[float] = None
    score_atratividade: float = 0.0