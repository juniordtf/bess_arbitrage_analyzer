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

class TipoRestricao(str, Enum):
    """
    Tipo de restrição identificada na análise consolidada.

    IMPORTANTE: os tipos abaixo distinguem a CONFIABILIDADE da fonte
    de detecção. Eventos detectados via dado real de COFF (prioritário)
    carregam o sufixo _COFF; eventos inferidos apenas via proxy de
    preço (PLD) carregam o sufixo _PROXY_PLD e devem ser tratados com
    ressalva em relatórios (energia "indicativa", não "confirmada").
    """
    CURTAILMENT_EOLICO_COFF = "CURTAILMENT_EOLICO_COFF"
    CURTAILMENT_SOLAR_COFF = "CURTAILMENT_SOLAR_COFF"
    CURTAILMENT_ELETRICO_COFF = "CURTAILMENT_ELETRICO_COFF"
    CURTAILMENT_ENERGETICO_COFF = "CURTAILMENT_ENERGETICO_COFF"
    RESTRICAO_TRANSMISSAO = "RESTRICAO_TRANSMISSAO"
    EXCESSO_OFERTA_PROXY_PLD = "EXCESSO_OFERTA_PROXY_PLD"  # renomeado p/ deixar claro que é proxy

    @property
    def is_dado_confirmado(self) -> bool:
        """Indica se o tipo de restrição vem de dado real (COFF) ou é apenas inferência via preço"""
        return self != TipoRestricao.EXCESSO_OFERTA_PROXY_PLD

    @property
    def label_amigavel(self) -> str:
        """Rótulo legível para exibição em relatórios/dashboard"""
        labels = {
            TipoRestricao.CURTAILMENT_EOLICO_COFF: "Curtailment Eólico (dado real ONS)",
            TipoRestricao.CURTAILMENT_SOLAR_COFF: "Curtailment Solar (dado real ONS)",
            TipoRestricao.CURTAILMENT_ELETRICO_COFF: "Restrição Elétrica/Transmissão (dado real ONS)",
            TipoRestricao.CURTAILMENT_ENERGETICO_COFF: "Restrição Energética (dado real ONS)",
            TipoRestricao.RESTRICAO_TRANSMISSAO: "Restrição de Transmissão (inferida)",
            TipoRestricao.EXCESSO_OFERTA_PROXY_PLD: "Excesso de Oferta (indicativo via PLD baixo)",
        }
        return labels.get(self, self.value)

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
    payback_anos: Optional[float]
    score_atratividade: float
    dados_confirmados: bool = True  # NOVO - False quando baseado apenas em proxy PLD

    @property
    def nivel_confiabilidade(self) -> str:
        """Rótulo legível do nível de confiabilidade do dimensionamento"""
        return "✅ Confirmado (dado real)" if self.dados_confirmados else "⚠️ Indicativo (proxy PLD)"

class RazaoRestricaoCOFF(str, Enum):
    """
    Classificação oficial de razão de corte usada pelo ONS no
    processo de restrição COFF (Constrained-Off).

    - ENERGETICA: corte por excesso de oferta frente à demanda
      (situação em que o PLD-proxy teria chance real de capturar o evento)
    - ELETRICA: corte por limitação de rede/transmissão local
      (PLD nacional/submercado normalmente NÃO reflete esse tipo de evento)
    - CONFIABILIDADE: corte por critérios de segurança operativa do SIN
    - OUTRA: razões não classificadas nas categorias acima
    """
    ENERGETICA = "ENERGETICA"
    ELETRICA = "ELETRICA"
    CONFIABILIDADE = "CONFIABILIDADE"
    OUTRA = "OUTRA"


@dataclass
class RestricaoCOFF:
    """
    Registro real de restrição de geração (curtailment), obtido
    diretamente do dataset oficial do ONS de Restrição COFF -
    granular por usina e intervalo, com razão classificada quando
    publicada, para medir restrições de geração no Brasil.
    """
    timestamp: datetime
    usina: str
    submercado: str
    fonte: "TipoFonte"
    potencia_restringida_mw: float
    razao_restricao: RazaoRestricaoCOFF
    geracao_disponivel_mw: Optional[float] = None  # capacidade que poderia ter sido gerada
    observacao: Optional[str] = None
    intervalo_horas: float = 1.0

    @property
    def energia_cortada_mwh(self) -> float:
        """Energia cortada no intervalo do registro (potência x duração)."""
        return self.potencia_restringida_mw * self.intervalo_horas

    def __post_init__(self):
        if self.potencia_restringida_mw < 0:
            raise ValueError(
                f"potencia_restringida_mw não pode ser negativa: "
                f"{self.potencia_restringida_mw}"
            )
        if self.intervalo_horas <= 0:
            raise ValueError(
                f"intervalo_horas deve ser positivo: {self.intervalo_horas}"
            )