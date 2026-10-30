# visualization/data_loader.py
"""
Camada de carregamento e cache de dados para o dashboard Streamlit.

Separa a lógica de obtenção/processamento de dados da camada de
apresentação, permitindo cache eficiente via st.cache_data/st.cache_resource
e reuso em diferentes páginas do dashboard.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
import logging

from config.settings import Settings
from models.data_models import RegistroGeracao, RegistroPLD, RestricaoCOFF, TipoFonte
from data_collectors.ons_collector import ONSDataCollector
from data_collectors.aneel_collector import ANEELDataCollector
from data_collectors.ccee_collector import CCEEDataCollector, CCEEDatasetUnavailableError
from analysis.curtailment_analyzer import CurtailmentAnalyzer
from analysis.price_spread_analyzer import PriceSpreadAnalyzer
from analysis.bess_opportunity_finder import BESSOpportunityFinder

logger = logging.getLogger(__name__)


# ======================================================================
# RECURSOS (instâncias pesadas, reutilizadas entre reruns)
# ======================================================================
@st.cache_resource
def get_ons_collector() -> ONSDataCollector:
    return ONSDataCollector(cache_dir=Settings.CACHE_DIR)


@st.cache_resource
def get_aneel_collector() -> ANEELDataCollector:
    return ANEELDataCollector(cache_dir=Settings.CACHE_DIR)


@st.cache_resource
def get_curtailment_analyzer() -> CurtailmentAnalyzer:
    return CurtailmentAnalyzer()


@st.cache_resource
def get_spread_analyzer(eficiencia: float) -> PriceSpreadAnalyzer:
    return PriceSpreadAnalyzer(eficiencia_bess=eficiencia)


@st.cache_resource
def get_opportunity_finder(eficiencia: float,
                             capex_mwh: float,
                             capex_mw: float) -> BESSOpportunityFinder:
    return BESSOpportunityFinder(
        curtailment_analyzer=get_curtailment_analyzer(),
        spread_analyzer=get_spread_analyzer(eficiencia),
        capex_rs_por_mwh=capex_mwh,
        capex_rs_por_mw=capex_mw,
    )


# ======================================================================
# DADOS (cacheados por parâmetros - invalida automaticamente se mudarem)
# ======================================================================
@st.cache_data(ttl=3600, show_spinner="Coletando dados de geração do ONS...")
def load_generation_data(fontes: Tuple[str, ...],
                          data_inicio: datetime,
                          data_fim: datetime) -> List[RegistroGeracao]:
    """Carrega dados de geração do ONS (cacheado por 1h)"""
    collector = get_ons_collector()

    fonte_map = {
        'EOLICA': TipoFonte.EOLICA,
        'SOLAR': TipoFonte.SOLAR,
        'HIDRAULICA': TipoFonte.HIDRAULICA,
        'TERMICA': TipoFonte.TERMICA,
    }

    registros = []
    for fonte_str in fontes:
        fonte_enum = fonte_map[fonte_str]
        try:
            registros.extend(
                collector.get_geracao_por_fonte(fonte_enum, data_inicio, data_fim)
            )
        except Exception as e:
            logger.error(f"Erro ao carregar {fonte_str}: {e}")
            st.warning(f"⚠️ Não foi possível carregar dados de {fonte_str}: {e}")

    return registros


@st.cache_data(ttl=3600, show_spinner="Coletando dados de restrição COFF do ONS...")
def load_coff_data(fonte: str,
                   data_inicio: datetime,
                   data_fim: datetime) -> List[RestricaoCOFF]:
    """Carrega registros oficiais mensais de restrição COFF."""
    collector = get_ons_collector()
    fonte_enum = TipoFonte[fonte]
    try:
        return collector.get_restricao_coff(fonte_enum, data_inicio, data_fim)
    except Exception as e:
        logger.error(f"Erro ao carregar COFF {fonte}: {e}")
        st.warning(
            f"⚠️ Não foi possível carregar o dataset COFF de {fonte}: {e}"
        )
        return []


@st.cache_data(ttl=3600, show_spinner="Coletando dados de PLD do ONS...")
def load_pld_data(data_inicio: datetime, data_fim: datetime) -> List[RegistroPLD]:
    """Carrega dados de PLD horário do ONS (cacheado por 1h)"""
    collector = get_ons_collector()
    try:
        return collector.get_pld_horario(data_inicio, data_fim)
    except Exception as e:
        logger.error(f"Erro ao carregar PLD: {e}")
        st.error(f"❌ Erro crítico ao carregar PLD: {e}")
        return []


@st.cache_data(ttl=86400, show_spinner="Coletando perfil de mercado ANEEL...")
def load_aneel_profile() -> pd.DataFrame:
    """Carrega perfil de capacidade/saturação ANEEL (cacheado por 24h)"""
    collector = get_aneel_collector()
    try:
        return collector.build_submercado_profile()
    except Exception as e:
        logger.warning(f"Erro ao carregar perfil ANEEL (não crítico): {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def convert_registros_to_df(_registros_geracao: List[RegistroGeracao]) -> pd.DataFrame:
    """Converte lista de RegistroGeracao em DataFrame para visualização direta"""
    if not _registros_geracao:
        return pd.DataFrame()

    data = [{
        'timestamp': r.timestamp,
        'submercado': r.submercado,
        'fonte': r.fonte.value,
        'geracao_programada_mw': r.geracao_programada_mw,
        'geracao_verificada_mw': r.geracao_verificada_mw,
        'curtailment_mw': r.curtailment_mw,
        'capacidade_instalada_mw': r.capacidade_instalada_mw,
        'fator_capacidade': r.fator_capacidade,
    } for r in _registros_geracao]

    return pd.DataFrame(data)


@st.cache_data(ttl=3600)
def convert_pld_to_df(_registros_pld: List[RegistroPLD]) -> pd.DataFrame:
    """Converte lista de RegistroPLD em DataFrame para visualização direta"""
    if not _registros_pld:
        return pd.DataFrame()

    data = [{
        'timestamp': r.timestamp,
        'submercado': r.submercado,
        'pld_mwh': r.pld_mwh,
        'carga_mw': r.carga_mw,
    } for r in _registros_pld]

    df = pd.DataFrame(data)
    df['hora'] = df['timestamp'].dt.hour
    df['data'] = df['timestamp'].dt.date
    return df


def filter_by_submercado(registros_geracao: List[RegistroGeracao],
                          registros_pld: List[RegistroPLD],
                          submercado: Optional[str]) -> Tuple[List, List]:
    """Filtra listas de registros por submercado (usado fora do cache)"""
    if not submercado or submercado == "TODOS":
        return registros_geracao, registros_pld

    geracao_filtrada = [r for r in registros_geracao if r.submercado == submercado]
    pld_filtrado = [r for r in registros_pld if r.submercado == submercado]
    return geracao_filtrada, pld_filtrado


@st.cache_resource
def get_ccee_collector() -> CCEEDataCollector:
    return CCEEDataCollector(cache_dir=Settings.CACHE_DIR)

@st.cache_data(ttl=3600, show_spinner="Coletando dados de PLD da CCEE...")
def load_pld_data(data_inicio: datetime, data_fim: datetime) -> List[RegistroPLD]:
    """
    Carrega dados de PLD horário, priorizando a CCEE (fonte oficial
    do índice) e usando o ONS como fallback secundário em caso de
    indisponibilidade - mesma lógica aplicada no main.py, mantendo
    consistência entre o pipeline CLI e o dashboard interativo.
    """
    ccee_collector = get_ccee_collector()
    ons_collector = get_ons_collector()

    try:
        registros = ccee_collector.get_pld_horario(data_inicio, data_fim)
        if registros:
            return registros
        logger.warning("CCEE retornou lista vazia - tentando fallback via ONS")

    except CCEEDatasetUnavailableError as e:
        logger.warning(f"CCEE indisponível ({e}) - tentando fallback via ONS")
        st.toast("⚠️ CCEE indisponível, usando ONS como fallback para PLD", icon="⚠️")
    except Exception as e:
        logger.error(f"Erro inesperado ao coletar PLD via CCEE: {e}")

    try:
        registros = ons_collector.get_pld_horario(data_inicio, data_fim)
        return registros
    except Exception as e:
        logger.error(f"Erro ao coletar PLD via ONS (fallback): {e}")
        st.error(f"❌ Erro crítico: PLD indisponível tanto na CCEE quanto no ONS: {e}")
        return []