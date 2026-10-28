# config/settings.py
from dataclasses import dataclass
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Garante que o diretório raiz do projeto está no path,
# permitindo execução tanto via `python main.py` na raiz
# quanto via `python -m bess_arbitrage_analyzer.main`
_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

@dataclass
class Settings:
    # URLs base dos dados abertos
    ONS_BASE_URL = "https://ons-aws-prod-opendata.s3.amazonaws.com/dataset"
    ONS_API_URL = "https://apicarga.ons.org.br/prd/apiload"
    ANEEL_API_URL = "https://dadosabertos.aneel.gov.br/api/3/action"
    
    # Submercados
    SUBMERCADOS = ["N", "NE", "S", "SE"]  # Norte, Nordeste, Sul, Sudeste/Centro-Oeste
    
    # Parâmetros de análise
    JANELA_ANALISE_DIAS = 90
    LIMIAR_CURTAILMENT_MW = 100
    EFICIENCIA_BESS = 0.85  # Round-trip efficiency
    
    # Parâmetros técnicos BESS (referência mercado)
    BESS_CAPACIDADE_MWH_PADRAO = [50, 100, 200, 500]  # MWh
    BESS_POTENCIA_MW_PADRAO = [25, 50, 100, 250]  # MW (C-rate 0.5)
    VIDA_UTIL_CICLOS = 6000
    
    # Diretórios
    CACHE_DIR = "./cache"
    OUTPUT_DIR = "./output"
    
    @staticmethod
    def get_date_range():
        end_date = datetime.now()
        start_date = end_date - timedelta(days=Settings.JANELA_ANALISE_DIAS)
        return start_date, end_date

    # ------------------------------------------------------------
    # CCEE - Câmara de Comercialização de Energia Elétrica
    # ------------------------------------------------------------
    CCEE_BASE_URL = "https://www.ccee.org.br"
    CCEE_DADOS_ABERTOS_URL = "https://dadosabertos.ccee.org.br"

    # Padrões de URL testados em ordem de prioridade - a CCEE
    # eventualmente reorganiza seu portal, então mantemos múltiplas
    # tentativas antes de desistir.
    # API CKAN da CCEE (quando disponível) - mesmo padrão usado
    # pelo ONS, pois ambos os portais de dados abertos do setor
    # elétrico brasileiro costumam usar CKAN
    CCEE_CKAN_API_URL = "https://dadosabertos.ccee.org.br/api/3/action"
    CCEE_CKAN_DATASET_PLD = "pld_horario"

    # ------------------------------------------------------------
    # Resolução dinâmica de ano (evita buscar anos não publicados)
    # ------------------------------------------------------------
    ANO_MINIMO_HISTORICO = 2020   # não tenta buscar antes disso
    MAX_TENTATIVAS_FALLBACK_ANO = 3  # quantos anos para trás tentar

    ONS_GERACAO_EOLICA_URL_PATTERN = (
        "https://ons-aws-prod-opendata.s3.amazonaws.com/dataset/"
        "geracao_eolica_ho/GERACAO_EOLICA_HO_{ano}.csv"
    )
    ONS_GERACAO_SOLAR_URL_PATTERN = (
        "https://ons-aws-prod-opendata.s3.amazonaws.com/dataset/"
        "geracao_solar_ho/GERACAO_SOLAR_HO_{ano}.csv"
    )
    ONS_GERACAO_HIDRAULICA_URL_PATTERN = (
        "https://ons-aws-prod-opendata.s3.amazonaws.com/dataset/"
        "geracao_hidraulica_ho/GERACAO_HIDRAULICA_HO_{ano}.csv"
    )
    ONS_GERACAO_TERMICA_URL_PATTERN = (
        "https://ons-aws-prod-opendata.s3.amazonaws.com/dataset/"
        "geracao_termica_ho/GERACAO_TERMICA_HO_{ano}.csv"
    )

    # O ONS não publica PLD horário no catálogo atual; usar a CCEE.