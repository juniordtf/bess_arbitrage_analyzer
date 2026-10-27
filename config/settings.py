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
    ONS_BASE_URL = "https://ons-dl-prod-opendata.s3.amazonaws.com/dataset"
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