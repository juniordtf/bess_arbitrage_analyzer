# data_collectors/aneel_collector.py
"""
Coletor de dados abertos da ANEEL (Agência Nacional de Energia Elétrica).

Fontes principais:
- Portal de Dados Abertos ANEEL (CKAN API): https://dadosabertos.aneel.gov.br/
- SIGA (Sistema de Informações de Geração da ANEEL): empreendimentos de
  geração outorgados, em operação, construção e planejados.

A ANEEL disponibiliza dados por empreendimento/UF (estado), não por
submercado. Por isso este coletor inclui um mapeamento UF -> Submercado
para permitir o cruzamento com os dados do ONS.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path
import logging
import time
import json

from config.settings import Settings
from models.data_models import TipoFonte

logger = logging.getLogger(__name__)


class ANEELDataCollector:
    """
    Coletor de dados abertos da ANEEL via API CKAN.
    """

    # IDs de datasets (packages) conhecidos no portal CKAN da ANEEL.
    # Esses nomes de package são relativamente estáveis; os resource_id
    # internos podem variar - por isso resolvemos dinamicamente via
    # package_show.
    PACKAGES = {
        "siga_geracao": "siga-sistema-de-informacoes-de-geracao-da-aneel",
        "empreendimentos_geracao": "empreendimentos-geracao",
        "capacidade_geracao": "capacidade-instalada-de-geracao-eletrica",
        "bess_armazenamento": "sistema-de-armazenamento-de-energia-sae",
    }

    # Mapeamento de tipos de fonte do SIGA/ANEEL para o Enum interno
    FONTE_MAPPING = {
        "UTE": TipoFonte.TERMICA,
        "UHE": TipoFonte.HIDRAULICA,
        "PCH": TipoFonte.HIDRAULICA,
        "CGH": TipoFonte.HIDRAULICA,
        "EOL": TipoFonte.EOLICA,
        "UFV": TipoFonte.SOLAR,
        "CGU": TipoFonte.BIOMASSA,
        "BIOMASSA": TipoFonte.BIOMASSA,
        "TERMICA": TipoFonte.TERMICA,
        "HIDRICA": TipoFonte.HIDRAULICA,
        "EOLICA": TipoFonte.EOLICA,
        "SOLAR": TipoFonte.SOLAR,
    }

    # Mapeamento UF -> Submercado (ONS)
    UF_SUBMERCADO = {
        # Submercado Norte
        "AC": "N", "AM": "N", "AP": "N", "PA": "N", "RO": "N", "RR": "N", "TO": "N",
        # Submercado Nordeste
        "AL": "NE", "BA": "NE", "CE": "NE", "MA": "NE", "PB": "NE",
        "PE": "NE", "PI": "NE", "RN": "NE", "SE": "NE",
        # Submercado Sul
        "PR": "S", "RS": "S", "SC": "S",
        # Submercado Sudeste/Centro-Oeste
        "DF": "SE", "ES": "SE", "GO": "SE", "MG": "SE", "MS": "SE",
        "MT": "SE", "RJ": "SE", "SP": "SE",
    }

    def __init__(self, cache_dir: str = None):
        self.cache_dir = Path(cache_dir or Settings.CACHE_DIR) / "aneel"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 BESS-Arbitrage-Analyzer/1.0',
            'Accept': 'application/json',
        })
        self._resource_id_cache: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Infra CKAN
    # ------------------------------------------------------------------
    def _package_show(self, package_name: str) -> Dict[str, Any]:
        """Consulta metadados de um package CKAN, retornando lista de resources"""
        url = f"{Settings.ANEEL_API_URL}/package_show"
        params = {"id": package_name}

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            payload = response.json()

            if not payload.get("success"):
                logger.warning(f"package_show falhou para {package_name}: {payload}")
                return {}

            return payload.get("result", {})

        except requests.exceptions.RequestException as e:
            logger.error(f"Erro ao consultar package {package_name}: {e}")
            return {}

    def _resolve_resource_id(self, package_name: str,
                              resource_name_hint: str = None) -> Optional[str]:
        """
        Resolve o resource_id mais recente/relevante de um package.
        Prioriza resources CSV cujo nome contenha o hint fornecido.
        """
        cache_key = f"{package_name}:{resource_name_hint}"
        if cache_key in self._resource_id_cache:
            return self._resource_id_cache[cache_key]

        package = self._package_show(package_name)
        resources = package.get("resources", [])

        if not resources:
            logger.warning(f"Nenhum resource encontrado para package {package_name}")
            return None

        candidatos = [
            r for r in resources
            if r.get("format", "").upper() in ("CSV", "JSON")
        ]

        if resource_name_hint:
            filtrados = [
                r for r in candidatos
                if resource_name_hint.lower() in r.get("name", "").lower()
            ]
            if filtrados:
                candidatos = filtrados

        if not candidatos:
            logger.warning(
                f"Nenhum resource CSV/JSON encontrado para {package_name} "
                f"(hint={resource_name_hint})"
            )
            return None

        # Pega o mais recentemente atualizado
        candidatos.sort(
            key=lambda r: r.get("last_modified") or r.get("created") or "",
            reverse=True
        )
        resource_id = candidatos[0]["id"]
        self._resource_id_cache[cache_key] = resource_id
        return resource_id

    def _datastore_search(self, resource_id: str,
                           filters: Optional[Dict] = None,
                           limit: int = 32000,
                           max_pages: int = 50) -> pd.DataFrame:
        """
        Busca registros via datastore_search com paginação.
        """
        url = f"{Settings.ANEEL_API_URL}/datastore_search"
        all_records = []
        offset = 0

        for page in range(max_pages):
            params = {
                "resource_id": resource_id,
                "limit": limit,
                "offset": offset,
            }
            if filters:
                params["filters"] = json.dumps(filters)

            try:
                response = self.session.get(url, params=params, timeout=60)
                response.raise_for_status()
                payload = response.json()

                if not payload.get("success"):
                    logger.warning(f"datastore_search falhou: {payload}")
                    break

                result = payload.get("result", {})
                records = result.get("records", [])

                if not records:
                    break

                all_records.extend(records)
                total = result.get("total", 0)
                offset += limit

                if offset >= total:
                    break

                time.sleep(0.2)  # cortesia com a API

            except requests.exceptions.RequestException as e:
                logger.error(f"Erro no datastore_search (offset={offset}): {e}")
                break

        if not all_records:
            return pd.DataFrame()

        df = pd.DataFrame(all_records)
        df = df.drop(columns=["_id"], errors="ignore")
        return df

    def _download_resource_csv(self, package_name: str,
                                resource_name_hint: str = None,
                                force_refresh: bool = False) -> pd.DataFrame:
        """
        Alternativa ao datastore_search: baixa diretamente o CSV do resource
        (útil quando o datastore não está ativo para o resource).
        """
        cache_path = self.cache_dir / f"{package_name}_{resource_name_hint or 'default'}.csv"

        if cache_path.exists() and not force_refresh:
            logger.info(f"Carregando cache ANEEL: {cache_path}")
            return self._safe_read_csv(cache_path)

        package = self._package_show(package_name)
        resources = package.get("resources", [])

        candidatos = [r for r in resources if r.get("format", "").upper() == "CSV"]
        if resource_name_hint:
            filtrados = [
                r for r in candidatos
                if resource_name_hint.lower() in r.get("name", "").lower()
            ]
            if filtrados:
                candidatos = filtrados

        if not candidatos:
            logger.warning(f"Nenhum CSV encontrado para {package_name}")
            return pd.DataFrame()

        candidatos.sort(
            key=lambda r: r.get("last_modified") or r.get("created") or "",
            reverse=True
        )
        resource_url = candidatos[0]["url"]

        try:
            logger.info(f"Baixando resource ANEEL: {resource_url}")
            response = self.session.get(resource_url, timeout=120)
            response.raise_for_status()
            cache_path.write_bytes(response.content)
            return self._safe_read_csv(cache_path)

        except requests.exceptions.RequestException as e:
            logger.error(f"Erro ao baixar resource {resource_url}: {e}")
            return pd.DataFrame()

    @staticmethod
    def _safe_read_csv(path: Path) -> pd.DataFrame:
        attempts = [
            {'sep': ';', 'encoding': 'utf-8'},
            {'sep': ';', 'encoding': 'latin-1'},
            {'sep': ',', 'encoding': 'utf-8'},
            {'sep': ',', 'encoding': 'latin-1'},
        ]
        last_err = None
        for kwargs in attempts:
            try:
                df = pd.read_csv(path, **kwargs, low_memory=False)
                if df.shape[1] > 1:
                    return df
            except Exception as e:
                last_err = e
                continue
        logger.error(f"Não foi possível ler o CSV {path}: {last_err}")
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Empreendimentos de geração (SIGA)
    # ------------------------------------------------------------------
    def get_empreendimentos_geracao(self,
                                     fonte: Optional[TipoFonte] = None,
                                     uf: Optional[str] = None,
                                     apenas_operacao: bool = True,
                                     force_refresh: bool = False) -> pd.DataFrame:
        """
        Busca empreendimentos de geração cadastrados no SIGA/ANEEL.
        Retorna DataFrame normalizado com colunas:
        - nome_usina, fonte, uf, submercado, potencia_outorgada_mw,
          potencia_fiscalizada_mw, data_entrada_operacao, situacao
        """
        df = self._download_resource_csv(
            self.PACKAGES["siga_geracao"],
            resource_name_hint="empreendimento",
            force_refresh=force_refresh
        )

        if df.empty:
            # Fallback: tenta datastore_search
            resource_id = self._resolve_resource_id(
                self.PACKAGES["empreendimentos_geracao"]
            )
            if resource_id:
                df = self._datastore_search(resource_id)

        if df.empty:
            logger.warning("Nenhum dado de empreendimentos de geração encontrado")
            return pd.DataFrame()

        df_normalizado = self._normalize_empreendimentos(df)

        if fonte:
            df_normalizado = df_normalizado[df_normalizado['fonte'] == fonte]

        if uf:
            df_normalizado = df_normalizado[
                df_normalizado['uf'].str.upper() == uf.upper()
            ]

        if apenas_operacao:
            df_normalizado = df_normalizado[
                df_normalizado['situacao'].str.contains(
                    'OPERA', case=False, na=False
                )
            ]

        return df_normalizado

    def _normalize_empreendimentos(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normaliza colunas variáveis do SIGA para schema interno padronizado"""
        columns_lower = {c.lower().strip(): c for c in df.columns}

        col_patterns = {
            'nome_usina': ['nom_empreendimento', 'nome_usina', 'nm_usina',
                           'nom_usina', 'empreendimento'],
            'fonte': ['sig_tipogeracao', 'tip_combustivel', 'sigla_fonte',
                      'fonte', 'tipo_fonte', 'dsc_fonte'],
            'uf': ['sig_uf', 'uf', 'sg_uf', 'estado'],
            'potencia_outorgada_mw': ['mda_potenciaoutorgadakw', 'potencia_outorgada',
                                       'val_potenciaoutorgada'],
            'potencia_fiscalizada_mw': ['mda_potenciafiscalizadakw', 'potencia_fiscalizada',
                                          'val_potenciafiscalizada'],
            'data_entrada_operacao': ['dat_entradaoperacao', 'data_operacao',
                                        'dat_inicio_operacao'],
            'situacao': ['dsc_situacaoacompanhamento', 'situacao', 'dsc_situacao'],
            'ceg': ['cod_ceg', 'ceg', 'codigo_ceg'],
            'municipio': ['nom_municipio', 'municipio'],
        }

        resultado = pd.DataFrame()

        for campo_destino, candidatos in col_patterns.items():
            for cand in candidatos:
                if cand in columns_lower:
                    resultado[campo_destino] = df[columns_lower[cand]]
                    break
            else:
                resultado[campo_destino] = None

        # Converte potências de kW para MW quando necessário
        for col_mw in ['potencia_outorgada_mw', 'potencia_fiscalizada_mw']:
            if col_mw in resultado.columns:
                resultado[col_mw] = resultado[col_mw].apply(self._to_float)
                # Heurística: se os valores parecem estar em kW (muito altos),
                # converte para MW. SIGA costuma trazer em kW.
                if resultado[col_mw].median() > 10000:
                    resultado[col_mw] = resultado[col_mw] / 1000.0

        # Normaliza fonte
        if 'fonte' in resultado.columns:
            resultado['fonte_raw'] = resultado['fonte']
            resultado['fonte'] = resultado['fonte'].apply(self._map_fonte)

        # Adiciona submercado baseado em UF
        if 'uf' in resultado.columns:
            resultado['submercado'] = resultado['uf'].apply(
                lambda x: self.UF_SUBMERCADO.get(str(x).strip().upper(), 'DESCONHECIDO')
            )

        # Normaliza data
        if 'data_entrada_operacao' in resultado.columns:
            resultado['data_entrada_operacao'] = pd.to_datetime(
                resultado['data_entrada_operacao'], errors='coerce'
            )

        return resultado

    def _map_fonte(self, valor_raw) -> Optional[TipoFonte]:
        """Mapeia string de fonte do SIGA para o Enum interno"""
        if pd.isna(valor_raw):
            return None

        valor_str = str(valor_raw).strip().upper()

        # Tenta match exato primeiro
        if valor_str in self.FONTE_MAPPING:
            return self.FONTE_MAPPING[valor_str]

        # Tenta match parcial (contains)
        for key, tipo in self.FONTE_MAPPING.items():
            if key in valor_str or valor_str in key:
                return tipo

        return None

    # ------------------------------------------------------------------
    # Capacidade instalada agregada
    # ------------------------------------------------------------------
    def get_capacidade_instalada_por_submercado(self,
                                                  force_refresh: bool = False) -> pd.DataFrame:
        """
        Retorna capacidade instalada agregada por submercado e fonte,
        útil para calcular fator de capacidade teórico máximo e
        dimensionar potencial de curtailment.
        """
        df_empreendimentos = self.get_empreendimentos_geracao(
            apenas_operacao=True,
            force_refresh=force_refresh
        )

        if df_empreendimentos.empty:
            return pd.DataFrame()

        df_agregado = df_empreendimentos.groupby(
            ['submercado', 'fonte'], dropna=False
        ).agg(
            potencia_total_mw=('potencia_fiscalizada_mw', 'sum'),
            num_usinas=('nome_usina', 'count')
        ).reset_index()

        return df_agregado

    # ------------------------------------------------------------------
    # BESS / Sistemas de Armazenamento já outorgados
    # ------------------------------------------------------------------
    def get_bess_outorgados(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Busca sistemas de armazenamento de energia (BESS/SAE) já
        outorgados ou em processo de outorga pela ANEEL.
        Útil para análise de mercado/concorrência por submercado.
        """
        df = self._download_resource_csv(
            self.PACKAGES["bess_armazenamento"],
            force_refresh=force_refresh
        )

        if df.empty:
            logger.info(
                "Nenhum dado específico de BESS encontrado. "
                "Tentando filtrar empreendimentos gerais por palavra-chave."
            )
            df_emp = self.get_empreendimentos_geracao(
                apenas_operacao=False,
                force_refresh=force_refresh
            )
            if df_emp.empty:
                return pd.DataFrame()

            mask = df_emp['nome_usina'].str.contains(
                'BESS|BATERIA|ARMAZENAMENTO|STORAGE',
                case=False, na=False, regex=True
            )
            return df_emp[mask]

        columns_lower = {c.lower().strip(): c for c in df.columns}
        resultado = pd.DataFrame()

        col_patterns = {
            'nome_projeto': ['nom_empreendimento', 'nome_projeto', 'nome'],
            'uf': ['sig_uf', 'uf', 'estado'],
            'potencia_mw': ['mda_potenciakw', 'potencia_mw', 'potencia'],
            'capacidade_mwh': ['capacidade_mwh', 'energia_mwh'],
            'situacao': ['dsc_situacao', 'situacao'],
            'data_prevista': ['dat_previsao', 'data_prevista_operacao'],
        }

        for campo_destino, candidatos in col_patterns.items():
            for cand in candidatos:
                if cand in columns_lower:
                    resultado[campo_destino] = df[columns_lower[cand]]
                    break
            else:
                resultado[campo_destino] = None

        if 'uf' in resultado.columns:
            resultado['submercado'] = resultado['uf'].apply(
                lambda x: self.UF_SUBMERCADO.get(str(x).strip().upper(), 'DESCONHECIDO')
            )

        return resultado

    # ------------------------------------------------------------------
    # Tarifas e bandeiras (contexto econômico)
    # ------------------------------------------------------------------
    def get_tarifas_distribuidoras(self,
                                     uf: Optional[str] = None,
                                     force_refresh: bool = False) -> pd.DataFrame:
        """
        Busca tarifas homologadas por distribuidora - útil para análise
        de arbitragem no lado da demanda (BESS atrás do medidor/C&I),
        complementar à arbitragem no mercado de curto prazo (PLD).
        """
        df = self._download_resource_csv(
            "tarifas-de-energia-eletrica",
            resource_name_hint="homologada",
            force_refresh=force_refresh
        )

        if df.empty:
            logger.warning("Dados de tarifas não disponíveis")
            return pd.DataFrame()

        columns_lower = {c.lower().strip(): c for c in df.columns}

        if uf and 'sig_uf' in columns_lower:
            df = df[df[columns_lower['sig_uf']].str.upper() == uf.upper()]
        elif uf and 'uf' in columns_lower:
            df = df[df[columns_lower['uf']].str.upper() == uf.upper()]

        return df

    # ------------------------------------------------------------------
    # Utilitários
    # ------------------------------------------------------------------
    @staticmethod
    def _to_float(value) -> float:
        """Converte valores potencialmente com vírgula decimal (padrão BR) para float"""
        if pd.isna(value):
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)

        value_str = str(value).strip()
        if not value_str:
            return 0.0

        # Trata formato BR: 1.234,56 -> 1234.56
        if ',' in value_str and '.' in value_str:
            value_str = value_str.replace('.', '').replace(',', '.')
        elif ',' in value_str:
            value_str = value_str.replace(',', '.')

        try:
            return float(value_str)
        except ValueError:
            return 0.0

    def clear_cache(self):
        """Limpa todo o cache local da ANEEL"""
        for f in self.cache_dir.glob("*.csv"):
            f.unlink()
        logger.info("Cache ANEEL limpo")

    # ------------------------------------------------------------------
    # Método de conveniência: snapshot completo por submercado
    # ------------------------------------------------------------------
    def build_submercado_profile(self,
                                   force_refresh: bool = False) -> pd.DataFrame:
        """
        Constrói um perfil consolidado por submercado, combinando:
        - Capacidade instalada por fonte
        - Projetos de BESS já outorgados (concorrência/saturação)

        Retorna DataFrame com colunas:
        submercado, fonte, potencia_total_mw, num_usinas,
        bess_outorgado_mw, bess_outorgado_mwh
        """
        df_capacidade = self.get_capacidade_instalada_por_submercado(
            force_refresh=force_refresh
        )

        if df_capacidade.empty:
            logger.warning("Não foi possível construir perfil - dados de capacidade vazios")
            return pd.DataFrame()

        df_bess = self.get_bess_outorgados(force_refresh=force_refresh)

        if not df_bess.empty and 'submercado' in df_bess.columns:
            df_bess_agg = df_bess.groupby('submercado').agg(
                bess_outorgado_mw=('potencia_mw', lambda x: pd.to_numeric(
                    x, errors='coerce'
                ).sum()),
                bess_outorgado_mwh=('capacidade_mwh', lambda x: pd.to_numeric(
                    x, errors='coerce'
                ).sum()),
                num_projetos_bess=('nome_projeto', 'count')
            ).reset_index()

            df_resultado = df_capacidade.merge(
                df_bess_agg, on='submercado', how='left'
            )
        else:
            df_resultado = df_capacidade.copy()
            df_resultado['bess_outorgado_mw'] = 0.0
            df_resultado['bess_outorgado_mwh'] = 0.0
            df_resultado['num_projetos_bess'] = 0

        df_resultado[['bess_outorgado_mw', 'bess_outorgado_mwh', 'num_projetos_bess']] = \
            df_resultado[['bess_outorgado_mw', 'bess_outorgado_mwh', 'num_projetos_bess']].fillna(0)

        return df_resultado


class ANEELDatasetUnavailableError(Exception):
    """Levantada quando nenhum dataset pôde ser baixado/parseado"""
    pass