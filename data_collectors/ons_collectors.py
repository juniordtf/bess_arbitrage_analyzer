# data_collectors/ons_collector.py
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import logging
from pathlib import Path
import time

from config.settings import Settings
from models.data_models import RegistroGeracao, RegistroPLD, TipoFonte

logger = logging.getLogger(__name__)


class ONSDataCollector:
    """
    Coletor de dados abertos do ONS.
    Baseado nos datasets públicos disponíveis em:
    https://dados.ons.org.br/
    """

    DATASETS = {
        "geracao_termica": "geracao_termica_ho",
        "geracao_eolica": "geracao_eolica_ho",
        "geracao_solar": "geracao_solar_ho",
        "geracao_hidraulica": "geracao_hidraulica_ho",
        "pld_horario": "pld_horario",
        "carga_verificada": "curva_carga_ho",
        "restricao_operativa": "restricao_coff_cmse",
        "balanco_energia": "balanco_energia_subsistemas_ho",
        "geracao_usina": "geracao_usina_2_ho",  # geração por usina individual
    }

    # Padrões de colunas conhecidos (o ONS alterna nomenclaturas entre datasets)
    COLUMN_PATTERNS = {
        'timestamp': ['din_instante', 'data', 'din_referenciautc',
                      'din_referencia', 'timestamp', 'dat_referencia'],
        'submercado': ['id_subsistema', 'nom_subsistema', 'cod_subsistema',
                       'sigla_subsistema', 'submercado'],
        'geracao': ['val_geracao', 'val_geracaoprogramada', 'val_geracaoverificada',
                    'val_geracaoenergiaho', 'val_potenciaverificada'],
        'programada': ['val_geracaoprogramada', 'val_programado', 'val_geracaoreferenciaho'],
        'capacidade': ['val_capacidadeinstalada', 'val_potenciainstalada', 'mda_potenciainstaladakw'],
        'pld': ['val_pld', 'pld', 'val_pldmedio'],
        'carga': ['val_cargaenergiamwmed', 'val_carga', 'val_cargaverificada'],
        'usina': ['nom_usina', 'id_ente', 'ceg'],
    }

    def __init__(self, cache_dir: str = None):
        self.cache_dir = Path(cache_dir or Settings.CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 BESS-Arbitrage-Analyzer/1.0'
        })

    # ------------------------------------------------------------------
    # Infra de cache / download
    # ------------------------------------------------------------------
    def _get_cache_path(self, dataset_name: str, year: int) -> Path:
        return self.cache_dir / f"{dataset_name}_{year}.csv"

    def _download_ons_csv(self, dataset_name: str, year: int,
                           force_refresh: bool = False,
                           max_retries: int = 3) -> pd.DataFrame:
        """
        Baixa CSV do ONS com retry e fallback de encoding.
        Padrão de URL: {BASE}/{dataset}/{DATASET}_{ANO}.csv
        """
        cache_path = self._get_cache_path(dataset_name, year)

        if cache_path.exists() and not force_refresh:
            logger.info(f"Carregando do cache: {cache_path}")
            return self._safe_read_csv(cache_path)

        url = f"{Settings.ONS_BASE_URL}/{dataset_name}/{dataset_name.upper()}_{year}.csv"

        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Baixando dados (tentativa {attempt}): {url}")
                response = self.session.get(url, timeout=60)
                response.raise_for_status()

                cache_path.write_bytes(response.content)
                df = self._safe_read_csv(cache_path)
                logger.info(f"Dados carregados: {len(df)} registros")
                return df

            except requests.exceptions.HTTPError as e:
                if response.status_code == 404:
                    logger.warning(f"Dataset não encontrado para {year}: {url}")
                    return pd.DataFrame()
                logger.warning(f"Erro HTTP (tentativa {attempt}): {e}")
                time.sleep(2 ** attempt)

            except requests.exceptions.RequestException as e:
                logger.warning(f"Erro de conexão (tentativa {attempt}): {e}")
                time.sleep(2 ** attempt)

        logger.error(f"Falha ao baixar {dataset_name} para {year} após {max_retries} tentativas")
        return pd.DataFrame()

    @staticmethod
    def _safe_read_csv(path: Path) -> pd.DataFrame:
        """Tenta múltiplos encodings/separadores comuns usados pelo ONS"""
        attempts = [
            {'sep': ';', 'encoding': 'utf-8'},
            {'sep': ';', 'encoding': 'latin-1'},
            {'sep': ',', 'encoding': 'utf-8'},
            {'sep': ',', 'encoding': 'latin-1'},
        ]
        last_err = None
        for kwargs in attempts:
            try:
                df = pd.read_csv(path, **kwargs)
                if df.shape[1] > 1:
                    return df
            except Exception as e:
                last_err = e
                continue
        logger.error(f"Não foi possível ler o CSV {path}: {last_err}")
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Detecção dinâmica de colunas
    # ------------------------------------------------------------------
    def _detect_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Detecta mapeamento de colunas baseado em padrões conhecidos do ONS.
        Retorna dict {campo_logico: nome_coluna_real} ou {} se timestamp/submercado
        não forem encontrados (campos obrigatórios).
        """
        if df.empty:
            return {}

        columns_lower = {c.lower().strip(): c for c in df.columns}
        mapping = {}

        for campo, candidatos in self.COLUMN_PATTERNS.items():
            for cand in candidatos:
                if cand in columns_lower:
                    mapping[campo] = columns_lower[cand]
                    break

        # Campos obrigatórios para qualquer parsing
        obrigatorios = ['timestamp', 'submercado']
        if not all(c in mapping for c in obrigatorios):
            logger.warning(
                f"Colunas obrigatórias não encontradas. "
                f"Disponíveis: {list(df.columns)}"
            )
            return {}

        return mapping

    # ------------------------------------------------------------------
    # Geração por fonte
    # ------------------------------------------------------------------
    def get_geracao_por_fonte(self, fonte: TipoFonte,
                               data_inicio: datetime,
                               data_fim: datetime) -> List[RegistroGeracao]:
        """Busca dados de geração por fonte e período"""

        dataset_map = {
            TipoFonte.EOLICA: "geracao_eolica",
            TipoFonte.SOLAR: "geracao_solar",
            TipoFonte.HIDRAULICA: "geracao_hidraulica",
            TipoFonte.TERMICA: "geracao_termica",
        }

        dataset_key = dataset_map.get(fonte)
        if not dataset_key:
            logger.warning(f"Fonte {fonte} não suportada")
            return []

        registros = []
        anos = range(data_inicio.year, data_fim.year + 1)

        for ano in anos:
            df = self._download_ons_csv(self.DATASETS[dataset_key], ano)
            if df.empty:
                continue
            registros.extend(
                self._parse_geracao_df(df, fonte, data_inicio, data_fim)
            )

        logger.info(f"{fonte.value}: {len(registros)} registros no período")
        return registros

    def _parse_geracao_df(self, df: pd.DataFrame, fonte: TipoFonte,
                           data_inicio: datetime, data_fim: datetime) -> List[RegistroGeracao]:
        """
        Parser genérico para DataFrames de geração do ONS.
        """
        registros = []
        col_mapping = self._detect_columns(df)

        if not col_mapping or 'geracao' not in col_mapping:
            logger.warning(f"Mapeamento de colunas insuficiente para {fonte.value}")
            return registros

        df = df.copy()
        df['_ts'] = pd.to_datetime(df[col_mapping['timestamp']], errors='coerce')
        df = df.dropna(subset=['_ts'])

        mask = (df['_ts'] >= data_inicio) & (df['_ts'] <= data_fim)
        df_filtered = df.loc[mask]

        geracao_col = col_mapping['geracao']
        programada_col = col_mapping.get('programada')
        capacidade_col = col_mapping.get('capacidade')
        submercado_col = col_mapping['submercado']

        for _, row in df_filtered.iterrows():
            try:
                geracao_verificada = self._to_float(row[geracao_col])
                geracao_programada = (
                    self._to_float(row[programada_col])
                    if programada_col else geracao_verificada
                )
                capacidade = (
                    self._to_float(row[capacidade_col])
                    if capacidade_col else 0.0
                )

                registro = RegistroGeracao(
                    timestamp=row['_ts'],
                    submercado=self._normalize_submercado(row[submercado_col]),
                    fonte=fonte,
                    geracao_programada_mw=geracao_programada,
                    geracao_verificada_mw=geracao_verificada,
                    capacidade_instalada_mw=capacidade
                )
                registros.append(registro)

            except (ValueError, KeyError, TypeError) as e:
                continue

        return registros

    # ------------------------------------------------------------------
    # PLD horário
    # ------------------------------------------------------------------
    def get_pld_horario(self, data_inicio: datetime,
                         data_fim: datetime) -> List[RegistroPLD]:
        """
        Busca PLD horário por submercado.
        """
        registros = []
        anos = range(data_inicio.year, data_fim.year + 1)

        for ano in anos:
            df = self._download_ons_csv(self.DATASETS['pld_horario'], ano)
            if df.empty:
                continue
            registros.extend(
                self._parse_pld_df(df, data_inicio, data_fim)
            )

        logger.info(f"PLD: {len(registros)} registros no período")
        return registros

    def _parse_pld_df(self, df: pd.DataFrame,
                       data_inicio: datetime,
                       data_fim: datetime) -> List[RegistroPLD]:
        registros = []
        col_mapping = self._detect_columns(df)

        if not col_mapping or 'pld' not in col_mapping:
            logger.warning("Mapeamento de colunas insuficiente para PLD")
            return registros

        df = df.copy()
        df['_ts'] = pd.to_datetime(df[col_mapping['timestamp']], errors='coerce')
        df = df.dropna(subset=['_ts'])

        mask = (df['_ts'] >= data_inicio) & (df['_ts'] <= data_fim)
        df_filtered = df.loc[mask]

        pld_col = col_mapping['pld']
        carga_col = col_mapping.get('carga')
        submercado_col = col_mapping['submercado']

        for _, row in df_filtered.iterrows():
            try:
                pld_valor = self._to_float(row[pld_col])
                carga_valor = (
                    self._to_float(row[carga_col])
                    if carga_col else 0.0
                )

                registro = RegistroPLD(
                    timestamp=row['_ts'],
                    submercado=self._normalize_submercado(row[submercado_col]),
                    pld_mwh=pld_valor,
                    carga_mw=carga_valor
                )
                registros.append(registro)

            except (ValueError, KeyError, TypeError):
                continue

        return registros

    # ------------------------------------------------------------------
    # Restrições operativas (cortes/COFF/CMSE)
    # ------------------------------------------------------------------
    def get_restricoes_operativas(self, data_inicio: datetime,
                                   data_fim: datetime) -> pd.DataFrame:
        """
        Busca dados de restrição operativa (cortes de geração por
        restrição de transmissão/segurança - COFF/CMSE).
        Retorna DataFrame bruto normalizado, já que a estrutura varia
        bastante entre os datasets do ONS.
        """
        frames = []
        anos = range(data_inicio.year, data_fim.year + 1)

        for ano in anos:
            df = self._download_ons_csv(self.DATASETS['restricao_operativa'], ano)
            if df.empty:
                continue

            col_mapping = self._detect_columns(df)
            if not col_mapping:
                continue

            df = df.copy()
            df['_ts'] = pd.to_datetime(df[col_mapping['timestamp']], errors='coerce')
            df = df.dropna(subset=['_ts'])
            mask = (df['_ts'] >= data_inicio) & (df['_ts'] <= data_fim)
            df_filtered = df.loc[mask].copy()

            if 'submercado' in col_mapping:
                df_filtered['submercado_norm'] = df_filtered[
                    col_mapping['submercado']
                ].apply(self._normalize_submercado)

            frames.append(df_filtered)

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------------
    # Carga verificada
    # ------------------------------------------------------------------
    def get_carga_verificada(self, data_inicio: datetime,
                              data_fim: datetime) -> pd.DataFrame:
        """Busca curva de carga verificada por submercado"""
        frames = []
        anos = range(data_inicio.year, data_fim.year + 1)

        for ano in anos:
            df = self._download_ons_csv(self.DATASETS['carga_verificada'], ano)
            if df.empty:
                continue

            col_mapping = self._detect_columns(df)
            if not col_mapping:
                continue

            df = df.copy()
            df['_ts'] = pd.to_datetime(df[col_mapping['timestamp']], errors='coerce')
            df = df.dropna(subset=['_ts'])
            mask = (df['_ts'] >= data_inicio) & (df['_ts'] <= data_fim)
            df_filtered = df.loc[mask].copy()

            df_filtered['submercado_norm'] = df_filtered[
                col_mapping['submercado']
            ].apply(self._normalize_submercado)

            if 'carga' in col_mapping:
                df_filtered['carga_mw'] = df_filtered[col_mapping['carga']].apply(
                    self._to_float
                )

            frames.append(df_filtered)

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------------
    # Dataset consolidado (balanço de energia por subsistema)
    # ------------------------------------------------------------------
    def get_balanco_energia(self, data_inicio: datetime,
                             data_fim: datetime) -> pd.DataFrame:
        """
        Busca o dataset de balanço de energia por subsistema, que
        normalmente já traz geração por fonte, carga e intercâmbio
        consolidados em uma única tabela horária - útil como fallback
        quando os datasets individuais de fonte não estão disponíveis.
        """
        frames = []
        anos = range(data_inicio.year, data_fim.year + 1)

        for ano in anos:
            df = self._download_ons_csv(self.DATASETS['balanco_energia'], ano)
            if df.empty:
                continue

            col_mapping = self._detect_columns(df)
            if not col_mapping:
                continue

            df = df.copy()
            df['_ts'] = pd.to_datetime(df[col_mapping['timestamp']], errors='coerce')
            df = df.dropna(subset=['_ts'])
            mask = (df['_ts'] >= data_inicio) & (df['_ts'] <= data_fim)
            df_filtered = df.loc[mask].copy()

            df_filtered['submercado_norm'] = df_filtered[
                col_mapping['submercado']
            ].apply(self._normalize_submercado)

            frames.append(df_filtered)

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------------
    # Utilitários
    # ------------------------------------------------------------------
    @staticmethod
    def _to_float(value) -> float:
        """Converte valores do ONS (que às vezes usam vírgula decimal) para float"""
        if pd.isna(value):
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        value_str = str(value).strip().replace('.', '').replace(',', '.') \
            if ',' in str(value) and '.' in str(value) else str(value).replace(',', '.')
        try:
            return float(value_str)
        except ValueError:
            return 0.0

    @staticmethod
    def _normalize_submercado(valor) -> str:
        """
        Normaliza identificação de submercado para padrão único: N, NE, S, SE.
        O ONS usa variações como 'Norte', 'NORTE', '1', 'N', etc.
        """
        if pd.isna(valor):
            return "DESCONHECIDO"

        valor_str = str(valor).strip().upper()

        mapping = {
            'NORTE': 'N', 'N': 'N', '1': 'N',
            'NORDESTE': 'NE', 'NE': 'NE', '2': 'NE',
            'SUL': 'S', 'S': 'S', '3': 'S',
            'SUDESTE': 'SE', 'SE': 'SE', 'SUDESTE/CENTRO-OESTE': 'SE',
            'SECO': 'SE', '4': 'SE',
        }

        return mapping.get(valor_str, valor_str)

    def clear_cache(self, dataset_name: Optional[str] = None,
                     year: Optional[int] = None):
        """Limpa cache local, por dataset/ano específico ou tudo"""
        if dataset_name and year:
            path = self._get_cache_path(dataset_name, year)
            if path.exists():
                path.unlink()
                logger.info(f"Cache removido: {path}")
        elif dataset_name:
            for f in self.cache_dir.glob(f"{dataset_name}_*.csv"):
                f.unlink()
                logger.info(f"Cache removido: {f}")
        else:
            for f in self.cache_dir.glob("*.csv"):
                f.unlink()
            logger.info("Todo o cache foi limpo")


class ONSDatasetUnavailableError(Exception):
    """Levantada quando nenhum dataset pôde ser baixado/parseado para o período"""
    pass