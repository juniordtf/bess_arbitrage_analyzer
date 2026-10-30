# data_collectors/ons_collector.py
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import logging
from pathlib import Path
import time
import unicodedata
import re

from config.settings import Settings
from data_collectors.http_utils import use_system_certificates
from models.data_models import RegistroGeracao, RegistroPLD, RestricaoCOFF, RazaoRestricaoCOFF, TipoFonte

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
        use_system_certificates(self.session)
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
                           max_retries: int = 3,
                           url: Optional[str] = None,
                           cache_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Baixa CSV do ONS com retry e fallback de encoding.
        Padrão de URL: {BASE}/{dataset}/{DATASET}_{ANO}.csv
        """
        cache_path = cache_path or self._get_cache_path(dataset_name, year)

        if cache_path.exists() and not force_refresh:
            logger.info(f"Carregando do cache: {cache_path}")
            return self._safe_read_csv(cache_path)

        url = url or (
            f"{Settings.ONS_BASE_URL}/{dataset_name}/"
            f"{dataset_name.upper()}_{year}.csv"
        )

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
    def get_geracao_por_fonte(self,
                                fonte: TipoFonte,
                                data_inicio: datetime,
                                data_fim: datetime) -> List[RegistroGeracao]:
        """Busca geração por fonte usando os CSVs atuais do ONS."""
        tipos_usina = {
            TipoFonte.EOLICA: {"EOLICA", "EOLIELETRICA"},
            TipoFonte.SOLAR: {"FOTOVOLTAICA"},
            TipoFonte.HIDRAULICA: {"HIDROELETRICA"},
            TipoFonte.TERMICA: {"TERMICA"},
        }
        tipo_usina = tipos_usina.get(fonte)
        if tipo_usina is None:
            logger.warning(f"Fonte {fonte.value} não suportada pelo dataset do ONS")
            return []

        registros = []
        for ano in range(data_inicio.year, data_fim.year + 1):
            if ano < 2022:
                periodos = [(None, None)]
            else:
                primeiro_mes = data_inicio.month if ano == data_inicio.year else 1
                ultimo_mes = data_fim.month if ano == data_fim.year else 12
                periodos = [(mes, mes) for mes in range(primeiro_mes, ultimo_mes + 1)]

            for mes, _ in periodos:
                if mes is None:
                    arquivo = f"GERACAO_USINA-2_{ano}.csv"
                    cache_path = self.cache_dir / f"geracao_usina_2_ho_{ano}.csv"
                else:
                    arquivo = f"GERACAO_USINA-2_{ano}_{mes:02d}.csv"
                    cache_path = (
                        self.cache_dir / f"geracao_usina_2_ho_{ano}_{mes:02d}.csv"
                    )

                url = (
                    f"{Settings.ONS_BASE_URL}/geracao_usina_2_ho/{arquivo}"
                )
                df = self._download_ons_csv(
                    "geracao_usina_2_ho",
                    ano,
                    url=url,
                    cache_path=cache_path,
                )
                if df.empty:
                    continue

                type_column = next(
                    (
                        column for column in df.columns
                        if column.strip().lower() == "nom_tipousina"
                    ),
                    None,
                )
                if type_column is None:
                    logger.warning(
                        f"Coluna nom_tipousina ausente no dataset de geração {ano}"
                    )
                    continue

                normalized_types = df[type_column].astype(str).map(
                    lambda value: unicodedata.normalize("NFKD", value)
                    .encode("ascii", "ignore")
                    .decode("ascii")
                    .strip()
                    .upper()
                )
                df_fonte = df.loc[normalized_types.isin(tipo_usina)]
                registros.extend(
                    self._parse_geracao_df(df_fonte, fonte, data_inicio, data_fim)
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
        O ONS não publica mais o dataset de PLD horário em seu catálogo.
        O PLD deve ser obtido pela fonte oficial da CCEE.
        """
        raise ONSDatasetUnavailableError(
            "O ONS não disponibiliza PLD horário no catálogo atual. "
            "Obtenha os dados pela CCEE."
        )

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

    _COFF_DATASET_IDS = {
        TipoFonte.EOLICA: "restricao_coff_eolica_detail",
        TipoFonte.SOLAR: "restricao_coff_fotovoltaica_detail",
    }
    _COFF_CATALOG_URL = "https://dados.ons.org.br/api/3/action/package_show"
    _COFF_COLUMNS = {
        "din_instante", "data_hora", "timestamp",
        "nom_usina", "nom_conjuntousina", "usina",
        "id_subsistema", "nom_subsistema", "submercado",
        "flg_geracaorestrita", "val_geracaoestimada",
        "val_geracaoverificada", "val_geracaoreferenciafinal",
        "val_geracaolimitada", "val_potenciarestringida",
        "potencia_restringida", "val_geracaodisponivel",
        "nom_razaorestricao", "razao_restricao",
        "cod_razaorestricao", "nom_motivo",
    }

    def get_restricao_coff(self,
                              fonte: TipoFonte,
                              data_inicio: datetime,
                              data_fim: datetime) -> List[RestricaoCOFF]:
        """
        Obtém dados REAIS de restrição de geração (curtailment) do
        ONS, com razão de corte classificada oficialmente
        (Energética, Elétrica, Confiabilidade).

        Esta é a fonte PRIMÁRIA e correta para medir curtailment -
        o uso de PLD como proxy (ver CurtailmentAnalyzer.detect_oversupply_events)
        deve ser tratado apenas como sinal complementar de contexto
        econômico, nunca como substituto deste dado.

        Args:
            fonte: TipoFonte.EOLICA ou TipoFonte.SOLAR (únicas fontes
                   com dataset de restrição COFF público no momento)
            data_inicio: início do período
            data_fim: fim do período

        Returns:
            Lista de RestricaoCOFF com granularidade por usina/hora
        """
        if fonte not in (TipoFonte.EOLICA, TipoFonte.SOLAR):
            logger.warning(
                f"Dataset de restrição COFF não disponível publicamente "
                f"para a fonte {fonte.value}. Retornando lista vazia."
            )
            return []

        recursos = self._listar_recursos_coff(fonte)
        registros = []
        mes_atual = datetime(data_inicio.year, data_inicio.month, 1)
        mes_final = datetime(data_fim.year, data_fim.month, 1)

        while mes_atual <= mes_final:
            url = recursos.get((mes_atual.year, mes_atual.month))
            if url is None:
                logger.warning(
                    f"Recurso COFF {fonte.value} não publicado para "
                    f"{mes_atual:%Y-%m}"
                )
            else:
                caminho = self._baixar_dataset_coff(
                    fonte, mes_atual.year, mes_atual.month, url
                )
                leitor = pd.read_csv(
                    caminho,
                    sep=";",
                    usecols=lambda coluna: coluna.strip().lower() in self._COFF_COLUMNS,
                    chunksize=100_000,
                    low_memory=False,
                )
                for bloco in leitor:
                    registros.extend(
                        self._converter_para_registros_coff(
                            bloco, fonte, data_inicio, data_fim
                        )
                    )

            if mes_atual.month == 12:
                mes_atual = datetime(mes_atual.year + 1, 1, 1)
            else:
                mes_atual = datetime(mes_atual.year, mes_atual.month + 1, 1)

        logger.info(
            f"COFF {fonte.value}: {len(registros)} registros de restrição "
            f"no período {data_inicio.date()} - {data_fim.date()}"
        )
        return registros

    def _listar_recursos_coff(self, fonte: TipoFonte) -> Dict[tuple, str]:
        """Descobre no CKAN os CSVs mensais publicados para a fonte."""
        dataset_id = self._COFF_DATASET_IDS[fonte]
        resposta = self.session.get(
            self._COFF_CATALOG_URL,
            params={"id": dataset_id},
            timeout=60,
        )
        resposta.raise_for_status()
        payload = resposta.json()
        if not payload.get("success"):
            raise ONSDatasetUnavailableError(
                f"Catálogo CKAN do ONS não encontrou {dataset_id}: "
                f"{payload.get('error', 'resposta sem detalhes')}"
            )

        recursos = {}
        for recurso in payload.get("result", {}).get("resources", []):
            if str(recurso.get("format", "")).upper() != "CSV":
                continue
            url = recurso.get("url")
            if not url:
                continue
            correspondencia = re.search(
                r"_(20\d{2})_(0[1-9]|1[0-2])\.csv(?:$|\?)",
                url,
                flags=re.IGNORECASE,
            )
            if correspondencia:
                ano, mes = map(int, correspondencia.groups())
                recursos[(ano, mes)] = url
        return recursos

    def _baixar_dataset_coff(self,
                               fonte: TipoFonte,
                               ano: int,
                               mes: int,
                               url: str) -> Path:
        """Baixa (ou recupera do cache) o CSV mensal de restrição COFF."""
        cache_path = self.cache_dir / f"coff_{fonte.value.lower()}_{ano}_{mes:02d}.csv"

        if cache_path.exists():
            idade_horas = (time.time() - cache_path.stat().st_mtime) / 3600
            mes_vigente = (ano, mes) == (datetime.now().year, datetime.now().month)
            ttl_horas = 12 if mes_vigente else 24 * 30

            if idade_horas < ttl_horas:
                logger.info(f"Usando cache local para COFF {fonte.value} {ano}")
                return cache_path

        caminho_temporario = cache_path.with_suffix(".csv.part")
        for tentativa in range(1, 4):
            try:
                logger.info(
                    f"Baixando restrição COFF {fonte.value} {ano}-{mes:02d} "
                    f"(tentativa {tentativa}/3)"
                )
                with self.session.get(url, timeout=180, stream=True) as resp:
                    resp.raise_for_status()
                    with caminho_temporario.open("wb") as arquivo:
                        for bloco in resp.iter_content(chunk_size=1024 * 1024):
                            if bloco:
                                arquivo.write(bloco)
                caminho_temporario.replace(cache_path)
                return cache_path
            except requests.RequestException as e:
                logger.warning(f"Falha na tentativa {tentativa}: {e}")
                caminho_temporario.unlink(missing_ok=True)
                if tentativa == 3:
                    raise ONSDatasetUnavailableError(
                        f"Não foi possível baixar restrição COFF {fonte.value} "
                        f"{ano}-{mes:02d}: {e}"
                    ) from e
                time.sleep(2 ** tentativa)

        raise ONSDatasetUnavailableError(
            f"Não foi possível baixar restrição COFF {fonte.value} {ano}-{mes:02d}"
        )

    def _converter_para_registros_coff(self,
                                          df: pd.DataFrame,
                                          fonte: TipoFonte,
                                          data_inicio: datetime,
                                          data_fim: datetime) -> List[RestricaoCOFF]:
        """
        Normaliza variantes do schema COFF, incluindo os CSVs mensais
        atuais com geração estimada/verificada e flag de restrição.
        """
        df = df.copy()
        df.columns = [c.strip().lower() for c in df.columns]

        def find_column(*nomes):
            return next((nome for nome in nomes if nome in df.columns), None)

        timestamp_col = find_column("din_instante", "data_hora", "timestamp")
        usina_col = find_column("nom_usina", "nom_conjuntousina", "usina")
        submercado_col = find_column(
            "id_subsistema", "nom_subsistema", "submercado"
        )
        if not timestamp_col or not usina_col or not submercado_col:
            raise ValueError(
                f"Schema COFF inesperado para {fonte.value}: "
                f"colunas disponíveis {list(df.columns)}"
            )

        df["timestamp"] = pd.to_datetime(df[timestamp_col], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df = df[
            (df["timestamp"] >= pd.Timestamp(data_inicio))
            & (df["timestamp"] <= pd.Timestamp(data_fim))
        ].copy()
        if df.empty:
            return []

        flag_col = find_column("flg_geracaorestrita")
        if flag_col:
            flags = df[flag_col].astype(str).str.strip().str.upper().isin(
                {"1", "S", "SIM", "TRUE", "T"}
            )
        else:
            flags = pd.Series(True, index=df.index)

        estimada_col = find_column(
            "val_geracaoestimada", "val_geracaoreferenciafinal"
        )
        verificada_col = find_column("val_geracaoverificada")
        limitada_col = find_column(
            "val_geracaolimitada", "val_potenciarestringida",
            "potencia_restringida",
        )
        disponivel_col = find_column("val_geracaodisponivel")
        if estimada_col and verificada_col:
            potencia_restringida = (
                pd.to_numeric(df[estimada_col], errors="coerce")
                - pd.to_numeric(df[verificada_col], errors="coerce")
            ).clip(lower=0)
        elif disponivel_col and limitada_col:
            potencia_restringida = (
                pd.to_numeric(df[disponivel_col], errors="coerce")
                - pd.to_numeric(df[limitada_col], errors="coerce")
            ).clip(lower=0)
        elif limitada_col:
            potencia_restringida = pd.to_numeric(
                df[limitada_col], errors="coerce"
            ).clip(lower=0)
        else:
            raise ValueError(
                f"Schema COFF inesperado para {fonte.value}: "
                "sem colunas de geração para calcular a potência restringida"
            )

        df["potencia_restringida_mw"] = potencia_restringida
        df = df[flags & df["potencia_restringida_mw"].gt(0)].copy()
        if df.empty:
            return []

        razao_col = find_column(
            "nom_razaorestricao", "razao_restricao",
            "cod_razaorestricao", "nom_motivo",
        )
        if razao_col:
            df["razao_normalizada"] = self._normalizar_razao_restricao(
                df[razao_col]
            )
        else:
            df["razao_normalizada"] = RazaoRestricaoCOFF.OUTRA.value

        geracao_disponivel = (
            pd.to_numeric(df[estimada_col], errors="coerce")
            if estimada_col else (
                pd.to_numeric(df[disponivel_col], errors="coerce")
                if disponivel_col else pd.Series(np.nan, index=df.index)
            )
        )

        registros = []
        for indice, row in df.iterrows():
            registros.append(RestricaoCOFF(
                timestamp=row["timestamp"],
                usina=str(row[usina_col]),
                submercado=self._normalize_submercado(row[submercado_col]),
                fonte=fonte,
                potencia_restringida_mw=float(row["potencia_restringida_mw"]),
                razao_restricao=RazaoRestricaoCOFF(row["razao_normalizada"]),
                geracao_disponivel_mw=(
                    float(geracao_disponivel.loc[indice])
                    if pd.notna(geracao_disponivel.loc[indice]) else None
                ),
                intervalo_horas=0.5,
            ))
        return registros

    @staticmethod
    def _normalizar_razao_restricao(serie: pd.Series) -> pd.Series:
        """
        Normaliza os valores de razão de restrição publicados pelo ONS
        (que variam em capitalização/acentuação ao longo dos anos) para
        o enum interno RazaoRestricaoCOFF.
        """
        mapeamento = {
            'ENERGETICA': RazaoRestricaoCOFF.ENERGETICA.value,
            'ENERGÉTICA': RazaoRestricaoCOFF.ENERGETICA.value,
            'RESTRICAO ENERGETICA': RazaoRestricaoCOFF.ENERGETICA.value,
            'ELETRICA': RazaoRestricaoCOFF.ELETRICA.value,
            'ELÉTRICA': RazaoRestricaoCOFF.ELETRICA.value,
            'RESTRICAO ELETRICA': RazaoRestricaoCOFF.ELETRICA.value,
            'RESTRICAO DE TRANSMISSAO': RazaoRestricaoCOFF.ELETRICA.value,
            'CONFIABILIDADE': RazaoRestricaoCOFF.CONFIABILIDADE.value,
            'SEGURANCA': RazaoRestricaoCOFF.CONFIABILIDADE.value,
            'SEGURANÇA': RazaoRestricaoCOFF.CONFIABILIDADE.value,
        }

        serie_normalizada = (
            serie.astype(str).str.strip().str.upper()
            .str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
        )

        return serie_normalizada.map(
            lambda v: mapeamento.get(v, RazaoRestricaoCOFF.OUTRA.value)
        )

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