# data_collectors/ccee_collector.py
"""
Coletor de dados abertos da CCEE (Câmara de Comercialização de Energia
Elétrica) - fonte oficial do PLD (Preço de Liquidação das Diferenças).

A CCEE é a entidade responsável pelo cálculo e publicação oficial do
PLD, sendo portanto a fonte primária recomendada para esse dado.

Implementa o mesmo contrato de interface do ONSDataCollector para
PLD, permitindo substituição transparente no pipeline.
"""

import io
import re
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional
import logging
import time

from config.settings import Settings
from data_collectors.http_utils import use_system_certificates
from models.data_models import RegistroPLD

logger = logging.getLogger(__name__)


class CCEEDatasetUnavailableError(Exception):
    """Levantada quando nenhum dataset da CCEE pôde ser localizado/baixado"""
    pass


class CCEEAccessDeniedError(CCEEDatasetUnavailableError):
    """Levantada quando a CCEE bloqueia o acesso ao catálogo ou ao recurso."""


class CCEEDataCollector:
    """
    Coletor de dados de PLD horário da CCEE, com cache local,
    retry automático e fallback de ano (quando o ano solicitado
    ainda não foi publicado).
    """

    def __init__(self,
                 cache_dir: str = None,
                 timeout_segundos: int = 30,
                 max_retries: int = 3):
        self.cache_dir = Path(cache_dir or Settings.CACHE_DIR) / "ccee"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_segundos = timeout_segundos
        self.max_retries = max_retries

        self._session = requests.Session()
        use_system_certificates(self._session)
        self._session.headers.update({
            "User-Agent": "BESS-Arbitrage-Analyzer/1.0 (+contato@seuprojeto.com)"
        })

    # ------------------------------------------------------------------
    # Resolução robusta de ano (núcleo da correção do bug)
    # ------------------------------------------------------------------
    def _resolver_ano_disponivel(self, ano_desejado: int) -> int:
        """
        Verifica se o dataset do ano desejado está disponível; caso
        não esteja (ex.: ano vigente ainda sem publicação completa),
        tenta recursivamente anos anteriores até
        MAX_TENTATIVAS_FALLBACK_ANO vezes.

        Isso evita que o pipeline quebre simplesmente porque o ano
        calendário atual ainda não teve seus dados consolidados
        publicados pela CCEE (defasagem típica de dias/semanas).
        """
        ano_atual = ano_desejado

        for tentativa in range(Settings.MAX_TENTATIVAS_FALLBACK_ANO + 1):
            if ano_atual < Settings.ANO_MINIMO_HISTORICO:
                raise CCEEDatasetUnavailableError(
                    f"Nenhum dataset de PLD disponível entre "
                    f"{ano_desejado} e {Settings.ANO_MINIMO_HISTORICO}"
                )

            if self._verificar_disponibilidade(ano_atual):
                if ano_atual != ano_desejado:
                    logger.warning(
                        f"Dataset de PLD para {ano_desejado} não disponível. "
                        f"Utilizando fallback para o ano {ano_atual}."
                    )
                return ano_atual

            ano_atual -= 1

        raise CCEEDatasetUnavailableError(
            f"Dataset de PLD não encontrado após "
            f"{Settings.MAX_TENTATIVAS_FALLBACK_ANO} tentativas de fallback "
            f"a partir de {ano_desejado}"
        )

    def _verificar_disponibilidade(self, ano: int) -> bool:
        """Consulta o catálogo CKAN oficial da CCEE."""
        try:
            return bool(self._listar_recursos_ckan(ano))
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 403:
                raise CCEEAccessDeniedError(
                    "A CCEE bloqueou o acesso ao catálogo (HTTP 403). "
                    "Verifique a política de acesso HTTPS da rede."
                ) from e
            logger.warning(f"Falha ao consultar o catálogo CCEE para {ano}: {e}")
            return False
        except requests.RequestException as e:
            logger.warning(f"Falha de conexão com o catálogo CCEE para {ano}: {e}")
            return False

    def _verificar_disponibilidade_ckan(self, ano: int) -> bool:
        """Mantido para compatibilidade com chamadas existentes."""
        return bool(self._listar_recursos_ckan(ano))

    def _listar_recursos_ckan(self, ano: int) -> list:
        """Retorna recursos CSV do ano solicitado no dataset oficial da CCEE."""
        url = f"{Settings.CCEE_CKAN_API_URL}/package_show"
        params = {"id": Settings.CCEE_CKAN_DATASET_PLD}

        resp = self._session.get(url, params=params, timeout=self.timeout_segundos)
        resp.raise_for_status()

        data = resp.json()
        if not data.get("success"):
            raise CCEEDatasetUnavailableError(
                f"Dataset CKAN '{Settings.CCEE_CKAN_DATASET_PLD}' "
                f"não encontrado: {data.get('error', 'resposta sem detalhes')}"
            )

        recursos = data.get("result", {}).get("resources", [])
        recursos_csv = [
            recurso for recurso in recursos
            if str(recurso.get("format", "")).upper() == "CSV"
            and recurso.get("url")
        ]
        recursos_do_ano = [
            recurso for recurso in recursos_csv
            if re.search(
                rf"(?<!\d){ano}(?!\d)",
                f"{recurso.get('name', '')} {recurso.get('url', '')}",
            )
        ]

        if recursos_do_ano:
            return recursos_do_ano
        if len(recursos_csv) == 1 and not re.search(
            r"(?<!\d)20\d{2}(?!\d)",
            f"{recursos_csv[0].get('name', '')} {recursos_csv[0].get('url', '')}",
        ):
            return recursos_csv
        return []

    # ------------------------------------------------------------------
    # Download com retry e cache local
    # ------------------------------------------------------------------
    def _baixar_dataset(self, ano: int) -> pd.DataFrame:
        """Baixa (ou recupera do cache) o dataset de PLD horário de um ano"""
        cache_path = self.cache_dir / f"pld_horario_{ano}.csv"

        if cache_path.exists():
            idade_cache_horas = (
                time.time() - cache_path.stat().st_mtime
            ) / 3600
            # Cache do ano vigente expira em 6h (dados mudam);
            # anos anteriores (já consolidados) ficam em cache indefinidamente
            ano_vigente = ano == datetime.now().year
            ttl_horas = 6 if ano_vigente else 24 * 30

            if idade_cache_horas < ttl_horas:
                logger.info(f"Usando cache local para PLD {ano}")
                return pd.read_csv(cache_path, sep=None, engine='python')

        recursos = self._listar_recursos_ckan(ano)
        if not recursos:
            raise CCEEDatasetUnavailableError(
                f"Nenhum recurso CSV de PLD foi encontrado no catálogo "
                f"da CCEE para {ano}"
            )

        dataframes = []
        ultimo_erro = None
        for recurso in recursos:
            url = recurso["url"]
            for tentativa in range(1, self.max_retries + 1):
                try:
                    logger.info(
                        f"Baixando PLD {ano} da CCEE "
                        f"(tentativa {tentativa}/{self.max_retries}): {url}"
                    )
                    resp = self._session.get(url, timeout=self.timeout_segundos)
                    resp.raise_for_status()
                    dataframes.append(pd.read_csv(
                        io.BytesIO(resp.content), sep=None, engine="python"
                    ))
                    break
                except requests.HTTPError as e:
                    if e.response is not None and e.response.status_code == 403:
                        raise CCEEAccessDeniedError(
                            "A CCEE bloqueou o download do recurso (HTTP 403). "
                            "Verifique a política de acesso HTTPS da rede."
                        ) from e
                    ultimo_erro = e
                    logger.warning(f"Falha ao baixar recurso CCEE: {e}")
                    if e.response is not None and e.response.status_code == 404:
                        break
                except requests.RequestException as e:
                    ultimo_erro = e
                    logger.warning(
                        f"Falha na tentativa {tentativa} para {url}: {e}"
                    )

                if tentativa < self.max_retries:
                    time.sleep(2 ** tentativa)

        if not dataframes:
            raise CCEEDatasetUnavailableError(
                f"Não foi possível baixar PLD de {ano} da CCEE. "
                f"Último erro: {ultimo_erro}"
            )

        df = pd.concat(dataframes, ignore_index=True)
        df.to_csv(cache_path, index=False)
        logger.info(f"Download concluído: {len(df)} linhas")
        return df

    # ------------------------------------------------------------------
    # Parsing e normalização (schema da CCEE difere do ONS)
    # ------------------------------------------------------------------
    def _normalizar_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normaliza o schema do arquivo da CCEE para o formato interno
        esperado pelos modelos de dados do projeto.

        A CCEE historicamente usa nomes de coluna em português com
        variações (ex.: 'Submercado', 'SUBMERCADO', 'subsistema'),
        então aplicamos detecção flexível de colunas.
        """
        df = df.copy()
        df.columns = [c.strip().lower() for c in df.columns]

        mapeamento_colunas = {
            'din_instante': 'timestamp',
            'data': 'timestamp',
            'data_hora': 'timestamp',
            'datahora': 'timestamp',
            'instante': 'timestamp',
            'nom_submercado': 'submercado',
            'submercado': 'submercado',
            'subsistema': 'submercado',
            'val_pld': 'pld_mwh',
            'pld': 'pld_mwh',
            'pld_hora': 'pld_mwh',
            'preco': 'pld_mwh',
            'val_cargaglobal': 'carga_mw',
            'carga': 'carga_mw',
        }

        colunas_renomeadas = {}
        for col in df.columns:
            if col in mapeamento_colunas:
                colunas_renomeadas[col] = mapeamento_colunas[col]

        df = df.rename(columns=colunas_renomeadas)

        if 'timestamp' not in df.columns and {
            'mes_referencia', 'dia', 'hora'
        }.issubset(df.columns):
            referencia = df['mes_referencia'].astype(str).str.extract(
                r'^(\d{4})\D?(\d{2})'
            )
            dia_numero = pd.to_numeric(df['dia'], errors='coerce')

            if referencia.notna().all().all() and dia_numero.notna().all():
                datas = pd.to_datetime({
                    'year': pd.to_numeric(referencia[0], errors='coerce'),
                    'month': pd.to_numeric(referencia[1], errors='coerce'),
                    'day': dia_numero,
                }, errors='coerce')
            else:
                datas = pd.to_datetime(df['dia'], dayfirst=True, errors='coerce')

            horas = pd.to_numeric(
                df['hora'].astype(str).str.extract(r'^(\d{1,2})')[0],
                errors='coerce',
            )
            df['timestamp'] = datas + pd.to_timedelta(horas, unit='h')

        colunas_obrigatorias = {'timestamp', 'submercado', 'pld_mwh'}
        colunas_faltantes = colunas_obrigatorias - set(df.columns)

        if colunas_faltantes:
            raise ValueError(
                f"Schema inesperado no dataset CCEE. "
                f"Colunas faltantes: {colunas_faltantes}. "
                f"Colunas disponíveis: {list(df.columns)}"
            )

        if 'carga_mw' not in df.columns:
            df['carga_mw'] = np.nan

        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])

        df['submercado'] = self._normalizar_submercado(df['submercado'])

        df['pld_mwh'] = pd.to_numeric(
            df['pld_mwh'].astype(str).str.replace(',', '.'), errors='coerce'
        )
        df = df.dropna(subset=['pld_mwh'])

        return df[['timestamp', 'submercado', 'pld_mwh', 'carga_mw']]

    @staticmethod
    def _normalizar_submercado(serie: pd.Series) -> pd.Series:
        """Normaliza nomes de submercado para o padrão N/NE/S/SE usado no projeto"""
        mapeamento = {
            'NORTE': 'N', 'N': 'N',
            'NORDESTE': 'NE', 'NE': 'NE',
            'SUL': 'S', 'S': 'S',
            'SUDESTE': 'SE', 'SUDESTE/CENTRO-OESTE': 'SE', 'SECO': 'SE',
            'SE': 'SE', 'SE/CO': 'SE',
        }
        serie_normalizada = serie.astype(str).str.strip().str.upper()
        return serie_normalizada.map(mapeamento).fillna(serie_normalizada)

    # ------------------------------------------------------------------
    # MÉTODO PÚBLICO PRINCIPAL
    # ------------------------------------------------------------------
    def get_pld_horario(self,
                          data_inicio: datetime,
                          data_fim: datetime) -> List[RegistroPLD]:
        """
        Obtém registros de PLD horário da CCEE para o período solicitado,
        com fallback automático de ano quando necessário.

        Args:
            data_inicio: data inicial do período (inclusive)
            data_fim: data final do período (inclusive)

        Returns:
            Lista de RegistroPLD ordenada por timestamp
        """
        anos_necessarios = range(data_inicio.year, data_fim.year + 1)
        dataframes = []

        for ano in anos_necessarios:
            try:
                ano_resolvido = self._resolver_ano_disponivel(ano)
                df_bruto = self._baixar_dataset(ano_resolvido)
                df_normalizado = self._normalizar_dataframe(df_bruto)
                dataframes.append(df_normalizado)

            except CCEEDatasetUnavailableError as e:
                if isinstance(e, CCEEAccessDeniedError):
                    raise
                logger.error(f"Dataset indisponível para ano {ano}: {e}")
                continue
            except ValueError as e:
                logger.error(f"Erro ao normalizar dataset de {ano}: {e}")
                continue

        if not dataframes:
            raise CCEEDatasetUnavailableError(
                f"Nenhum dado de PLD disponível na CCEE para o período "
                f"{data_inicio.date()} a {data_fim.date()}"
            )

        df_final = pd.concat(dataframes, ignore_index=True)

        mask_periodo = (
            (df_final['timestamp'] >= pd.Timestamp(data_inicio)) &
            (df_final['timestamp'] <= pd.Timestamp(data_fim))
        )
        df_final = df_final[mask_periodo].sort_values('timestamp')

        registros = [
            RegistroPLD(
                timestamp=row['timestamp'],
                submercado=row['submercado'],
                pld_mwh=row['pld_mwh'],
                carga_mw=row['carga_mw'] if not pd.isna(row['carga_mw']) else 0.0,
            )
            for _, row in df_final.iterrows()
        ]

        logger.info(
            f"CCEE: {len(registros)} registros de PLD coletados para "
            f"{data_inicio.date()} - {data_fim.date()}"
        )

        return registros

        # data_collectors/ccee_collector.py (métodos auxiliares finais)

    def get_pld_horario_df(self,
                             data_inicio: datetime,
                             data_fim: datetime) -> pd.DataFrame:
        """
        Variante que retorna DataFrame diretamente (útil para o
        dashboard, que já trabalha com DataFrames em várias camadas
        de visualização sem precisar reconverter a partir de dataclasses).
        """
        registros = self.get_pld_horario(data_inicio, data_fim)
        if not registros:
            return pd.DataFrame(columns=['timestamp', 'submercado', 'pld_mwh', 'carga_mw'])

        return pd.DataFrame([{
            'timestamp': r.timestamp,
            'submercado': r.submercado,
            'pld_mwh': r.pld_mwh,
            'carga_mw': r.carga_mw,
        } for r in registros])

    def limpar_cache(self, ano: Optional[int] = None):
        """Remove arquivos de cache local, opcionalmente filtrando por ano"""
        if ano:
            arquivo = self.cache_dir / f"pld_horario_{ano}.csv"
            if arquivo.exists():
                arquivo.unlink()
                logger.info(f"Cache removido: {arquivo}")
        else:
            for arquivo in self.cache_dir.glob("pld_horario_*.csv"):
                arquivo.unlink()
            logger.info("Todo o cache de PLD da CCEE foi removido")