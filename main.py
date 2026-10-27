# main.py
"""
Pipeline principal de análise: Mapeamento de Restrições de Geração e
Identificação de Oportunidades de BESS para Arbitragem de Energia.

Fluxo de execução:
1. Coleta dados do ONS (geração por fonte + PLD horário)
2. Coleta dados da ANEEL (perfil de capacidade + saturação de BESS)
3. Executa análise de curtailment/excesso de geração por submercado
4. Executa análise de spread de preços (PLD)
5. Consolida oportunidades de BESS com dimensionamento e viabilidade
6. Gera relatórios (JSON, CSV) e sumário no console

Uso:
    python main.py --dias 90 --output ./output
    python main.py --submercado SE --dias 180 --min-score 40
    python main.py --use-cache-only  # não baixa dados novos, usa apenas cache local
"""

import argparse
import logging
import sys
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd

from config.settings import Settings
from models.data_models import RegistroGeracao, RegistroPLD, TipoFonte
from data_collectors.ons_collector import ONSDataCollector
from data_collectors.aneel_collector import ANEELDataCollector
from analysis.curtailment_analyzer import CurtailmentAnalyzer
from analysis.price_spread_analyzer import PriceSpreadAnalyzer
from analysis.bess_opportunity_finder import BESSOpportunityFinder


# ======================================================================
# CONFIGURAÇÃO DE LOGGING
# ======================================================================
def setup_logging(verbose: bool = False, log_file: Optional[str] = None):
    """Configura logging para console e opcionalmente arquivo"""
    log_level = logging.DEBUG if verbose else logging.INFO

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers,
    )

    # Reduz verbosidade de bibliotecas de rede
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)


logger = logging.getLogger("bess_analyzer")


# ======================================================================
# ARGUMENTOS DE LINHA DE COMANDO
# ======================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mapeamento de restrições de geração e oportunidades de BESS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        '--dias', type=int, default=Settings.JANELA_ANALISE_DIAS,
        help="Número de dias retroativos para análise"
    )
    parser.add_argument(
        '--data-inicio', type=str, default=None,
        help="Data de início (YYYY-MM-DD). Sobrescreve --dias se fornecido."
    )
    parser.add_argument(
        '--data-fim', type=str, default=None,
        help="Data de fim (YYYY-MM-DD). Padrão: hoje."
    )
    parser.add_argument(
        '--submercado', type=str, default=None, choices=Settings.SUBMERCADOS,
        help="Filtra análise para um único submercado. Padrão: todos."
    )
    parser.add_argument(
        '--fontes', type=str, nargs='+',
        default=['EOLICA', 'SOLAR', 'HIDRAULICA', 'TERMICA'],
        choices=['EOLICA', 'SOLAR', 'HIDRAULICA', 'TERMICA'],
        help="Fontes de geração a analisar"
    )
    parser.add_argument(
        '--min-score', type=float, default=0.0,
        help="Score mínimo de atratividade para incluir no relatório final"
    )
    parser.add_argument(
        '--output', type=str, default=Settings.OUTPUT_DIR,
        help="Diretório de saída para relatórios"
    )
    parser.add_argument(
        '--cache-dir', type=str, default=Settings.CACHE_DIR,
        help="Diretório de cache local"
    )
    parser.add_argument(
        '--force-refresh', action='store_true',
        help="Força novo download, ignorando cache existente"
    )
    parser.add_argument(
        '--capex-mwh', type=float, default=None,
        help="CAPEX customizado por MWh (R$). Sobrescreve padrão de mercado."
    )
    parser.add_argument(
        '--capex-mw', type=float, default=None,
        help="CAPEX customizado por MW (R$). Sobrescreve padrão de mercado."
    )
    parser.add_argument(
        '--eficiencia', type=float, default=None,
        help="Eficiência round-trip do BESS (0-1)"
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help="Ativa logging detalhado (DEBUG)"
    )
    parser.add_argument(
        '--skip-aneel', action='store_true',
        help="Pula coleta ANEEL (usa apenas análise de curtailment/spread do ONS)"
    )
    parser.add_argument(
        '--sensitivity', action='store_true',
        help="Executa análise de sensibilidade para a melhor oportunidade"
    )

    return parser.parse_args()


# ======================================================================
# ETAPAS DO PIPELINE
# ======================================================================
def resolve_date_range(args: argparse.Namespace) -> tuple:
    """Resolve o período de análise a partir dos argumentos"""
    if args.data_fim:
        data_fim = datetime.strptime(args.data_fim, "%Y-%m-%d")
    else:
        data_fim = datetime.now()

    if args.data_inicio:
        data_inicio = datetime.strptime(args.data_inicio, "%Y-%m-%d")
    else:
        data_inicio = data_fim - timedelta(days=args.dias)

    return data_inicio, data_fim


def collect_generation_data(ons_collector: ONSDataCollector,
                              fontes: List[str],
                              data_inicio: datetime,
                              data_fim: datetime) -> List[RegistroGeracao]:
    """Coleta dados de geração para todas as fontes solicitadas"""
    registros_totais = []

    fonte_map = {
        'EOLICA': TipoFonte.EOLICA,
        'SOLAR': TipoFonte.SOLAR,
        'HIDRAULICA': TipoFonte.HIDRAULICA,
        'TERMICA': TipoFonte.TERMICA,
    }

    for fonte_str in fontes:
        fonte_enum = fonte_map[fonte_str]
        logger.info(f"Coletando geração {fonte_str}...")

        try:
            registros = ons_collector.get_geracao_por_fonte(
                fonte_enum, data_inicio, data_fim
            )
            registros_totais.extend(registros)
            logger.info(f"  -> {len(registros)} registros coletados para {fonte_str}")

        except Exception as e:
            logger.error(f"Erro ao coletar {fonte_str}: {e}", exc_info=True)
            continue

    return registros_totais


def collect_pld_data(ons_collector: ONSDataCollector,
                       data_inicio: datetime,
                       data_fim: datetime) -> List[RegistroPLD]:
    """Coleta dados de PLD horário"""
    logger.info("Coletando PLD horário...")
    try:
        registros = ons_collector.get_pld_horario(data_inicio, data_fim)
        logger.info(f"  -> {len(registros)} registros de PLD coletados")
        return registros
    except Exception as e:
        logger.error(f"Erro ao coletar PLD: {e}", exc_info=True)
        return []


def collect_aneel_profile(aneel_collector: ANEELDataCollector,
                            force_refresh: bool) -> Optional[pd.DataFrame]:
    """Coleta perfil de capacidade/saturação da ANEEL (opcional)"""
    logger.info("Coletando perfil de mercado ANEEL...")
    try:
        df_perfil = aneel_collector.build_submercado_profile(
            force_refresh=force_refresh
        )
        if df_perfil.empty:
            logger.warning("Perfil ANEEL vazio - análise seguirá sem fator de saturação")
            return None

        logger.info(f"  -> Perfil ANEEL coletado: {len(df_perfil)} registros")
        return df_perfil

    except Exception as e:
        logger.warning(f"Erro ao coletar perfil ANEEL (não crítico): {e}")
        return None


def filter_by_submercado(registros_geracao: List[RegistroGeracao],
                           registros_pld: List[RegistroPLD],
                           submercado: Optional[str]) -> tuple:
    """Filtra registros para um submercado específico, se solicitado"""
    if not submercado:
        return registros_geracao, registros_pld

    registros_geracao_filtrados = [
        r for r in registros_geracao if r.submercado == submercado
    ]
    registros_pld_filtrados = [
        r for r in registros_pld if r.submercado == submercado
    ]

    logger.info(
        f"Filtrado para submercado {submercado}: "
        f"{len(registros_geracao_filtrados)} registros de geração, "
        f"{len(registros_pld_filtrados)} registros de PLD"
    )

    return registros_geracao_filtrados, registros_pld_filtrados


def validate_data_sufficiency(registros_geracao: List[RegistroGeracao],
                                 registros_pld: List[RegistroPLD]) -> bool:
    """Valida se há dados suficientes para prosseguir com a análise"""
    if not registros_pld:
        logger.error(
            "Nenhum dado de PLD disponível. A análise de arbitragem "
            "requer dados de preço - abortando."
        )
        return False

    if not registros_geracao:
        logger.warning(
            "Nenhum dado de geração disponível. A análise seguirá "
            "apenas com excesso de geração via PLD (sem curtailment "
            "físico confirmado)."
        )

    submercados_pld = set(r.submercado for r in registros_pld)
    if len(submercados_pld) == 0:
        logger.error("Dados de PLD sem submercado identificável - abortando.")
        return False

    logger.info(f"Submercados disponíveis nos dados: {sorted(submercados_pld)}")
    return True


# ======================================================================
# EXIBIÇÃO DE RESULTADOS NO CONSOLE
# ======================================================================
def print_console_summary(resumo: dict):
    """Imprime sumário executivo formatado no console"""
    print("\n" + "=" * 80)
    print(" RESUMO EXECUTIVO - OPORTUNIDADES DE BESS PARA ARBITRAGEM DE ENERGIA")
    print("=" * 80)

    resumo_exec = resumo.get('resumo_executivo', {})

    if resumo_exec.get('total_oportunidades_identificadas', 0) == 0:
        print("\n⚠️  Nenhuma oportunidade identificada com os critérios atuais.")
        print("    Considere ajustar --min-score ou expandir o período de análise.\n")
        return

    print(f"\n📊 Total de oportunidades identificadas: "
          f"{resumo_exec['total_oportunidades_identificadas']}")
    print(f"🏆 Submercado mais atrativo: {resumo_exec['submercado_mais_atrativo']} "
          f"(score: {resumo_exec['score_melhor_oportunidade']:.1f}/100)")
    print(f"⚡ Energia total identificada: "
          f"{resumo_exec['energia_total_identificada_mwh']:,.0f} MWh")
    print(f"🔋 Capacidade total recomendada: "
          f"{resumo_exec['capacidade_total_recomendada_mwh']:,.0f} MWh")
    print(f"💰 Receita potencial anual (agregada): "
          f"R$ {resumo_exec['receita_total_potencial_anual_rs']:,.0f}")

    if resumo_exec.get('payback_medio_anos'):
        print(f"⏱️  Payback médio estimado: "
              f"{resumo_exec['payback_medio_anos']:.1f} anos")

    print("\n" + "-" * 80)
    print(" TOP 5 OPORTUNIDADES DETALHADAS")
    print("-" * 80)

    for i, op in enumerate(resumo.get('top_5_analise_detalhada', []), 1):
        print(f"\n[{i}] Submercado: {op['submercado']} | "
              f"Score: {op['score_atratividade']:.1f}/100")
        print(f"    Tipo de restrição predominante: {op['tipo_restricao']}")
        print(f"    Dimensionamento recomendado: "
              f"{op['dimensionamento']['capacidade_mwh']:.0f} MWh / "
              f"{op['dimensionamento']['potencia_mw']:.0f} MW "
              f"(C-rate: {op['dimensionamento']['c_rate']})")
        print(f"    Spread médio: R$ {op['operacional']['spread_medio_rs_mwh']:.2f}/MWh "
              f"(carga: R$ {op['operacional']['preco_medio_carga_rs_mwh']:.2f} | "
              f"descarga: R$ {op['operacional']['preco_medio_descarga_rs_mwh']:.2f})")
        print(f"    Ciclos anuais estimados: "
              f"{op['operacional']['ciclos_anuais_estimados']:.0f}")

        fin = op['financeiro']
        print(f"    CAPEX estimado: R$ {fin['capex_total_rs']:,.0f}")
        print(f"    VPL: R$ {fin['vpl_rs']:,.0f} | "
              f"TIR: {fin['tir_pct']:.1f}% " if fin.get('tir_pct') else
              f"    VPL: R$ {fin['vpl_rs']:,.0f} | TIR: N/A ")
        print(f"    Payback simples: "
              f"{fin['payback_simples_anos']:.1f} anos" if fin.get('payback_simples_anos')
              else "    Payback simples: N/A (fluxo negativo)")
        print(f"    Viável (VPL > 0): {'✅ SIM' if fin['viavel'] else '❌ NÃO'}")

    print("\n" + "=" * 80)
    print(f" Relatório completo salvo em: {resumo.get('_output_path', 'N/A')}")
    print("=" * 80 + "\n")


def print_sensitivity_summary(df_sensitivity: pd.DataFrame, submercado: str):
    """Imprime resumo da análise de sensibilidade"""
    print("\n" + "-" * 80)
    print(f" ANÁLISE DE SENSIBILIDADE - {submercado}")
    print("-" * 80)

    pivot_vpl = df_sensitivity.pivot(
        index='variacao_capex_pct',
        columns='variacao_spread_pct',
        values='vpl_rs'
    )

    print("\nVPL (R$) por cenário [linhas: Δ CAPEX%, colunas: Δ Spread%]:\n")
    print(pivot_vpl.to_string(float_format=lambda x: f"{x:,.0f}"))

    pct_viavel = (df_sensitivity['viavel'].sum() / len(df_sensitivity)) * 100
    print(f"\n% de cenários viáveis (VPL > 0): {pct_viavel:.0f}%")
    print("-" * 80 + "\n")


# ======================================================================
# EXPORTAÇÃO DE ARQUIVOS
# ======================================================================
def export_results(resumo: dict,
                     df_oportunidades: pd.DataFrame,
                     output_dir: str) -> dict:
    """Exporta relatório JSON e CSV, retorna caminhos gerados"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = output_path / f"relatorio_bess_{timestamp_str}.json"
    csv_path = output_path / f"oportunidades_bess_{timestamp_str}.csv"

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(resumo, f, ensure_ascii=False, indent=2, default=_json_serializer)

    if not df_oportunidades.empty:
        df_oportunidades.to_csv(csv_path, index=False, sep=';', encoding='utf-8-sig')

    logger.info(f"Relatório JSON salvo: {json_path}")
    logger.info(f"Oportunidades CSV salvas: {csv_path}")

    return {'json': str(json_path), 'csv': str(csv_path)}


def _json_serializer(obj):
    """Serializador customizado para tipos não nativos do JSON"""
    import numpy as np
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if pd.isna(obj):
        return None
    raise TypeError(f"Tipo não serializável: {type(obj)}")


# ======================================================================
# PIPELINE PRINCIPAL
# ======================================================================
def run_pipeline(args: argparse.Namespace) -> Optional[dict]:
    """Executa o pipeline completo de análise"""

    data_inicio, data_fim = resolve_date_range(args)
    logger.info(f"Período de análise: {data_inicio.date()} até {data_fim.date()}")

    # --- Inicialização dos coletores ---
    ons_collector = ONSDataCollector(cache_dir=args.cache_dir)
    aneel_collector = ANEELDataCollector(cache_dir=args.cache_dir)

    # --- ETAPA 1: Coleta de dados ---
    logger.info("=" * 60)
    logger.info("ETAPA 1/4: Coleta de dados ONS")
    logger.info("=" * 60)

    registros_geracao = collect_generation_data(
        ons_collector, args.fontes, data_inicio, data_fim
    )
    registros_pld = collect_pld_data(ons_collector, data_inicio, data_fim)

    if not validate_data_sufficiency(registros_geracao, registros_pld):
        return None

    df_perfil_aneel = None
    if not args.skip_aneel:
        logger.info("=" * 60)
        logger.info("ETAPA 1.5/4: Coleta de dados ANEEL")
        logger.info("=" * 60)
        df_perfil_aneel = collect_aneel_profile(aneel_collector, args.force_refresh)

    # --- Filtro por submercado (se solicitado) ---
    registros_geracao, registros_pld = filter_by_submercado(
        registros_geracao, registros_pld, args.submercado
    )

    # --- ETAPA 2: Análise de curtailment ---
    logger.info("=" * 60)
    logger.info("ETAPA 2/4: Análise de curtailment e excesso de geração")
    logger.info("=" * 60)

    curtailment_analyzer = CurtailmentAnalyzer()

    ranking_curtailment = curtailment_analyzer.rank_submercados_por_potencial(
        registros_geracao, registros_pld
    )
    if not ranking_curtailment.empty:
        logger.info("\nRanking de submercados por potencial de curtailment:\n" +
                     ranking_curtailment[
                         ['submercado', 'score_potencial_bess',
                          'energia_cortada_total_mwh', 'spread_medio_pld']
                     ].to_string(index=False))

    # --- ETAPA 3: Análise de spread de preços ---
    logger.info("=" * 60)
    logger.info("ETAPA 3/4: Análise de spread de preços (PLD)")
    logger.info("=" * 60)

    spread_analyzer = PriceSpreadAnalyzer(
        eficiencia_bess=args.eficiencia
    )

    ranking_spread = spread_analyzer.rank_submercados_por_spread(registros_pld)
    if not ranking_spread.empty:
        logger.info("\nRanking de submercados por spread de preço:\n" +
                     ranking_spread[
                         ['submercado', 'score_spread', 'spread_medio_rs_mwh',
                          'consistencia_horario_pct']
                     ].to_string(index=False))

    # --- ETAPA 4: Consolidação de oportunidades BESS ---
    logger.info("=" * 60)
    logger.info("ETAPA 4/4: Identificação de oportunidades de BESS")
    logger.info("=" * 60)

    opportunity_finder = BESSOpportunityFinder(
        curtailment_analyzer=curtailment_analyzer,
        spread_analyzer=spread_analyzer,
        capex_rs_por_mwh=args.capex_mwh,
        capex_rs_por_mw=args.capex_mw,
    )

    oportunidades = opportunity_finder.find_opportunities(
        registros_geracao=registros_geracao,
        registros_pld=registros_pld,
        df_perfil_aneel=df_perfil_aneel,
        min_score_corte=args.min_score,
    )

    if not oportunidades:
        logger.warning(
            "Nenhuma oportunidade de BESS identificada com os critérios atuais "
            f"(min_score={args.min_score}). Considere relaxar os filtros ou "
            "expandir o período de análise."
        )
        return {
            'resumo_executivo': {'total_oportunidades_identificadas': 0},
            'ranking_completo': [],
            'top_5_analise_detalhada': [],
        }

    resumo = opportunity_finder.generate_executive_report(oportunidades)
    df_oportunidades = opportunity_finder.export_opportunities_to_dataframe(oportunidades)

    # --- Exportação de arquivos ---
    caminhos_exportados = export_results(resumo, df_oportunidades, args.output)
    resumo['_output_path'] = caminhos_exportados['json']

    # --- Análise de sensibilidade (opcional) ---
    if args.sensitivity and oportunidades:
        logger.info("=" * 60)
        logger.info("EXTRA: Análise de sensibilidade (melhor oportunidade)")
        logger.info("=" * 60)

        melhor_oportunidade = oportunidades[0]
        df_sensitivity = opportunity_finder.sensitivity_analysis(melhor_oportunidade)

        sensitivity_csv_path = Path(args.output) / (
            f"sensibilidade_{melhor_oportunidade.submercado}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        df_sensitivity.to_csv(sensitivity_csv_path, index=False, sep=';', encoding='utf-8-sig')
        logger.info(f"Análise de sensibilidade salva: {sensitivity_csv_path}")

        print_sensitivity_summary(df_sensitivity, melhor_oportunidade.submercado)

    return resumo


# ======================================================================
# VALIDAÇÃO DE AMBIENTE
# ======================================================================
def validate_environment() -> bool:
    """Valida dependências e configurações mínimas antes de executar"""
    try:
        import requests
        import pandas
        import numpy
    except ImportError as e:
        logger.error(f"Dependência faltante: {e}")
        logger.error("Execute: pip install -r requirements.txt")
        return False

    return True


def create_requirements_file():
    """Gera arquivo requirements.txt se não existir"""
    requirements_path = Path("requirements.txt")
    if requirements_path.exists():
        return

    conteudo = """requests>=2.31.0
pandas>=2.0.0
numpy>=1.24.0
python-dateutil>=2.8.2
"""
    requirements_path.write_text(conteudo, encoding='utf-8')
    logger.info("requirements.txt criado")


# ======================================================================
# ENTRYPOINT
# ======================================================================
def main():
    args = parse_args()

    log_file = Path(args.output) / "logs" / f"execucao_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    setup_logging(verbose=args.verbose, log_file=str(log_file))

    logger.info("=" * 80)
    logger.info(" BESS ARBITRAGE ANALYZER - Mapeamento de Oportunidades")
    logger.info(" Fonte de dados: ONS (geração/PLD) + ANEEL (capacidade/saturação)")
    logger.info("=" * 80)

    if not validate_environment():
        sys.exit(1)

    create_requirements_file()

    inicio_execucao = datetime.now()

    try:
        resumo = run_pipeline(args)

        if resumo is None:
            logger.error("Pipeline abortado por falta de dados suficientes.")
            sys.exit(1)

        print_console_summary(resumo)

        duracao = (datetime.now() - inicio_execucao).total_seconds()
        logger.info(f"Pipeline concluído em {duracao:.1f}s")

    except KeyboardInterrupt:
        logger.warning("Execução interrompida pelo usuário")
        sys.exit(130)

    except Exception as e:
        logger.error(f"Erro fatal no pipeline: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()