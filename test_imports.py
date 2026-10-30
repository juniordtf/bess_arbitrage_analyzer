# test_imports.py (script de smoke test rápido)
"""Smoke test: valida que todos os imports do pacote funcionam sem erro"""

def test_all_imports():
    from config import Settings
    from models import (
        TipoFonte, TipoRestricao, RegistroGeracao,
        RegistroPLD, RestricaoOperativa, OportunidadeBESS
    )
    from data_collectors import ONSDataCollector, ANEELDataCollector
    from analysis import (
        CurtailmentAnalyzer, PriceSpreadAnalyzer, BESSOpportunityFinder
    )

    # Instancia classes principais para garantir que não há erro de __init__
    settings = Settings()
    ons = ONSDataCollector()
    aneel = ANEELDataCollector()
    curtailment = CurtailmentAnalyzer()
    spread = PriceSpreadAnalyzer()
    finder = BESSOpportunityFinder(curtailment, spread)

    print("✅ Todos os imports e instanciações funcionaram corretamente!")

    # Valida enums
    assert TipoFonte.EOLICA.value == "EOLICA"
    assert (
        TipoRestricao.CURTAILMENT_EOLICO_COFF.value
        == "CURTAILMENT_EOLICO_COFF"
    )

    print("✅ Enums validados corretamente!")

    # Valida que os diretórios de cache/output foram criados
    import os
    assert os.path.exists(settings.CACHE_DIR), "Diretório de cache não foi criado"

    print("✅ Diretórios de infraestrutura validados!")
    print("\n🎉 Smoke test concluído com sucesso - projeto pronto para uso.\n")


def test_detect_curtailment_events_with_multiple_sources():
    from datetime import datetime

    from analysis import CurtailmentAnalyzer
    from models.data_models import (
        RazaoRestricaoCOFF,
        RestricaoCOFF,
        TipoFonte,
        TipoRestricao,
    )

    registros = [
        RestricaoCOFF(
            timestamp=datetime(2026, 1, 1),
            usina="Usina",
            submercado="SE",
            fonte=fonte,
            potencia_restringida_mw=10.0,
            razao_restricao=RazaoRestricaoCOFF.OUTRA,
        )
        for fonte in (TipoFonte.EOLICA, TipoFonte.SOLAR)
    ]

    eventos = CurtailmentAnalyzer().detect_curtailment_events_real(registros)

    assert len(eventos) == 2
    assert {evento.tipo for evento in eventos} == {
        TipoRestricao.CURTAILMENT_EOLICO_COFF,
        TipoRestricao.CURTAILMENT_SOLAR_COFF,
    }


if __name__ == "__main__":
    try:
        test_all_imports()
    except ImportError as e:
        print(f"❌ ERRO DE IMPORT: {e}")
        print("\nVerifique se:")
        print("  1. Todos os __init__.py estão presentes")
        print("  2. O script está sendo executado a partir do diretório raiz do projeto")
        print("  3. As dependências estão instaladas (pip install -r requirements.txt)")
        raise
    except AssertionError as e:
        print(f"❌ ERRO DE VALIDAÇÃO: {e}")
        raise