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
    assert TipoRestricao.CURTAILMENT_EOLICO.value == "CURTAILMENT_EOLICO"

    print("✅ Enums validados corretamente!")

    # Valida que os diretórios de cache/output foram criados
    import os
    assert os.path.exists(settings.CACHE_DIR), "Diretório de cache não foi criado"

    print("✅ Diretórios de infraestrutura validados!")
    print("\n🎉 Smoke test concluído com sucesso - projeto pronto para uso.\n")


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