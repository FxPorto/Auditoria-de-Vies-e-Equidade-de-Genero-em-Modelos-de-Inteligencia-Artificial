"""Validação da base (Guia, Passo 3): reconhece a versão certa e as versões incompletas."""
from motor.sintetico import gerar_adult_sintetico
from servicos.validacao_base import conferir_hash, registrar_hash, validar


def _situacao(checks, nome):
    return next(c for c in checks if c["verificacao"] == nome)["situacao"]


def test_estrutura_da_adult_passa_e_versoes_incompletas_falham():
    completa = gerar_adult_sintetico(48842, seed=0)
    checks = validar(completa)
    assert _situacao(checks, "Número de registros") == "OK"
    assert _situacao(checks, "Colunas") == "OK"
    assert not [c for c in checks if c["situacao"] == "FALHA"]

    treino = validar(completa.head(32561))
    reg = next(c for c in treino if c["verificacao"] == "Número de registros")
    assert reg["situacao"] == "FALHA" and "treino" in reg["dica"]

    kaggle = validar(completa.rename(columns={"education-num": "education.num"}))
    col = next(c for c in kaggle if c["verificacao"] == "Colunas")
    assert col["situacao"] == "FALHA" and "Kaggle" in col["dica"]


def test_hash_registrado_detecta_troca_de_base(tmp_path):
    arq = tmp_path / "adult.csv"
    gerar_adult_sintetico(200, seed=0).to_csv(arq, index=False, lineterminator="\n")
    registrado = registrar_hash(arq)
    assert conferir_hash(arq) == (registrado, registrado)
    arq.write_text(arq.read_text().replace("Male", "Mal3", 1))
    atual, reg = conferir_hash(arq)
    assert atual != reg
