"""T01 (exemplo feito à mão) e T02 (conferência com scikit-learn e Fairlearn)."""
import numpy as np
import pytest
from fairlearn.metrics import demographic_parity_difference, demographic_parity_ratio, equalized_odds_difference
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from motor import metricas


def test_t01_exemplo_feito_a_mao():
    # grupo A: y=[1,1,0,0,0], p=[1,0,1,0,0] -> sel 2/5, TPR 1/2, FPR 1/3
    # grupo B: y=[1,1,1,0,0], p=[1,1,1,1,0] -> sel 4/5, TPR 3/3, FPR 1/2
    y = [1, 1, 0, 0, 0, 1, 1, 1, 0, 0]
    p = [1, 0, 1, 0, 0, 1, 1, 1, 1, 0]
    s = ["A"] * 5 + ["B"] * 5
    m = metricas.calcular(y, p, s, grupo_referencia="B")
    assert m["desempenho"]["acuracia"] == pytest.approx(7 / 10)
    assert m["por_grupo"]["A"]["selecao"] == pytest.approx(2 / 5)
    assert m["por_grupo"]["A"]["tpr"] == pytest.approx(1 / 2)
    assert m["por_grupo"]["A"]["fpr"] == pytest.approx(1 / 3)
    assert m["por_grupo"]["B"]["tpr"] == pytest.approx(1.0)
    assert m["por_grupo"]["B"]["fpr"] == pytest.approx(1 / 2)
    assert m["por_grupo"]["A"]["taxa_base"] == pytest.approx(2 / 5)
    assert m["equidade"]["dif_paridade"] == pytest.approx(4 / 5 - 2 / 5)
    assert m["equidade"]["razao_paridade"] == pytest.approx((2 / 5) / (4 / 5))
    assert m["equidade"]["dif_ig_oport"] == pytest.approx(1 - 1 / 2)
    assert m["equidade"]["dif_equalized_odds"] == pytest.approx(max(1 - 1 / 2, 1 / 2 - 1 / 3))
    assert m["equidade"]["dif_paridade_assinada"] == pytest.approx(4 / 5 - 2 / 5)


def test_t02_confere_com_bibliotecas():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500); p = rng.integers(0, 2, 500); s = rng.choice(["F", "M"], 500)
    sc = rng.random(500)
    m = metricas.calcular(y, p, s, sc)
    assert m["desempenho"]["acuracia"] == pytest.approx(accuracy_score(y, p))
    assert m["desempenho"]["precisao"] == pytest.approx(precision_score(y, p))
    assert m["desempenho"]["recall"] == pytest.approx(recall_score(y, p))
    assert m["desempenho"]["f1"] == pytest.approx(f1_score(y, p))
    assert m["equidade"]["dif_paridade"] == pytest.approx(demographic_parity_difference(y, p, sensitive_features=s))
    assert m["equidade"]["razao_paridade"] == pytest.approx(demographic_parity_ratio(y, p, sensitive_features=s))
    assert m["equidade"]["dif_equalized_odds"] == pytest.approx(equalized_odds_difference(y, p, sensitive_features=s))
    ref = metricas.conferir_com_fairlearn(y, p, s)
    assert m["equidade"]["dif_ig_oport"] == pytest.approx(ref["dif_tpr"])
    for g in ("F", "M"):
        assert m["por_grupo"][g]["tpr"] == pytest.approx(ref["por_grupo"][g]["tpr"])
        assert m["por_grupo"][g]["fpr"] == pytest.approx(ref["por_grupo"][g]["fpr"])
        assert m["por_grupo"][g]["selecao"] == pytest.approx(ref["por_grupo"][g]["selecao"])


def test_metrica_indefinida_vira_nan():
    """EX07: grupo sem casos positivos -> TPR indefinida, sem erro."""
    m = metricas.calcular([0, 0, 1, 1], [0, 1, 1, 0], ["A", "A", "B", "B"])
    assert np.isnan(m["por_grupo"]["A"]["tpr"])
