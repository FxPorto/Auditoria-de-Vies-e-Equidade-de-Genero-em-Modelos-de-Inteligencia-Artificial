"""T03 a T09: sensibilidade, vazamento, divisão, isolamento do teste, mitigações e reprodutibilidade."""
import numpy as np
import pandas as pd
import pytest

from motor import mitigacao, protocolo
from motor.dados import preparar


def _agregado(res):
    return pd.DataFrame(res["agregado"]).set_index(["modelo", "cenario", "metrica"])


def test_t03_detecta_vies_conhecido(df_vies, cfg_rapido):
    cfg_rapido.sementes = [0, 1, 2]
    X, y, s = preparar(df_vies, cfg_rapido)
    ag = _agregado(protocolo.executar_protocolo(X, y, s, cfg_rapido))
    r = ag.loc[("regressao_logistica", "original", "dif_paridade_assinada")]
    assert r["media"] > 0.10 and r["ic_inf"] > 0       # homens mais selecionados, de forma consistente


def test_t03_sem_vies_nao_inventa_diferenca(df_sem_vies, cfg_rapido):
    cfg_rapido.sementes = [0, 1, 2, 3, 4]
    cfg_rapido.tecnicas = []
    X, y, s = preparar(df_sem_vies, cfg_rapido)
    ag = _agregado(protocolo.executar_protocolo(X, y, s, cfg_rapido))
    r = ag.loc[("regressao_logistica", "original", "dif_paridade_assinada")]
    assert abs(r["media"]) < 0.03


def test_t04_rotulos_embaralhados_auc_perto_de_meio(df_vies, cfg_rapido):
    cfg_rapido.sementes, cfg_rapido.tecnicas, cfg_rapido.modelos = [0], [], ["regressao_logistica"]
    X, y, s = preparar(df_vies, cfg_rapido)
    y_emb = pd.Series(np.random.default_rng(0).permutation(y.to_numpy()))
    ag = _agregado(protocolo.executar_protocolo(X, y_emb, s, cfg_rapido))
    assert 0.40 < ag.loc[("regressao_logistica", "original", "auc"), "media"] < 0.60


def test_t05_divisao(df_vies):
    y = df_vies["income"]
    tr, va, te = protocolo.dividir(y, 0, (0.6, 0.2, 0.2))
    n = len(y)
    assert abs(len(tr) - 0.6 * n) <= 2 and abs(len(va) - 0.2 * n) <= 2 and abs(len(te) - 0.2 * n) <= 2
    assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))
    assert len(set(tr) | set(va) | set(te)) == n
    prev = y.mean()
    for idx in (tr, va, te):
        assert abs(y.iloc[idx].mean() - prev) < 0.005


def test_t06_teste_nao_influencia_ajustes(df_vies, cfg_rapido):
    X, y, s = preparar(df_vies, cfg_rapido)
    ind = protocolo.dividir(y, 0, cfg_rapido.proporcoes)
    a = protocolo.executar_semente(X, y, s, cfg_rapido, 0, ind)
    y2 = y.copy()
    te = ind[2]
    y2.iloc[te] = np.random.default_rng(1).permutation(y2.iloc[te].to_numpy())
    b = protocolo.executar_semente(X, y2, s, cfg_rapido, 0, ind)
    hp_a = {(r["modelo"], r["cenario"]): r["hiperparametros"] for r in a["registros"]}
    hp_b = {(r["modelo"], r["cenario"]): r["hiperparametros"] for r in b["registros"]}
    assert hp_a == hp_b
    Xv, sv = X.iloc[ind[1]], s.iloc[ind[1]]
    for m in cfg_rapido.modelos:
        pa = mitigacao.prever_limiar(a["objetos"][(m, "limiar_por_grupo")], Xv, sv, 0)
        pb = mitigacao.prever_limiar(b["objetos"][(m, "limiar_por_grupo")], Xv, sv, 0)
        assert np.array_equal(pa, pb)


def test_t07_pesos_da_reponderacao(df_vies):
    y, s = df_vies["income"], df_vies["sex"]
    w = mitigacao.pesos_reponderacao(y, s)
    taxas = pd.Series(w * y.to_numpy()).groupby(s.to_numpy()).sum() / pd.Series(w).groupby(s.to_numpy()).sum()
    assert taxas.max() - taxas.min() < 1e-9
    assert taxas.iloc[0] == pytest.approx(y.mean())


def test_t08_limiar_atende_restricao_na_validacao(df_vies, cfg_rapido):
    X, y, s = preparar(df_vies, cfg_rapido)
    ind = protocolo.dividir(y, 0, cfg_rapido.proporcoes)
    r = protocolo.executar_semente(X, y, s, cfg_rapido, 0, ind)
    Xv, yv, sv = X.iloc[ind[1]], y.iloc[ind[1]], s.iloc[ind[1]]
    from motor import metricas
    for m in cfg_rapido.modelos:
        pv = mitigacao.prever_limiar(r["objetos"][(m, "limiar_por_grupo")], Xv, sv, 0)
        assert metricas.calcular(yv, pv, sv)["equidade"]["dif_paridade"] < 0.05


def test_t09_reprodutibilidade(df_vies, cfg_rapido):
    X, y, s = preparar(df_vies, cfg_rapido)
    a = _agregado(protocolo.executar_protocolo(X, y, s, cfg_rapido))["media"]
    b = _agregado(protocolo.executar_protocolo(X, y, s, cfg_rapido))["media"]
    pd.testing.assert_series_equal(a, b)
    cfg_rapido.sementes = [5, 6]
    c = _agregado(protocolo.executar_protocolo(X, y, s, cfg_rapido))["media"]
    assert not np.allclose(a.fillna(0).to_numpy(), c.fillna(0).to_numpy())


def test_sexo_fora_das_entradas_por_padrao(df_vies, cfg_rapido):
    X, _, _ = preparar(df_vies, cfg_rapido)
    assert "sex" not in X.columns and "fnlwgt" not in X.columns and "education" not in X.columns
    cfg_rapido.usar_sexo_como_entrada = True
    X2, _, _ = preparar(df_vies, cfg_rapido)
    assert "sex" in X2.columns
