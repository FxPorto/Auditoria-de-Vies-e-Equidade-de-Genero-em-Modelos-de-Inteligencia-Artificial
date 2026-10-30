"""Protocolo experimental (Documentação, Seção 3.4; Plano, Parte 2).

Para cada semente:
  1. divisão estratificada 60/20/20 (treino, validação, teste);
  2. ajuste de hiperparâmetros na validação e treino de cada modelo;
  3. cenário original, reponderação e limiar por grupo, todos avaliados no MESMO teste;
Depois: média, desvio-padrão, IC 95% e diferenças pareadas entre cenários.
"""
from __future__ import annotations

import platform
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import estatistica, metricas, mitigacao, modelos
from .config import Config

METRICAS_COMPARADAS = ["acuracia", "f1", "auc", "dif_paridade", "dif_ig_oport", "dif_equalized_odds",
                       "dif_paridade_assinada", "dif_ig_oport_assinada"]


def versoes() -> dict:
    import fairlearn, sklearn
    return {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
            "fairlearn": fairlearn.__version__, "pandas": pd.__version__, "numpy": np.__version__}


def dividir(y: pd.Series, semente: int, proporcoes=(0.6, 0.2, 0.2)):
    """Índices de treino, validação e teste, estratificados pelo alvo (RN02)."""
    _, p_val, p_te = proporcoes
    idx = np.arange(len(y))
    idx_resto, idx_te = train_test_split(idx, test_size=p_te, stratify=y, random_state=semente)
    idx_tr, idx_val = train_test_split(idx_resto, test_size=p_val / (1 - p_te),
                                       stratify=y.iloc[idx_resto], random_state=semente)
    return idx_tr, idx_val, idx_te


def executar_semente(X, y, s, cfg: Config, semente: int, indices=None) -> dict:
    """Executa uma repetição. Devolve registros de métricas, previsões e objetos ajustados."""
    idx_tr, idx_val, idx_te = indices if indices is not None else dividir(y, semente, cfg.proporcoes)
    Xtr, Xval, Xte = X.iloc[idx_tr], X.iloc[idx_val], X.iloc[idx_te]
    ytr, yval, yte = y.iloc[idx_tr], y.iloc[idx_val], y.iloc[idx_te]
    str_, sval, ste = s.iloc[idx_tr], s.iloc[idx_val], s.iloc[idx_te]
    registros, previsoes, objetos = [], [], {}
    base = {"semente": semente, "n_treino": len(idx_tr), "n_validacao": len(idx_val), "n_teste": len(idx_te)}

    def registrar(modelo, cenario, yp, score, params, auc_val):
        m = metricas.calcular(yte, yp, ste, score, cfg.grupo_referencia)
        registros.append({**base, "modelo": modelo, "cenario": cenario, "hiperparametros": params,
                          "auc_validacao": auc_val, "metricas": metricas.achatar(m)})
        previsoes.append(pd.DataFrame({
            "semente": semente, "modelo": modelo, "cenario": cenario, "indice": idx_te,
            "y_true": yte.to_numpy(), "sexo": ste.to_numpy(),
            "score": score if score is not None else np.nan, "y_pred": yp}))

    for nome in cfg.modelos:
        pipe, params, auc_val = modelos.ajustar_e_treinar(nome, Xtr, ytr, Xval, yval, cfg, semente)
        score = pipe.predict_proba(Xte)[:, 1]
        registrar(nome, "original", (score >= 0.5).astype(int), score, params, auc_val)
        objetos[(nome, "original")] = pipe

        if "reponderacao" in cfg.tecnicas:
            pesos = mitigacao.pesos_reponderacao(ytr, str_)
            pipe_w = modelos.treinar_com_pesos(nome, params, Xtr, ytr, pesos, cfg, semente)
            score_w = pipe_w.predict_proba(Xte)[:, 1]
            registrar(nome, "reponderacao", (score_w >= 0.5).astype(int), score_w, params, auc_val)
            objetos[(nome, "reponderacao")] = pipe_w

        if "limiar_por_grupo" in cfg.tecnicas:
            to = mitigacao.ajustar_limiar(pipe, Xval, yval, sval, cfg.restricao)   # validação, não teste
            yp_t = mitigacao.prever_limiar(to, Xte, ste, semente)
            registrar(nome, "limiar_por_grupo", yp_t, None, params, auc_val)
            objetos[(nome, "limiar_por_grupo")] = to

    return {"registros": registros, "previsoes": pd.concat(previsoes, ignore_index=True),
            "objetos": objetos, "indices": (idx_tr, idx_val, idx_te)}


def agregar(registros: list, cfg: Config) -> list:
    df = pd.DataFrame([{"modelo": r["modelo"], "cenario": r["cenario"], **r["metricas"]} for r in registros])
    linhas = []
    for (modelo, cenario), g in df.groupby(["modelo", "cenario"], sort=False):
        for met in [c for c in g.columns if c not in ("modelo", "cenario")]:
            linhas.append({"modelo": modelo, "cenario": cenario, "metrica": met,
                           **estatistica.resumir(g[met].to_numpy(), cfg.n_bootstrap)})
    return linhas


def comparar(registros: list, cfg: Config) -> list:
    """Diferenças pareadas por semente (P2, P3 e P4 do plano de testes)."""
    df = pd.DataFrame([{"semente": r["semente"], "modelo": r["modelo"], "cenario": r["cenario"],
                        **r["metricas"]} for r in registros])
    razao = cfg.proporcoes[2] / cfg.proporcoes[0]
    pares = []
    for m in cfg.modelos:
        cen = cfg.cenarios()
        for a, b, pergunta in [("reponderacao", "original", "P2"), ("limiar_por_grupo", "original", "P3"),
                               ("limiar_por_grupo", "reponderacao", "P3")]:
            if a in cen and b in cen:
                pares.append((pergunta, f"{m}: {a} − {b}", (m, a), (m, b)))
    if len(cfg.modelos) == 2:
        for c in cfg.cenarios():
            pares.append(("P4", f"{c}: random_forest − regressao_logistica",
                          ("random_forest", c), ("regressao_logistica", c)))
    linhas = []
    for pergunta, rotulo, (ma, ca), (mb, cb) in pares:
        A = df[(df.modelo == ma) & (df.cenario == ca)].set_index("semente")
        B = df[(df.modelo == mb) & (df.cenario == cb)].set_index("semente")
        comuns = A.index.intersection(B.index)
        for met in METRICAS_COMPARADAS:
            r = estatistica.diferenca_pareada(A.loc[comuns, met], B.loc[comuns, met], razao, cfg.n_bootstrap)
            limite = {"acuracia": cfg.limite_acuracia, "dif_paridade": cfg.limite_paridade,
                      "dif_paridade_assinada": cfg.limite_paridade}.get(met)
            r["relevante"] = (bool(r["consistente"] and abs(r["media"]) >= limite)
                              if limite is not None else None)
            linhas.append({"pergunta": pergunta, "comparacao": rotulo, "metrica": met, **r})
    return linhas


def executar_protocolo(X, y, s, cfg: Config, info_dados: dict | None = None, progresso=None) -> dict:
    """Executa todas as sementes e agrega. `progresso(i, total, mensagem)` é opcional."""
    cfg.validar()
    registros, previsoes, tempos = [], [], []
    for i, semente in enumerate(cfg.sementes):
        if progresso:
            progresso(i, len(cfg.sementes), f"Semente {semente}")
        t0 = time.perf_counter()
        r = executar_semente(X, y, s, cfg, semente)
        tempos.append(time.perf_counter() - t0)
        registros += r["registros"]
        previsoes.append(r["previsoes"])
    if progresso:
        progresso(len(cfg.sementes), len(cfg.sementes), "Agregando resultados")
    return {
        "config": cfg.to_dict(), "dados": info_dados or {}, "versoes": versoes(),
        "grupos": sorted(pd.unique(s.astype(str)).tolist()),
        "tempo_por_semente_s": tempos, "registros": registros,
        "agregado": agregar(registros, cfg), "pareado": comparar(registros, cfg),
        "previsoes": pd.concat(previsoes, ignore_index=True),
    }
