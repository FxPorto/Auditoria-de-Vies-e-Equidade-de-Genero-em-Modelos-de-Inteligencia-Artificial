"""Checagens de sanidade na base real (Plano, seção 5.3).

Os valores de referência são aproximados e servem só para pegar erro grosseiro.
"""
from __future__ import annotations

import pandas as pd


def checar(df: pd.DataFrame, agregado: list | None = None, alvo="income", sensivel="sex",
           base_adult: bool = True) -> list[dict]:
    out = []

    def item(nome, valor, ok, referencia):
        out.append({"verificacao": nome, "valor": valor, "referencia": referencia,
                    "situacao": "OK" if ok else "ALERTA"})

    if base_adult:
        item("Número de registros", len(df), len(df) == 48842, "48.842")
    prev = df[alvo].mean()
    item("Proporção com renda alta", round(prev, 3), 0.20 <= prev <= 0.28, "cerca de 0,24")
    taxa = df.groupby(sensivel)[alvo].mean()
    prop = df[sensivel].value_counts(normalize=True)
    if {"Male", "Female"} <= set(taxa.index):
        item("Proporção de homens", round(prop["Male"], 3), 0.60 <= prop["Male"] <= 0.72, "cerca de 0,67")
        item("Taxa-base homens / mulheres", f"{taxa['Male']:.3f} / {taxa['Female']:.3f}",
             taxa["Male"] > taxa["Female"], "cerca de 0,30 / 0,11")
    linha_base = max(prev, 1 - prev)
    if agregado:
        ag = pd.DataFrame(agregado)
        orig = ag[(ag.cenario == "original")].set_index(["modelo", "metrica"])["media"]
        for m in dict.fromkeys(ag.modelo):
            acc, auc = orig.get((m, "acuracia")), orig.get((m, "auc"))
            if acc is not None:
                item(f"Acurácia original ({m})", round(acc, 3), linha_base < acc < 0.95,
                     f"acima da linha de base ({linha_base:.3f}) e abaixo de 0,95; cerca de 0,85")
            if auc is not None:
                item(f"AUC original ({m})", round(auc, 3), 0.75 <= auc <= 0.97, "cerca de 0,90")
    return out
