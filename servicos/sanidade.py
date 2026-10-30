"""Checagens de sanidade na base real (Plano, seção 5.3).

Os valores de referência são aproximados e servem só para pegar erro grosseiro.
"""
from __future__ import annotations

import pandas as pd


def checar(df: pd.DataFrame, agregado: list | None = None, alvo="income", sensivel="sex",
           base_adult: bool = True, registros_esperados: int | None = None) -> list[dict]:
    """base_adult=True usa as referências da Adult; False (ACSIncome ou outra base) usa faixas amplas."""
    out = []

    def item(nome, valor, ok, referencia):
        out.append({"verificacao": nome, "valor": valor, "referencia": referencia,
                    "situacao": "OK" if ok else "ALERTA"})

    if base_adult:
        item("Número de registros", len(df), len(df) == 48842, "48.842")
    elif registros_esperados:
        item("Número de registros", len(df), abs(len(df) - registros_esperados) <= 0.01 * registros_esperados,
             f"cerca de {registros_esperados:,}".replace(",", "."))
    prev = df[alvo].mean()
    if base_adult:
        item("Proporção com renda alta", round(prev, 3), 0.20 <= prev <= 0.28, "cerca de 0,24")
    else:
        item("Proporção com renda alta", round(prev, 3), 0.15 <= prev <= 0.60, "entre 0,15 e 0,60 (faixa ampla)")
    taxa = df.groupby(sensivel)[alvo].mean()
    prop = df[sensivel].value_counts(normalize=True)
    if {"Male", "Female"} <= set(taxa.index):
        if base_adult:
            item("Proporção de homens", round(prop["Male"], 3), 0.60 <= prop["Male"] <= 0.72, "cerca de 0,67")
        else:
            item("Proporção de homens", round(prop["Male"], 3), 0.40 <= prop["Male"] <= 0.70, "entre 0,40 e 0,70")
        item("Taxa-base homens / mulheres", f"{taxa['Male']:.3f} / {taxa['Female']:.3f}",
             taxa["Male"] > taxa["Female"], "homens acima de mulheres" if not base_adult else "cerca de 0,30 / 0,11")
    linha_base = max(prev, 1 - prev)
    if agregado:
        ag = pd.DataFrame(agregado)
        orig = ag[(ag.cenario == "original")].set_index(["modelo", "metrica"])["media"]
        for m in dict.fromkeys(ag.modelo):
            acc, auc = orig.get((m, "acuracia")), orig.get((m, "auc"))
            if acc is not None:
                item(f"Acurácia original ({m})", round(acc, 3), linha_base < acc < 0.95,
                     f"acima da linha de base ({linha_base:.3f}) e abaixo de 0,95" + ("; cerca de 0,85" if base_adult else ""))
            if auc is not None:
                item(f"AUC original ({m})", round(auc, 3), (0.75 if base_adult else 0.70) <= auc <= 0.97,
                     "cerca de 0,90" if base_adult else "entre 0,70 e 0,97")
    return out
