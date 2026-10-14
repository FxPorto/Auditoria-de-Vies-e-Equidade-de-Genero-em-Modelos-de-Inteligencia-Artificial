"""Base sintética com as mesmas colunas da Adult, para testes (T03) e demonstração.

`vies` controla a dependência entre sexo e renda:
  0   -> sexo independente de tudo (nenhuma disparidade esperada);
  > 0 -> efeito direto do sexo na renda, horas maiores para homens e
         relationship (Husband/Wife) funcionando como proxy do sexo.
NUNCA usar estes números no artigo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_EDUC = {9: "HS-grad", 10: "Some-college", 13: "Bachelors", 14: "Masters", 7: "11th",
         11: "Assoc-voc", 12: "Assoc-acdm", 15: "Prof-school", 16: "Doctorate", 5: "9th"}


def gerar_adult_sintetico(n: int = 6000, seed: int = 0, vies: float = 1.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    homem = rng.random(n) < 0.67
    idade = rng.integers(17, 80, n)
    edu = rng.choice(list(_EDUC), n, p=[0.32, 0.22, 0.17, 0.06, 0.06, 0.05, 0.04, 0.03, 0.02, 0.03])
    casado = rng.random(n) < 0.46
    horas = np.clip(rng.normal(38 + 4 * vies * homem, 11, n), 1, 99).round()
    ganho = np.where(rng.random(n) < 0.08, rng.integers(1000, 30000, n), 0)
    perda = np.where(rng.random(n) < 0.05, rng.integers(500, 3000, n), 0)
    if vies > 0:
        rel_casado = np.where(homem, "Husband", "Wife")
    else:
        rel_casado = np.where(rng.random(n) < 0.5, "Husband", "Wife")
    rel_solteiro = rng.choice(["Not-in-family", "Own-child", "Unmarried"], n)
    relationship = np.where(casado, rel_casado, rel_solteiro)
    marital = np.where(casado, "Married-civ-spouse",
                       rng.choice(["Never-married", "Divorced", "Separated", "Widowed"], n))
    workclass = rng.choice(["Private", "Self-emp-not-inc", "Local-gov", "State-gov", "Federal-gov"],
                           n, p=[0.74, 0.09, 0.07, 0.05, 0.05]).astype(object)
    workclass[rng.random(n) < 0.05] = np.nan
    occupation = rng.choice(["Prof-specialty", "Craft-repair", "Exec-managerial", "Adm-clerical",
                             "Sales", "Other-service", "Machine-op-inspct"], n).astype(object)
    occupation[rng.random(n) < 0.05] = np.nan
    logit = (-9.6 + 0.035 * idade + 0.33 * edu + 0.03 * horas + 1.6 * casado
             + 0.00015 * ganho + vies * 0.9 * homem + rng.normal(0, 0.6, n))
    renda = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({
        "age": idade, "workclass": workclass, "fnlwgt": rng.integers(20000, 500000, n),
        "education": [_EDUC[e] for e in edu], "education-num": edu, "marital-status": marital,
        "occupation": occupation, "relationship": relationship,
        "race": rng.choice(["White", "Black", "Asian-Pac-Islander", "Other"], n, p=[0.85, 0.1, 0.03, 0.02]),
        "sex": np.where(homem, "Male", "Female"), "capital-gain": ganho, "capital-loss": perda,
        "hours-per-week": horas, "native-country": "United-States", "income": renda,
    })
