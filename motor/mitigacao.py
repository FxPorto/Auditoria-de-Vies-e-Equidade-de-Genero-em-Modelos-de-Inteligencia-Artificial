"""Técnicas de mitigação (RF09).

- Reponderação (KAMIRAN; CALDERS, 2012): pré-processamento; o sexo é usado só no treino.
- Limiar por grupo (ThresholdOptimizer, Fairlearn; HARDT; PRICE; SREBRO, 2016):
  pós-processamento ajustado na VALIDAÇÃO; o sexo é usado na decisão.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from fairlearn.postprocessing import ThresholdOptimizer


def pesos_reponderacao(y: pd.Series, s: pd.Series) -> np.ndarray:
    """w(s, y) = P(s) * P(y) / P(s, y), estimados no treino.

    Com esses pesos, a taxa ponderada de y = 1 é igual em todos os grupos (teste T07).
    """
    y = pd.Series(np.asarray(y)).reset_index(drop=True)
    s = pd.Series(np.asarray(s)).reset_index(drop=True)
    p_s = s.value_counts(normalize=True)
    p_y = y.value_counts(normalize=True)
    p_sy = pd.crosstab(s, y, normalize=True)
    w = np.array([p_s[si] * p_y[yi] / p_sy.loc[si, yi] for si, yi in zip(s, y)], dtype=float)
    return w


def ajustar_limiar(pipe, X_val, y_val, s_val, restricao: str) -> ThresholdOptimizer:
    to = ThresholdOptimizer(estimator=pipe, constraints=restricao, objective="accuracy_score",
                            predict_method="predict_proba", prefit=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        to.fit(X_val, y_val, sensitive_features=s_val)
    return to


def prever_limiar(to: ThresholdOptimizer, X, s, semente: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.asarray(to.predict(X, sensitive_features=s, random_state=semente)).astype(int)
