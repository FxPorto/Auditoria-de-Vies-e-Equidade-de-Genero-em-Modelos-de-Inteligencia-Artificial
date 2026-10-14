"""Pré-processamento, modelos e ajuste de hiperparâmetros na validação (RF04, RN03)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import Config


def preprocessador(X: pd.DataFrame) -> ColumnTransformer:
    num = X.select_dtypes(include=np.number).columns.tolist()
    cat = [c for c in X.columns if c not in num]
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("esc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])


def estimador(nome: str, params: dict, cfg: Config, semente: int):
    if nome == "regressao_logistica":
        return LogisticRegression(max_iter=1000, solver="lbfgs", random_state=semente, **params)
    if nome == "random_forest":
        return RandomForestClassifier(n_estimators=cfg.rf_n_estimators, random_state=semente,
                                      n_jobs=cfg.n_jobs, **params)
    raise ValueError(f"Modelo desconhecido: {nome}")


def pipeline(nome: str, params: dict, X: pd.DataFrame, cfg: Config, semente: int) -> Pipeline:
    return Pipeline([("pre", preprocessador(X)), ("clf", estimador(nome, params, cfg, semente))])


def ajustar_e_treinar(nome, X_tr, y_tr, X_val, y_val, cfg: Config, semente: int):
    """Busca em grade: treina no treino, escolhe pela maior AUC na validação.

    O conjunto de teste nunca é passado a esta função (RN03). Devolve o pipeline
    escolhido (treinado só no treino), os hiperparâmetros e a AUC de validação.
    """
    melhor = None
    for params in ParameterGrid(cfg.grades.get(nome, {})):
        pipe = pipeline(nome, params, X_tr, cfg, semente).fit(X_tr, y_tr)
        auc = roc_auc_score(y_val, pipe.predict_proba(X_val)[:, 1])
        if melhor is None or auc > melhor[2]:
            melhor = (pipe, dict(params), float(auc))
    return melhor


def treinar_com_pesos(nome, params, X_tr, y_tr, pesos, cfg: Config, semente: int) -> Pipeline:
    """Treina o modelo com os mesmos hiperparâmetros e pesos por exemplo (reponderação)."""
    pipe = pipeline(nome, params, X_tr, cfg, semente)
    return pipe.fit(X_tr, y_tr, clf__sample_weight=pesos)
