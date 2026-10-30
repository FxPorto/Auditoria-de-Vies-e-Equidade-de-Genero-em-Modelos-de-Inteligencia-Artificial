"""Métricas de desempenho, por grupo e de equidade (RF06, RF07, RF08)."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from fairlearn.metrics import (MetricFrame, demographic_parity_difference,
                               demographic_parity_ratio, equalized_odds_difference)
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score)

# descrição curta + limitação de cada métrica (RNF05); usada na interface e no relatório
DESCRICOES = {
    "acuracia": ("Acurácia", "Proporção de acertos. Pode ser alta mesmo com erros concentrados em um grupo."),
    "precisao": ("Precisão", "Dos classificados como renda alta, quantos realmente têm renda alta."),
    "recall": ("Recall", "Das pessoas com renda alta, quantas o modelo reconhece."),
    "f1": ("F1", "Média harmônica de precisão e recall."),
    "auc": ("AUC", "Capacidade de ordenar pessoas por probabilidade; não se aplica ao limiar por grupo, que devolve decisões."),
    "dif_paridade": ("Dif. de paridade demográfica", "Diferença entre as taxas de seleção dos grupos (0 = iguais). Ignora se as taxas-base diferem."),
    "razao_paridade": ("Razão de paridade demográfica", "Menor taxa de seleção dividida pela maior (1 = iguais)."),
    "dif_ig_oport": ("Dif. de igualdade de oportunidades", "Diferença de TPR entre os grupos (0 = iguais)."),
    "dif_equalized_odds": ("Dif. de equalized odds", "Maior diferença entre grupos em TPR ou FPR (0 = iguais)."),
    "dif_paridade_assinada": ("Dif. de seleção com sinal", "Seleção do grupo de referência menos a do outro grupo; o sinal mostra quem é mais selecionado."),
    "dif_ig_oport_assinada": ("Dif. de TPR com sinal", "TPR do grupo de referência menos a do outro grupo; o sinal mostra qual grupo é mais reconhecido entre quem tem o desfecho positivo. Use junto com a diferença sem sinal, que esconde inversões."),
}

METRICAS_PRINCIPAIS = ["acuracia", "precisao", "recall", "f1", "auc", "dif_paridade", "razao_paridade",
                       "dif_ig_oport", "dif_equalized_odds", "dif_paridade_assinada", "dif_ig_oport_assinada"]


def _taxa(num, den):
    return float(num / den) if den > 0 else float("nan")   # EX07: indefinida


def _por_grupo(y, yp, s) -> dict:
    out = {}
    for g in sorted(pd.unique(s)):
        m = s == g
        yt, pr = y[m], yp[m]
        tp = int(((yt == 1) & (pr == 1)).sum()); fp = int(((yt == 0) & (pr == 1)).sum())
        pos = int((yt == 1).sum()); neg = int((yt == 0).sum()); sel = int((pr == 1).sum())
        out[str(g)] = {"n": int(m.sum()), "taxa_base": _taxa(pos, len(yt)), "selecao": _taxa(sel, len(yt)),
                       "tpr": _taxa(tp, pos), "fpr": _taxa(fp, neg), "precisao": _taxa(tp, sel),
                       "recall": _taxa(tp, pos)}
    return out


def calcular(y_true, y_pred, s, y_score=None, grupo_referencia: str | None = None) -> dict:
    y = np.asarray(y_true).astype(int)
    yp = np.asarray(y_pred).astype(int)
    s = np.asarray(s).astype(str)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        desempenho = {
            "acuracia": float(accuracy_score(y, yp)),
            "precisao": float(precision_score(y, yp, zero_division=0)),
            "recall": float(recall_score(y, yp, zero_division=0)),
            "f1": float(f1_score(y, yp, zero_division=0)),
            "auc": float(roc_auc_score(y, y_score)) if y_score is not None and len(set(y)) > 1 else float("nan"),
        }
        grupos = _por_grupo(y, yp, s)
        tprs = [g["tpr"] for g in grupos.values()]
        equidade = {
            "dif_paridade": float(demographic_parity_difference(y, yp, sensitive_features=s)),
            "razao_paridade": float(demographic_parity_ratio(y, yp, sensitive_features=s)),
            "dif_ig_oport": float(np.nanmax(tprs) - np.nanmin(tprs)) if not all(np.isnan(tprs)) else float("nan"),
            "dif_equalized_odds": float(equalized_odds_difference(y, yp, sensitive_features=s)),
        }
    nomes = list(grupos)
    if grupo_referencia in grupos and len(nomes) == 2:
        outro = [g for g in nomes if g != grupo_referencia][0]
        equidade["dif_paridade_assinada"] = grupos[grupo_referencia]["selecao"] - grupos[outro]["selecao"]
        equidade["dif_ig_oport_assinada"] = grupos[grupo_referencia]["tpr"] - grupos[outro]["tpr"]
    else:
        equidade["dif_paridade_assinada"] = equidade["dif_ig_oport_assinada"] = float("nan")
    return {"desempenho": desempenho, "por_grupo": grupos, "equidade": equidade}


def achatar(m: dict) -> dict:
    """Uma linha de números: métricas principais + métricas por grupo (ex.: selecao_Male)."""
    linha = {**m["desempenho"], **m["equidade"]}
    for g, vals in m["por_grupo"].items():
        for k, v in vals.items():
            if k != "n":
                linha[f"{k}_{g}"] = v
    return linha


def conferir_com_fairlearn(y_true, y_pred, s) -> dict:
    """Mesmas medidas pelo MetricFrame do Fairlearn (usado no teste T02)."""
    from fairlearn.metrics import false_positive_rate, selection_rate, true_positive_rate
    mf = MetricFrame(metrics={"selecao": selection_rate, "tpr": true_positive_rate, "fpr": false_positive_rate},
                     y_true=np.asarray(y_true), y_pred=np.asarray(y_pred), sensitive_features=np.asarray(s))
    return {"por_grupo": mf.by_group.to_dict(orient="index"), "dif_tpr": float(mf.difference()["tpr"])}
