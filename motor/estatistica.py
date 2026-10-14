"""Agregação sobre as repetições: média, desvio-padrão e IC 95% (RF19).

- IC por bootstrap percentil sobre os valores das repetições.
- Diferenças pareadas entre cenários (mesmo conjunto de teste em cada semente).
- Conferência com o teste t corrigido para reamostragem (NADEAU; BENGIO, 2003),
  que leva em conta que as repetições compartilham dados.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def ic_bootstrap(valores, n_boot: int = 2000, semente: int = 0, alfa: float = 0.05):
    v = np.asarray(valores, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return float("nan"), float("nan")
    if len(v) == 1:
        return float(v[0]), float(v[0])
    rng = np.random.default_rng(semente)
    medias = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return float(np.quantile(medias, alfa / 2)), float(np.quantile(medias, 1 - alfa / 2))


def resumir(valores, n_boot: int = 2000, semente: int = 0) -> dict:
    v = np.asarray(valores, dtype=float)
    validos = v[~np.isnan(v)]
    lo, hi = ic_bootstrap(validos, n_boot, semente)
    return {"media": float(validos.mean()) if len(validos) else float("nan"),
            "dp": float(validos.std(ddof=1)) if len(validos) > 1 else float("nan"),
            "ic_inf": lo, "ic_sup": hi, "n": int(len(validos)), "indefinidos": int(np.isnan(v).sum())}


def ic_t_corrigido(diferencas, razao_teste_treino: float, alfa: float = 0.05):
    """Nadeau e Bengio (2003): var corrigida = (1/J + n_teste/n_treino) * s²."""
    d = np.asarray(diferencas, dtype=float)
    d = d[~np.isnan(d)]
    j = len(d)
    if j < 2:
        return float("nan"), float("nan")
    var = (1 / j + razao_teste_treino) * d.var(ddof=1)
    t = stats.t.ppf(1 - alfa / 2, j - 1)
    return float(d.mean() - t * np.sqrt(var)), float(d.mean() + t * np.sqrt(var))


def diferenca_pareada(a, b, razao_teste_treino: float, n_boot: int = 2000, semente: int = 0) -> dict:
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    r = resumir(d, n_boot, semente)
    t_lo, t_hi = ic_t_corrigido(d, razao_teste_treino)
    r.update({"ic_t_inf": t_lo, "ic_t_sup": t_hi,
              "consistente": bool(not np.isnan(r["ic_inf"]) and (r["ic_inf"] > 0 or r["ic_sup"] < 0)),
              "consistente_t": bool(not np.isnan(t_lo) and (t_lo > 0 or t_hi < 0))})
    return r
