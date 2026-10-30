"""Tabelas e figura do artigo, geradas a partir dos resultados congelados (Plano, seção 5.5)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from motor.config import CENARIOS, MODELOS
from motor.metricas import DESCRICOES

COLUNAS_T1 = ["acuracia", "f1", "auc", "dif_paridade", "razao_paridade", "dif_ig_oport", "dif_ig_oport_assinada", "dif_equalized_odds"]
COR_MODELO = {"regressao_logistica": "#2a78d6", "random_forest": "#eb6834"}   # validadas (dataviz)
MARCADOR = {"original": "o", "reponderacao": "s", "limiar_por_grupo": "^", "importado": "D"}
TINTA, TINTA2, GRADE = "#0b0b0b", "#52514e", "#e4e3df"


def num(v, casas=3):
    return "—" if pd.isna(v) else f"{v:.{casas}f}".replace(".", ",")


def fmt_ic(media, lo, hi, casas=3):
    if pd.isna(media):
        return "n/a"
    if pd.isna(lo) or lo == hi:
        return num(media, casas)
    return f"{num(media, casas)} ({num(lo, casas)} a {num(hi, casas)})"


def _nome(m):
    return MODELOS.get(m, m)


def tabela1(agregado: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tabela 1: média e IC 95% por modelo e cenário. Devolve (numérica, formatada)."""
    ag = pd.DataFrame(agregado)
    ag = ag[ag.metrica.isin(COLUNAS_T1)]
    numerica = ag.pivot_table(index=["modelo", "cenario"], columns="metrica",
                              values=["media", "dp", "ic_inf", "ic_sup"], sort=False)
    linhas = []
    for (m, c), g in ag.groupby(["modelo", "cenario"], sort=False):
        lin = {"Modelo": _nome(m), "Cenário": CENARIOS.get(c, c)}
        for met in COLUNAS_T1:
            r = g[g.metrica == met]
            lin[DESCRICOES[met][0]] = fmt_ic(*r[["media", "ic_inf", "ic_sup"]].iloc[0]) if len(r) else "n/a"
        linhas.append(lin)
    return numerica, pd.DataFrame(linhas)


def tabela2(agregado: list, grupos: list) -> pd.DataFrame:
    """Tabela 2: por grupo de sexo registrado, taxa-base, seleção, TPR e FPR (média e IC)."""
    ag = pd.DataFrame(agregado).set_index(["modelo", "cenario", "metrica"])
    linhas = []
    for (m, c) in dict.fromkeys(zip(*[ag.index.get_level_values(i) for i in (0, 1)])):
        for g in grupos:
            lin = {"Modelo": _nome(m), "Cenário": CENARIOS.get(c, c), "Sexo registrado": g}
            for k, rot in [("taxa_base", "Taxa-base"), ("selecao", "Taxa de seleção"), ("tpr", "TPR"), ("fpr", "FPR")]:
                chave = (m, c, f"{k}_{g}")
                lin[rot] = fmt_ic(*ag.loc[chave, ["media", "ic_inf", "ic_sup"]]) if chave in ag.index else "n/a"
            linhas.append(lin)
    return pd.DataFrame(linhas)


def tabela3(pareado: list) -> pd.DataFrame:
    """Tabela 3: diferenças pareadas com IC por bootstrap e pelo t corrigido (P2 a P4)."""
    linhas = []
    for r in pareado:
        comp = r["comparacao"]
        for k, v in {**MODELOS, **CENARIOS}.items():
            comp = comp.replace(k, v)
        linhas.append({
            "Pergunta": r["pergunta"], "Comparação": comp, "Métrica": DESCRICOES[r["metrica"]][0],
            "Diferença média": num(r["media"], 4),
            "IC 95% bootstrap": f"{num(r['ic_inf'], 4)} a {num(r['ic_sup'], 4)}",
            "IC 95% t corrigido": f"{num(r['ic_t_inf'], 4)} a {num(r['ic_t_sup'], 4)}",
            "Consistente (bootstrap / t)": f"{'sim' if r['consistente'] else 'não'} / {'sim' if r['consistente_t'] else 'não'}",
            "Relevante": "—" if r.get("relevante") is None else ("sim" if r["relevante"] else "não"),
        })
    return pd.DataFrame(linhas)


def figura_tradeoff(agregado: list, caminho=None):
    """Acurácia × diferença de paridade, com IC 95% nos dois eixos (Figura do artigo)."""
    ag = pd.DataFrame(agregado).set_index(["modelo", "cenario", "metrica"])
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=200)
    for m in dict.fromkeys(ag.index.get_level_values(0)):
        cenarios = list(dict.fromkeys(ag.loc[m].index.get_level_values(0)))
        pts = {}
        for c in cenarios:
            a, p = ag.loc[(m, c, "acuracia")], ag.loc[(m, c, "dif_paridade")]
            pts[c] = (a["media"], p["media"])
            ax.errorbar(a["media"], p["media"],
                        xerr=[[a["media"] - a["ic_inf"]], [a["ic_sup"] - a["media"]]],
                        yerr=[[p["media"] - p["ic_inf"]], [p["ic_sup"] - p["media"]]],
                        fmt=MARCADOR.get(c, "o"), ms=7, color=COR_MODELO.get(m, "#6250d6"),
                        mec="#fcfcfb", mew=1.2, elinewidth=1.2, capsize=2, zorder=3)
        if "original" in pts:
            for c in cenarios:
                if c != "original":
                    ax.plot(*zip(pts["original"], pts[c]), color=COR_MODELO.get(m), lw=1, ls="--", alpha=0.6, zorder=2)
            dx, ha = ((8, "left") if m == "regressao_logistica" else (-8, "right"))
            ax.annotate(_nome(m), pts["original"], xytext=(dx, 9), textcoords="offset points", ha=ha,
                        fontsize=8, color=TINTA, zorder=4,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    ax.set_xlabel("Acurácia (maior é melhor)", color=TINTA2, fontsize=9)
    ax.set_ylabel("Diferença de paridade demográfica\n(menor é melhor)", color=TINTA2, fontsize=9)
    from matplotlib.ticker import FuncFormatter
    virgula = FuncFormatter(lambda v, _: f"{v:.3f}".rstrip("0").rstrip(".").replace(".", ","))
    ax.xaxis.set_major_formatter(virgula)
    ax.yaxis.set_major_formatter(virgula)
    ax.grid(color=GRADE, lw=0.6)
    ax.tick_params(colors=TINTA2, labelsize=8)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(GRADE)
    # legenda: cor = modelo, forma = cenário
    from matplotlib.lines import Line2D
    itens = [Line2D([], [], color=COR_MODELO[m], marker="o", ls="", label=_nome(m))
             for m in dict.fromkeys(ag.index.get_level_values(0)) if m in COR_MODELO]
    itens += [Line2D([], [], color=TINTA2, marker=MARCADOR[c], ls="", label=CENARIOS[c])
              for c in dict.fromkeys(ag.index.get_level_values(1)) if c in CENARIOS]
    ax.legend(handles=itens, fontsize=7.5, frameon=False, loc="best")
    fig.tight_layout()
    if caminho:
        fig.savefig(caminho)
        plt.close(fig)
    return fig
