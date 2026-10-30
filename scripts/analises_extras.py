"""Análises complementares a partir das previsões já gravadas (sem treinar nada de novo).

Exemplo (na raiz do projeto, depois do experimento principal):
    python scripts/analises_extras.py --nome principal

Gera em resultados/<nome>/extras/:
  interseccional.csv          sexo × raça: taxa-base, seleção, TPR e precisão (média e IC nas repetições)
  calibracao.csv / .png       escore médio previsto × taxa real de renda alta, por sexo (cenários com escore)
  brier_por_grupo.csv         erro quadrático médio do escore, por sexo
  fronteira.csv / .png        acurácia × diferença de paridade para limiares por grupo (exploratória)

As previsões guardam a posição de cada pessoa na base (coluna "indice"); por isso a base usada tem de ser a
mesma do experimento. O script confere isso antes de começar.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from motor.config import CENARIOS, MODELOS  # noqa: E402
from motor.dados import carregar_base  # noqa: E402
from motor.estatistica import resumir  # noqa: E402
from motor.sintetico import gerar_adult_sintetico  # noqa: E402

COR_SEXO = {"Female": "#2a78d6", "Male": "#8a8a85"}
NOME_SEXO = {"Female": "Mulheres", "Male": "Homens"}
TINTA2, GRADE = "#52514e", "#e4e3df"
VIRGULA = FuncFormatter(lambda v, _: f"{v:.3f}".rstrip("0").rstrip(".").replace(".", ","))
MIN_POSITIVOS = 30   # abaixo disso, TPR e precisão do subgrupo são instáveis


def csv(df: pd.DataFrame, caminho: Path):
    df.to_csv(caminho, sep=";", decimal=",", index=False, encoding="utf-8-sig")


def _eixos(ax):
    ax.grid(color=GRADE, lw=0.6)
    ax.tick_params(colors=TINTA2, labelsize=8)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(GRADE)
    ax.xaxis.set_major_formatter(VIRGULA)
    ax.yaxis.set_major_formatter(VIRGULA)


def _taxas(g: pd.DataFrame) -> dict:
    pos = g.y_true == 1
    sel = g.y_pred == 1
    return {"pessoas": len(g), "positivos": int(pos.sum()),
            "taxa_base": pos.mean(), "selecao": sel.mean(),
            "tpr": (sel & pos).sum() / pos.sum() if pos.sum() else np.nan,
            "precisao": (sel & pos).sum() / sel.sum() if sel.sum() else np.nan}


def interseccional(prev: pd.DataFrame) -> pd.DataFrame:
    """Sexo × raça. Cada medida é calculada por semente e resumida nas repetições."""
    por_semente = (prev.groupby(["modelo", "cenario", "sexo", "raca", "semente"], sort=False)
                   .apply(lambda g: pd.Series(_taxas(g)), include_groups=False).reset_index())
    linhas = []
    for (m, c, s, r), g in por_semente.groupby(["modelo", "cenario", "sexo", "raca"], sort=False):
        lin = {"Modelo": MODELOS.get(m, m), "Cenário": CENARIOS.get(c, c), "Sexo registrado": s, "Raça": r,
               "Pessoas por teste (média)": round(g.pessoas.mean()),
               "Renda alta por teste (média)": round(g.positivos.mean())}
        for k, rot in [("taxa_base", "Taxa-base"), ("selecao", "Seleção"), ("tpr", "TPR"), ("precisao", "Precisão")]:
            res = resumir(g[k].to_numpy())
            lin[rot] = res["media"]
            lin[f"{rot} IC inf"], lin[f"{rot} IC sup"] = res["ic_inf"], res["ic_sup"]
        lin["Amostra pequena"] = "sim" if g.positivos.mean() < MIN_POSITIVOS else "não"
        linhas.append(lin)
    return pd.DataFrame(linhas).round(4)


def calibracao(prev: pd.DataFrame, pasta: Path):
    """Escore médio × taxa real em 10 faixas de escore, por sexo; e Brier por grupo e semente."""
    com_escore = prev[prev.score.notna()].copy()
    if com_escore.empty:
        return None, None
    com_escore["faixa"] = np.minimum((com_escore.score * 10).astype(int), 9)
    tab = (com_escore.groupby(["modelo", "cenario", "sexo", "faixa"], sort=False)
           .agg(pessoas=("y_true", "size"), escore_medio=("score", "mean"), taxa_real=("y_true", "mean"))
           .reset_index())
    tab = tab.sort_values(["modelo", "cenario", "sexo", "faixa"], kind="stable")
    tab["faixa"] = tab.faixa.map(lambda f: f"{f / 10:.1f} a {(f + 1) / 10:.1f}".replace(".", ","))
    com_escore["erro2"] = (com_escore.score - com_escore.y_true) ** 2
    brier_sem = com_escore.groupby(["modelo", "cenario", "sexo", "semente"], sort=False).erro2.mean().reset_index()
    brier = []
    for (m, c, s), g in brier_sem.groupby(["modelo", "cenario", "sexo"], sort=False):
        res = resumir(g.erro2.to_numpy())
        brier.append({"Modelo": MODELOS.get(m, m), "Cenário": CENARIOS.get(c, c), "Sexo registrado": s,
                      "Brier (média)": res["media"], "IC inf": res["ic_inf"], "IC sup": res["ic_sup"]})

    ordem = {k: i for i, k in enumerate([*MODELOS, *CENARIOS])}
    combos = sorted(dict.fromkeys(zip(tab.modelo, tab.cenario)), key=lambda mc: (ordem.get(mc[0], 99), ordem.get(mc[1], 99)))
    fig, eixos = plt.subplots(1, len(combos), figsize=(3.2 * len(combos), 3.4), dpi=200, sharey=True, squeeze=False)
    for ax, (m, c) in zip(eixos[0], combos):
        ax.plot([0, 1], [0, 1], color=GRADE, lw=1, ls="--", zorder=1)
        for s in ("Male", "Female"):
            g = tab[(tab.modelo == m) & (tab.cenario == c) & (tab.sexo == s) & (tab.pessoas >= 30)].sort_values("escore_medio")
            ax.plot(g.escore_medio, g.taxa_real, marker="o", ms=4, lw=1.5, color=COR_SEXO.get(s, "#6250d6"),
                    label=NOME_SEXO.get(s, s), zorder=3)
        ax.set_title(f"{MODELOS.get(m, m)}\n{CENARIOS.get(c, c)}", fontsize=8, color=TINTA2)
        ax.set_xlabel("Escore médio previsto", fontsize=8, color=TINTA2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        _eixos(ax)
    eixos[0][0].set_ylabel("Proporção real com renda alta", fontsize=8, color=TINTA2)
    eixos[0][0].legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(pasta / "calibracao.png")
    plt.close(fig)
    return tab.round(4), pd.DataFrame(brier).round(4)


def fronteira(prev: pd.DataFrame, agregado: pd.DataFrame | None, pasta: Path, passo: float = 0.025):
    """Para cada par de limiares (homens, mulheres), acurácia e diferença de paridade médias nas sementes.

    EXPLORATÓRIA: os limiares são varridos no próprio conjunto de teste, então a fronteira é otimista.
    Serve para mostrar onde as técnicas ficam em relação ao melhor equilíbrio possível, não para escolher limiar.
    """
    limiares = np.round(np.arange(passo, 1, passo), 4)
    base = prev[(prev.cenario == "original") & prev.score.notna()]
    linhas = []
    for m, gm in base.groupby("modelo", sort=False):
        acc = np.zeros((len(limiares), len(limiares)))
        dp = np.zeros_like(acc)
        sementes = gm.semente.unique()
        for _, g in gm.groupby("semente"):
            homem = (g.sexo == "Male").to_numpy()
            sc, y = g.score.to_numpy(), g.y_true.to_numpy()
            sel_h = sc[homem][None, :] >= limiares[:, None]          # limiar × pessoa
            sel_m = sc[~homem][None, :] >= limiares[:, None]
            certo_h = (sel_h == y[homem][None, :]).sum(axis=1)
            certo_m = (sel_m == y[~homem][None, :]).sum(axis=1)
            acc += (certo_h[:, None] + certo_m[None, :]) / len(y)
            dp += np.abs(sel_h.mean(axis=1)[:, None] - sel_m.mean(axis=1)[None, :])
        acc /= len(sementes)
        dp /= len(sementes)
        for i, th in enumerate(limiares):
            for j, tm in enumerate(limiares):
                linhas.append({"modelo": m, "limiar_homens": th, "limiar_mulheres": tm,
                               "acuracia": acc[i, j], "dif_paridade": dp[i, j]})
    tab = pd.DataFrame(linhas)
    # fronteira de Pareto: para cada nível de paridade, a maior acurácia possível
    tab["na_fronteira"] = False
    for m, g in tab.groupby("modelo"):
        ordem = g.sort_values(["dif_paridade", "acuracia"], ascending=[True, False])
        melhor = -1.0
        for idx, r in ordem.iterrows():
            if r.acuracia > melhor:
                tab.loc[idx, "na_fronteira"] = True
                melhor = r.acuracia

    from servicos.tabelas import COR_MODELO, MARCADOR
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=200)
    for m, g in tab.groupby("modelo", sort=False):
        f = g[g.na_fronteira].sort_values("dif_paridade")
        ax.plot(f.acuracia, f.dif_paridade, color=COR_MODELO.get(m, "#6250d6"), lw=1.2, alpha=0.8,
                label=f"Fronteira: {MODELOS.get(m, m)}")
        if agregado is not None:
            ag = agregado[agregado.modelo == m].set_index(["cenario", "metrica"])["media"]
            for c in dict.fromkeys(agregado[agregado.modelo == m].cenario):
                if (c, "acuracia") in ag and (c, "dif_paridade") in ag:
                    ax.plot(ag[(c, "acuracia")], ag[(c, "dif_paridade")], MARCADOR.get(c, "o"), ms=7,
                            color=COR_MODELO.get(m, "#6250d6"), mec="white", mew=1.2, zorder=3)
    from matplotlib.lines import Line2D
    itens = ax.get_legend_handles_labels()[0]
    itens += [Line2D([], [], color=TINTA2, marker=MARCADOR[c], ls="", label=CENARIOS[c]) for c in CENARIOS]
    ax.legend(handles=itens, fontsize=7.5, frameon=False, loc="best")
    ax.set_xlabel("Acurácia (maior é melhor)", color=TINTA2, fontsize=9)
    ax.set_ylabel("Diferença de paridade demográfica\n(menor é melhor)", color=TINTA2, fontsize=9)
    lo = tab[tab.na_fronteira].acuracia.min()
    ax.set_xlim(max(lo - 0.01, 0.5), tab.acuracia.max() + 0.005)
    ax.set_ylim(-0.005, tab[tab.na_fronteira].dif_paridade.max() + 0.02)
    _eixos(ax)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.3f}".rstrip("0").rstrip(".").replace(".", ",")))
    fig.text(0.01, 0.01, "Exploratória: limiares varridos no conjunto de teste (fronteira otimista).",
             fontsize=6.5, color=TINTA2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(pasta / "fronteira.png")
    plt.close(fig)
    return tab


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nome", required=True, help="pasta em resultados/ (ex.: principal)")
    p.add_argument("--saida", default=str(RAIZ / "resultados"))
    p.add_argument("--caminho", default=str(RAIZ / "dados" / "adult.csv"))
    p.add_argument("--fonte", choices=["adult", "acsincome", "sintetica"], default="adult", help="sintetica só para teste")
    p.add_argument("--estado", default="CA", help="ACSIncome: sigla do estado (padrão CA)")
    p.add_argument("--ano", type=int, default=2018, help="ACSIncome: ano (padrão 2018)")
    a = p.parse_args(argv)

    pasta = Path(a.saida) / a.nome
    arq = pasta / "previsoes.csv.gz"
    if not arq.exists():
        sys.exit(f"ERRO: não encontrei {arq}. Rode antes o rodar_experimento.py com --nome {a.nome}.")
    prev = pd.read_csv(arq, float_precision="round_trip")
    if a.fonte == "sintetica":
        df = gerar_adult_sintetico(8000, seed=0, vies=1.0)
    elif a.fonte == "acsincome":
        from motor.acs import carregar_acsincome
        df = carregar_acsincome(a.estado, a.ano, Path(a.caminho).parent / f"acsincome_{a.estado.upper()}_{a.ano}.csv", False)[0]
    else:
        df = carregar_base(a.caminho, False)[0]
    if prev.indice.max() >= len(df) or not np.array_equal(df["income"].to_numpy()[prev.indice], prev.y_true):
        sys.exit("ERRO: as previsões não batem com esta base (rótulos diferentes nas mesmas posições). "
                 "Use a mesma base do experimento (confira o hash com scripts/validar_base.py).")
    prev["sexo"] = prev.sexo.astype(str)
    prev["raca"] = df["race"].astype(str).to_numpy()[prev.indice]
    ag_arq = pasta / "agregado_completo.csv"
    agregado = pd.read_csv(ag_arq, sep=";", decimal=",") if ag_arq.exists() else None

    out = pasta / "extras"
    out.mkdir(exist_ok=True)
    print("Sexo × raça...")
    csv(interseccional(prev), out / "interseccional.csv")
    print("Calibração por sexo...")
    cal, brier = calibracao(prev, out)
    if cal is not None:
        csv(cal, out / "calibracao.csv")
        csv(brier, out / "brier_por_grupo.csv")
    print("Fronteira acurácia × paridade (exploratória)...")
    fr = fronteira(prev, agregado, out)
    csv(fr[fr.na_fronteira].round(4), out / "fronteira.csv")
    print(f"Arquivos em {out}")


if __name__ == "__main__":
    main()
