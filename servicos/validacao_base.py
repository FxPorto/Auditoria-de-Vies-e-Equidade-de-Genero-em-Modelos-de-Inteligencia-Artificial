"""Validação da base antes do experimento (Guia de execução, Passo 3).

Confere se o arquivo é a versão certa da Adult (OpenML 1590, 48.842 registros) e se o conteúdo bate com os
valores conhecidos da base. FALHA = não use o arquivo; ALERTA = confira antes de seguir.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from motor.dados import _normalizar, hash_arquivo

REGISTROS_ADULT = 48842
COLUNAS_ADULT = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
                 "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
                 "hours-per-week", "native-country", "income"]
COLUNAS_COM_AUSENTES = {"workclass", "occupation", "native-country"}

# contagens de versões incompletas que circulam (UCI / Kaggle)
VERSOES_CONHECIDAS = {
    32561: "parece a versão só de treino (adult.data), a mais comum no Kaggle",
    16281: "parece só o conjunto de teste (adult.test)",
    30162: "parece o treino sem as linhas com valores ausentes",
    15060: "parece o teste sem as linhas com valores ausentes",
    45222: "parece a base completa sem as linhas com valores ausentes",
}


def _item(out, verificacao, valor, situacao, referencia, dica=""):
    out.append({"verificacao": verificacao, "valor": valor, "referencia": referencia,
                "situacao": situacao, "dica": dica})


def validar(bruto: pd.DataFrame) -> list[dict]:
    """Checagens sobre o arquivo lido sem tratamento (pd.read_csv)."""
    out: list[dict] = []
    n = len(bruto)
    dica = "" if n == REGISTROS_ADULT else (
        VERSOES_CONHECIDAS.get(n, "versão desconhecida") + ". Use a OpenML 1590 baixada pelo sistema.")
    _item(out, "Número de registros", n, "OK" if n == REGISTROS_ADULT else "FALHA", "48.842", dica)

    colunas = [str(c).strip() for c in bruto.columns]
    faltam = [c for c in COLUNAS_ADULT if c not in colunas]
    sobram = [c for c in colunas if c not in COLUNAS_ADULT]
    dica = ""
    if any("." in c for c in sobram):
        dica = "Nomes com ponto (education.num etc.) indicam a versão do Kaggle."
    elif "class" in sobram:
        dica = "Alvo chamado 'class' indica o arquivo bruto do OpenML; deixe o sistema gerar a cópia."
    valor = "todas presentes" if not faltam and not sobram else (
        f"faltam: {', '.join(faltam) or '-'}; sobram: {', '.join(sobram) or '-'}")
    _item(out, "Colunas", valor, "OK" if not faltam and not sobram else "FALHA",
          "14 variáveis + income", dica)
    if "income" not in colunas or "sex" not in colunas:
        return out

    df = _normalizar(bruto.rename(columns=lambda c: str(c).strip()))
    classes = sorted(df["income"].unique().tolist())
    _item(out, "Alvo (income) com duas classes", str(classes), "OK" if classes == [0, 1] else "FALHA",
          "[0, 1] (>50K = 1)")
    sexos = sorted(df["sex"].dropna().unique().tolist())
    _item(out, "Sexo registrado", ", ".join(sexos), "OK" if sexos == ["Female", "Male"] else "FALHA",
          "Female, Male")
    if classes != [0, 1] or sexos != ["Female", "Male"]:
        return out

    def faixa(nome, v, lo, hi, ref):
        _item(out, nome, f"{v:.3f}".replace(".", ","), "OK" if lo <= v <= hi else "ALERTA", ref)

    faixa("Proporção com renda acima de US$ 50 mil", df["income"].mean(), 0.235, 0.245, "0,239")
    faixa("Proporção de homens", (df["sex"] == "Male").mean(), 0.660, 0.675, "0,668")
    taxa = df.groupby("sex")["income"].mean()
    faixa("Taxa-base dos homens", taxa["Male"], 0.29, 0.32, "0,304")
    faixa("Taxa-base das mulheres", taxa["Female"], 0.10, 0.12, "0,109")

    ausentes = df.drop(columns="income").isna().sum()
    ausentes = ausentes[ausentes > 0]
    fora = sorted(set(ausentes.index) - COLUNAS_COM_AUSENTES)
    valor = "; ".join(f"{c}: {v}" for c, v in ausentes.items()) or "nenhum"
    _item(out, "Valores ausentes", valor, "OK" if not fora and len(ausentes) else "ALERTA",
          "só em workclass, occupation e native-country (cerca de 2.800, 2.800 e 860)",
          f"Ausentes inesperados em: {', '.join(fora)}" if fora else (
              "" if len(ausentes) else "Nenhum ausente: as linhas incompletas podem ter sido removidas."))
    if "age" in df:
        idade = pd.to_numeric(df["age"], errors="coerce")
        ok = idade.min() >= 17 and idade.max() <= 90
        _item(out, "Faixa de idade", f"{idade.min():.0f} a {idade.max():.0f}", "OK" if ok else "ALERTA", "17 a 90")
    _item(out, "Linhas repetidas (informativo)", int(df.duplicated().sum()), "INFO",
          "existem na base original; não remover")
    return out


def arquivo_hash(caminho: str | Path) -> Path:
    """Arquivo onde o hash oficial da base fica registrado (formato do sha256sum)."""
    caminho = Path(caminho)
    return caminho.with_suffix(".sha256")


def hash_registrado(caminho: str | Path) -> str | None:
    reg = arquivo_hash(caminho)
    if not reg.exists():
        return None
    return reg.read_text(encoding="utf-8").split()[0].strip().lower()


def registrar_hash(caminho: str | Path) -> str:
    caminho = Path(caminho)
    h = hash_arquivo(caminho)
    arquivo_hash(caminho).write_text(f"{h}  {caminho.name}\n", encoding="utf-8", newline="\n")
    return h


def conferir_hash(caminho: str | Path) -> tuple[str, str | None]:
    """Devolve (hash atual, hash registrado ou None)."""
    return hash_arquivo(caminho), hash_registrado(caminho)
