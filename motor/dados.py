"""Carga, identificação (hash) e preparação da base (RF01, RF02, RF16)."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config

ID_OPENML_ADULT = 1590
CAMINHO_PADRAO = Path("dados") / "adult.csv"


def hash_arquivo(caminho: str | Path) -> str:
    """SHA-256 dos bytes do arquivo; qualquer alteração gera outro hash (RF16)."""
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def hash_dataframe(df: pd.DataFrame) -> str:
    return hashlib.sha256(df.to_csv(index=False).encode("utf-8")).hexdigest()


def _normalizar(df: pd.DataFrame, alvo: str = "income") -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if c != alvo and not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype("object").where(df[c].notna(), np.nan)
            df[c] = df[c].map(lambda v: v.strip() if isinstance(v, str) else v).replace({"?": np.nan})
    if not pd.api.types.is_numeric_dtype(df[alvo]):
        df[alvo] = df[alvo].astype(str).str.contains(">50K").astype(int)
    df[alvo] = df[alvo].astype(int)
    return df


def baixar_adult(destino: str | Path = CAMINHO_PADRAO) -> Path:
    """Baixa a base Adult do OpenML (id 1590) e grava uma cópia local em CSV."""
    from fairlearn.datasets import fetch_adult

    dados = fetch_adult(as_frame=True)
    df = dados.data.copy()          # só as variáveis; o alvo entra à parte (evita vazamento)
    df["income"] = dados.target.astype(str).str.contains(">50K").astype(int)
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    # \n fixo: o mesmo arquivo (e o mesmo hash) em Windows, Mac e Linux
    _normalizar(df).to_csv(destino, index=False, lineterminator="\n")
    return destino


def carregar_base(caminho: str | Path = CAMINHO_PADRAO, baixar_se_faltar: bool = True) -> tuple[pd.DataFrame, dict]:
    """Lê a cópia local (ou baixa na primeira vez) e devolve (df, info com hash e origem)."""
    caminho = Path(caminho)
    origem = "cópia local"
    if not caminho.exists():
        if not baixar_se_faltar:
            raise FileNotFoundError(f"Base não encontrada em {caminho}.")
        try:
            baixar_adult(caminho)
            origem = f"OpenML (id {ID_OPENML_ADULT})"
        except Exception as e:  # EX01
            raise RuntimeError(
                "Não foi possível obter a base Adult. Verifique a conexão com a internet "
                f"ou coloque a cópia local em {caminho}."
            ) from e
    df = _normalizar(pd.read_csv(caminho, na_values=["?"]))
    info = {"caminho": str(caminho), "origem": origem, "hash": hash_arquivo(caminho),
            "versao": f"OpenML {ID_OPENML_ADULT}", "registros": len(df)}
    return df, info


def resumo(df: pd.DataFrame, alvo: str = "income", sensivel: str = "sex") -> dict:
    """Resumo da base para a área Dados (RF02)."""
    ausentes = df.isna().sum()
    por_grupo = df.groupby(sensivel)[alvo].agg(["count", "mean"]).rename(
        columns={"count": "registros", "mean": "taxa_base"})
    por_grupo["proporcao"] = por_grupo["registros"] / len(df)
    return {
        "registros": int(len(df)),
        "variaveis": int(df.shape[1] - 1),
        "ausentes": {c: int(v) for c, v in ausentes.items() if v > 0},
        "prevalencia": float(df[alvo].mean()),
        "por_grupo": por_grupo.reset_index().to_dict(orient="records"),
    }


def preparar(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Separa X (entradas), y (alvo) e s (sexo registrado) conforme as decisões da configuração."""
    y = df[cfg.alvo].astype(int).reset_index(drop=True)
    s = df[cfg.sensivel].astype(str).reset_index(drop=True)
    remover = [cfg.alvo] + [c for c in cfg.colunas_removidas if c in df.columns]
    if not cfg.usar_sexo_como_entrada:
        remover.append(cfg.sensivel)
    X = df.drop(columns=remover).reset_index(drop=True)
    return X, y, s
