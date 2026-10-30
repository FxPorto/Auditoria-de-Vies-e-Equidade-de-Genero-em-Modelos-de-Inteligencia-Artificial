"""Base ACSIncome (DING et al., 2021) a partir dos microdados do American Community Survey (PUMS, 1 ano).

Reproduz a definição do ACSIncome do pacote folktables, sem depender dele:
  - pessoas com idade > 16, renda > US$ 100, horas trabalhadas > 0 e peso amostral >= 1;
  - alvo: renda pessoal (PINCP) acima de US$ 50 mil;
  - 10 variáveis: AGEP, COW, SCHL, MAR, OCCP, POBP, RELP, WKHP, SEX e RAC1P.

As colunas recebem os nomes equivalentes da Adult (age, workclass, ...), para que o mesmo protocolo,
os mesmos scripts e as mesmas opções (por exemplo, --remover-colunas relationship,marital-status) se
apliquem às duas bases sem adaptação.
"""
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .dados import _normalizar, hash_arquivo

URL = "https://www2.census.gov/programs-surveys/acs/data/pums/{ano}/1-Year/csv_p{uf}.zip"

# ACS -> nome equivalente na Adult
COLUNAS = {
    "AGEP": "age", "COW": "workclass", "SCHL": "education-num", "MAR": "marital-status",
    "OCCP": "occupation", "POBP": "place-of-birth", "RELP": "relationship", "WKHP": "hours-per-week",
    "SEX": "sex", "RAC1P": "race",
}
CATEGORICAS = ["workclass", "marital-status", "occupation", "place-of-birth", "relationship", "race"]
RACA = {1: "White", 2: "Black", 3: "American-Indian", 4: "Alaska-Native", 5: "Amer-Indian-Alaska-Native",
        6: "Asian", 7: "Pacific-Islander", 8: "Other", 9: "Two-or-more"}
# referência de tamanho publicada para CA 2018 (DING et al., 2021); usada só como alerta
REGISTROS_CONHECIDOS = {("CA", 2018): 195665}


def caminho_padrao(estado: str, ano: int, pasta: str | Path = "dados") -> Path:
    return Path(pasta) / f"acsincome_{estado.upper()}_{ano}.csv"


def preparar_acs(bruto: pd.DataFrame) -> pd.DataFrame:
    """Aplica o filtro e a definição do ACSIncome a um recorte de pessoas do PUMS."""
    df = bruto.copy()
    if "RELP" not in df.columns and "RELSHIPP" in df.columns:   # nome da variável a partir de 2019
        df = df.rename(columns={"RELSHIPP": "RELP"})
    df = df[(df.AGEP > 16) & (df.PINCP > 100) & (df.WKHP > 0) & (df.PWGTP >= 1)]
    out = pd.DataFrame({novo: df[antigo].to_numpy() for antigo, novo in COLUNAS.items()})
    out["sex"] = np.where(out["sex"].astype(int) == 1, "Male", "Female")
    out["race"] = out["race"].astype(int).map(RACA).fillna("Other")
    for c in [c for c in CATEGORICAS if c != "race"]:
        v = pd.to_numeric(out[c], errors="coerce").fillna(-1).astype(int)
        out[c] = c[:3] + "_" + v.astype(str)          # código como texto: tratado como categoria
    out["income"] = (df["PINCP"].to_numpy() > 50000).astype(int)
    return out.reset_index(drop=True)


def baixar_acsincome(estado: str = "CA", ano: int = 2018, destino: str | Path | None = None) -> Path:
    """Baixa o PUMS de pessoas (1 ano) do estado, aplica a definição do ACSIncome e grava um CSV local."""
    destino = Path(destino) if destino else caminho_padrao(estado, ano)
    url = URL.format(ano=ano, uf=estado.lower())
    with urllib.request.urlopen(url, timeout=300) as r:
        conteudo = r.read()
    with zipfile.ZipFile(io.BytesIO(conteudo)) as z:
        nome = next(n for n in z.namelist() if n.lower().startswith("psam_p") and n.lower().endswith(".csv"))
        cabecalho = pd.read_csv(z.open(nome), nrows=0).columns
        rel = "RELP" if "RELP" in cabecalho else "RELSHIPP"
        usar = [c for c in COLUNAS if c != "RELP"] + [rel, "PINCP", "PWGTP"]
        bruto = pd.read_csv(z.open(nome), usecols=usar)
    df = preparar_acs(bruto)
    destino.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destino, index=False, lineterminator="\n")
    return destino


def carregar_acsincome(estado: str = "CA", ano: int = 2018, caminho: str | Path | None = None,
                       baixar_se_faltar: bool = True) -> tuple[pd.DataFrame, dict]:
    caminho = Path(caminho) if caminho else caminho_padrao(estado, ano)
    origem = "cópia local"
    if not caminho.exists():
        if not baixar_se_faltar:
            raise FileNotFoundError(f"Base não encontrada em {caminho}.")
        try:
            baixar_acsincome(estado, ano, caminho)
            origem = f"ACS PUMS {ano} (1 ano), {estado.upper()}"
        except Exception as e:  # EX01
            raise RuntimeError(
                f"Não foi possível obter o ACS PUMS {ano} de {estado.upper()}. Verifique a internet "
                f"ou coloque a cópia local em {caminho}.") from e
    df = pd.read_csv(caminho, dtype={c: str for c in CATEGORICAS})
    df = _normalizar(df)
    info = {"caminho": str(caminho), "origem": origem, "hash": hash_arquivo(caminho),
            "versao": f"ACSIncome, ACS PUMS {ano} 1 ano, {estado.upper()}", "registros": len(df),
            "estado": estado.upper(), "ano": int(ano)}
    return df, info
