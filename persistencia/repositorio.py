"""Persistência: tabela execucoes (SQLite) e arquivo de previsões por execução (RF14, RF20, RF21).

- Só execuções concluídas são gravadas (RNF06, EX05).
- Registros e arquivos de previsões são somente leitura depois de gravados (RNF11).
- No modo Avaliador o banco é aberto apenas para leitura (EX09).
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ESQUEMA = """CREATE TABLE IF NOT EXISTS execucoes (
    id TEXT PRIMARY KEY, criado_em TEXT NOT NULL, modo TEXT NOT NULL, fonte TEXT NOT NULL,
    hash_dados TEXT NOT NULL, config TEXT NOT NULL, versoes TEXT NOT NULL, resultados TEXT NOT NULL,
    arquivo_previsoes TEXT, hash_previsoes TEXT)"""


class ModoSomenteLeitura(PermissionError):
    pass


def _hash(caminho: Path) -> str:
    return hashlib.sha256(caminho.read_bytes()).hexdigest()


class Repositorio:
    def __init__(self, pasta: str | Path = "execucoes", somente_leitura: bool = False):
        self.pasta = Path(pasta)
        self.db = self.pasta / "auditorias.db"
        self.pasta_prev = self.pasta / "previsoes"
        self.somente_leitura = somente_leitura
        if not somente_leitura:
            self.pasta_prev.mkdir(parents=True, exist_ok=True)
            with self._conectar() as con:
                con.execute(ESQUEMA)

    def _conectar(self):
        if self.somente_leitura:
            return sqlite3.connect(f"file:{self.db.as_posix()}?mode=ro", uri=True)
        return sqlite3.connect(self.db)

    # ------------------------------------------------------------------ gravação
    def salvar(self, resultado: dict, modo: str, fonte: str) -> str:
        if self.somente_leitura:
            raise ModoSomenteLeitura("Modo Avaliador: a aplicação está em modo somente leitura.")
        eid = uuid.uuid4().hex[:10]
        arq = self.pasta_prev / f"{eid}.csv.gz"
        tmp = arq.with_suffix(".tmp")
        resultado["previsoes"].to_csv(tmp, index=False, compression={"method": "gzip", "mtime": 0})
        os.replace(tmp, arq)
        corpo = {k: v for k, v in resultado.items() if k not in ("previsoes", "config", "versoes")}
        try:
            with self._conectar() as con:
                con.execute("INSERT INTO execucoes VALUES (?,?,?,?,?,?,?,?,?,?)", (
                    eid, datetime.now(timezone.utc).isoformat(timespec="seconds"), modo, fonte,
                    resultado.get("dados", {}).get("hash", ""), json.dumps(resultado["config"]),
                    json.dumps(resultado["versoes"]), json.dumps(corpo, default=float),
                    arq.name, _hash(arq)))
        except Exception:
            arq.unlink(missing_ok=True)       # falha de persistência: nada fica registrado
            raise
        os.chmod(arq, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
        return eid

    # ------------------------------------------------------------------ leitura
    def listar(self) -> pd.DataFrame:
        if not self.db.exists():
            return pd.DataFrame(columns=["id", "criado_em", "modo", "fonte", "hash_dados"])
        with self._conectar() as con:
            return pd.read_sql("SELECT id, criado_em, modo, fonte, hash_dados FROM execucoes "
                               "ORDER BY criado_em DESC", con)

    def carregar(self, eid: str) -> dict:
        with self._conectar() as con:
            con.row_factory = sqlite3.Row
            linha = con.execute("SELECT * FROM execucoes WHERE id = ?", (eid,)).fetchone()
        if linha is None:
            raise KeyError(f"Execução {eid} não encontrada.")
        res = json.loads(linha["resultados"])
        res.update({"id": linha["id"], "criado_em": linha["criado_em"], "modo": linha["modo"],
                    "fonte": linha["fonte"], "config": json.loads(linha["config"]),
                    "versoes": json.loads(linha["versoes"]), "hash_previsoes": linha["hash_previsoes"]})
        arq = self.pasta_prev / (linha["arquivo_previsoes"] or "")
        res["alerta_previsoes"] = None
        if arq.is_file():
            if _hash(arq) != linha["hash_previsoes"]:      # EX08
                res["alerta_previsoes"] = "O arquivo de previsões foi alterado depois da execução (hash diferente)."
            res["previsoes"] = pd.read_csv(arq, float_precision="round_trip")  # leitura exata dos escores
        else:
            res["previsoes"] = None
            res["alerta_previsoes"] = "Arquivo de previsões não encontrado."
        return res
