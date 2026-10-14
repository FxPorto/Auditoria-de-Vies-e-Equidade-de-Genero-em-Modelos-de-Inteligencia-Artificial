"""Camada de serviços: valida entradas, coordena o motor e a persistência (Documentação, Seção 3.2)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from motor import dados, metricas, protocolo
from motor.config import Config
from motor.estatistica import resumir
from motor.sintetico import gerar_adult_sintetico
from persistencia.repositorio import Repositorio

FONTES = {"adult": "Adult Census Income", "sintetica": "Sintética (somente teste)"}


class ServicoAuditoria:
    def __init__(self, repositorio: Repositorio, caminho_base=dados.CAMINHO_PADRAO):
        self.repo = repositorio
        self.caminho_base = caminho_base

    # ------------------------------------------------------------------ dados
    def carregar_dados(self, fonte: str, n_sintetico: int = 8000) -> tuple[pd.DataFrame, dict]:
        if fonte == "adult":
            return dados.carregar_base(self.caminho_base)
        if fonte == "sintetica":
            df = gerar_adult_sintetico(n_sintetico, seed=0, vies=1.0)
            return df, {"origem": "sintética (somente teste)", "hash": dados.hash_dataframe(df),
                        "versao": "gerador sintético", "registros": len(df)}
        raise ValueError(f"Fonte desconhecida: {fonte}")

    # ------------------------------------------------------------------ treino (UC03)
    def executar(self, cfg: Config, fonte: str = "adult", progresso=None, df=None, info=None) -> dict:
        cfg.validar()
        if df is None:
            df, info = self.carregar_dados(fonte)
        X, y, s = dados.preparar(df, cfg)
        res = protocolo.executar_protocolo(X, y, s, cfg, info, progresso)
        res["id"] = self.repo.salvar(res, "treino", FONTES.get(fonte, fonte))
        return res

    # ------------------------------------------------------------------ importação (UC04)
    @staticmethod
    def validar_arquivo(df: pd.DataFrame, col_y, col_pred, col_s, col_score=None) -> list[str]:
        """RF18 / EX02 / EX03: recusa o arquivo antes de qualquer cálculo."""
        erros = []
        for rot, c in [("rótulo real", col_y), ("previsão", col_pred), ("sexo registrado", col_s)]:
            if not c or c not in df.columns:
                erros.append(f"Coluna obrigatória ausente: {rot}.")
        if erros:
            return erros
        for rot, c in [("rótulo real", col_y), ("previsão", col_pred)]:
            invalidas = int((~df[c].isin([0, 1])).sum())
            if invalidas:
                erros.append(f"A coluna de {rot} tem {invalidas} linhas com valores diferentes de 0 e 1.")
        if df[col_s].isna().any():
            erros.append(f"A coluna de sexo registrado tem {int(df[col_s].isna().sum())} linhas vazias.")
        if df[col_s].nunique() < 2:
            erros.append("O arquivo precisa ter ao menos dois grupos de sexo registrado.")
        elif df[col_s].nunique() > 10:
            erros.append(f"A coluna escolhida como sexo registrado tem {df[col_s].nunique()} valores diferentes; "
                         "confira se é a coluna certa.")
        if col_score:
            sc = pd.to_numeric(df[col_score], errors="coerce")
            fora = int((sc.isna() | (sc < 0) | (sc > 1)).sum())
            if fora:
                erros.append(f"A coluna de probabilidade tem {fora} linhas fora do intervalo de 0 a 1.")
        return erros

    def importar_previsoes(self, df: pd.DataFrame, col_y, col_pred, col_s, col_score=None,
                           nome_modelo="Modelo externo", nome_arquivo="arquivo.csv",
                           grupo_referencia="Male") -> dict:
        erros = self.validar_arquivo(df, col_y, col_pred, col_s, col_score)
        if erros:
            raise ValueError(" ".join(erros))
        y, yp, s = df[col_y].astype(int), df[col_pred].astype(int), df[col_s].astype(str)
        score = df[col_score].astype(float).to_numpy() if col_score else None
        m = metricas.achatar(metricas.calcular(y, yp, s, score, grupo_referencia))
        agregado = [{"modelo": nome_modelo, "cenario": "importado", "metrica": k, **resumir([v], 10)}
                    for k, v in m.items()]
        res = {"config": {"modo": "importacao", "colunas": [col_y, col_pred, col_s, col_score],
                          "grupo_referencia": grupo_referencia, "sementes": [0]},
               "dados": {"hash": dados.hash_dataframe(df), "origem": nome_arquivo, "registros": len(df)},
               "versoes": protocolo.versoes(), "grupos": sorted(s.unique().tolist()),
               "tempo_por_semente_s": [], "agregado": agregado, "pareado": [],
               "registros": [{"semente": 0, "modelo": nome_modelo, "cenario": "importado",
                              "hiperparametros": {}, "auc_validacao": None, "metricas": m}],
               "previsoes": pd.DataFrame({"semente": 0, "modelo": nome_modelo, "cenario": "importado",
                                          "indice": np.arange(len(df)), "y_true": y, "sexo": s,
                                          "score": score if score is not None else np.nan, "y_pred": yp})}
        res["id"] = self.repo.salvar(res, "importacao", f"CSV: {nome_arquivo}")
        return res

    # ------------------------------------------------------------------ verificação (RF20)
    @staticmethod
    def recalcular_das_previsoes(previsoes: pd.DataFrame, grupo_referencia="Male") -> dict:
        """Recalcula as métricas de cada repetição a partir do arquivo de previsões, sem treinar."""
        out = {}
        for (sem, mod, cen), g in previsoes.groupby(["semente", "modelo", "cenario"]):
            score = None if g["score"].isna().all() else g["score"].to_numpy()
            out[(int(sem), mod, cen)] = metricas.achatar(
                metricas.calcular(g["y_true"], g["y_pred"], g["sexo"].astype(str), score, grupo_referencia))
        return out

    def verificar_execucao(self, execucao: dict, tolerancia=1e-9) -> list[str]:
        """Compara métricas gravadas com as recalculadas do arquivo; devolve divergências."""
        if execucao.get("previsoes") is None:
            return ["Arquivo de previsões indisponível."]
        recal = self.recalcular_das_previsoes(execucao["previsoes"],
                                              execucao["config"].get("grupo_referencia", "Male"))
        problemas = []
        for r in execucao["registros"]:
            chave = (int(r["semente"]), r["modelo"], r["cenario"])
            for k, v in r["metricas"].items():
                w = recal.get(chave, {}).get(k)
                if w is None or (pd.isna(v) != pd.isna(w)) or (not pd.isna(v) and abs(v - w) > tolerancia):
                    problemas.append(f"{chave} {k}: gravado {v}, recalculado {w}")
        return problemas
