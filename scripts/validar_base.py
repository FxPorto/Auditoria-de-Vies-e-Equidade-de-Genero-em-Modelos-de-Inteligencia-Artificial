"""Valida a base antes do experimento (Guia de execução, Passo 3).

Exemplos (na raiz do projeto):
    python scripts/validar_base.py                        # valida dados/adult.csv (baixa se faltar)
    python scripts/validar_base.py --registrar            # valida e registra o hash oficial (só a 1ª vez)
    python scripts/validar_base.py --caminho outra.csv    # confere outro arquivo, sem baixar nada
    python scripts/validar_base.py --base acsincome --estado CA --ano 2018 [--registrar]   # replicação ACSIncome

Mostra as checagens (OK / ALERTA / FALHA / INFO) e o hash SHA-256, e grava o mesmo texto em
evidencias/validacao_base.txt. Termina com código 1 se houver FALHA.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import pandas as pd  # noqa: E402

from motor.dados import ID_OPENML_ADULT, baixar_adult  # noqa: E402
from motor import acs  # noqa: E402
from servicos.validacao_base import arquivo_hash, conferir_hash, registrar_hash, validar, validar_acs  # noqa: E402

PADRAO = RAIZ / "dados" / "adult.csv"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", choices=["adult", "acsincome"], default="adult")
    p.add_argument("--estado", default="CA", help="ACSIncome: sigla do estado (padrão CA)")
    p.add_argument("--ano", type=int, default=2018, help="ACSIncome: ano do ACS PUMS (padrão 2018)")
    p.add_argument("--caminho", default=None, help="arquivo a validar (padrão: dados/adult.csv)")
    p.add_argument("--registrar", action="store_true",
                   help="grava o hash em <arquivo>.sha256 (fazer uma vez e versionar no git)")
    p.add_argument("--evidencia", default=None, help="padrão: evidencias/validacao_<base>.txt")
    a = p.parse_args(argv)

    linhas: list[str] = []
    if a.base == "acsincome":
        caminho = Path(a.caminho) if a.caminho else acs.caminho_padrao(a.estado, a.ano, RAIZ / "dados")
        if not caminho.exists():
            print(f"Baixando o ACS PUMS {a.ano} de {a.estado.upper()} (pode levar alguns minutos)...")
        try:
            df_acs, _ = acs.carregar_acsincome(a.estado, a.ano, caminho)
        except Exception as e:  # EX01
            print(f"Não foi possível obter a base: {e}")
            return 1
        checks = validar_acs(df_acs, acs.REGISTROS_CONHECIDOS.get((a.estado.upper(), a.ano)))
        return _concluir(caminho, checks, a, linhas)
    caminho = Path(a.caminho) if a.caminho else PADRAO
    if not caminho.exists():
        if caminho.resolve() != PADRAO.resolve():
            print(f"Arquivo não encontrado: {caminho}")
            return 1
        print(f"Baixando a base Adult do OpenML (id {ID_OPENML_ADULT})...")
        try:
            baixar_adult(caminho)
        except Exception as e:  # EX01
            print("Não foi possível baixar a base. Verifique a internet ou coloque a cópia em dados/adult.csv.")
            print(f"Detalhe: {e}")
            return 1

    bruto = pd.read_csv(caminho, na_values=["?"], skipinitialspace=True)
    checks = validar(bruto)
    return _concluir(caminho, checks, a, linhas)


def _concluir(caminho, checks, a, linhas) -> int:
    def dizer(txt=""):
        print(txt)
        linhas.append(txt)

    dizer(f"Arquivo: {caminho}")
    dizer()
    for c in checks:
        dizer(f"[{c['situacao']:6}] {c['verificacao']}: {c['valor']}  (esperado: {c['referencia']})")
        if c["dica"]:
            dizer(f"         -> {c['dica']}")
    falhas = [c for c in checks if c["situacao"] == "FALHA"]
    alertas = [c for c in checks if c["situacao"] == "ALERTA"]

    atual, registrado = conferir_hash(caminho)
    dizer()
    dizer(f"Hash SHA-256: {atual}")
    if a.registrar and not falhas:
        if registrado and registrado != atual:
            dizer(f"[FALHA ] Já existe outro hash registrado em {arquivo_hash(caminho).name}. "
                  "Apague esse arquivo só se o grupo decidiu trocar a base.")
            falhas.append({"verificacao": "hash"})
        else:
            registrar_hash(caminho)
            dizer(f"[OK    ] Hash registrado em {arquivo_hash(caminho).name}. "
                  "Façam commit deste arquivo junto com a base.")
    elif registrado is None:
        dizer("[ALERTA] Nenhum hash registrado ainda. Quem baixou a base primeiro roda com --registrar.")
    elif registrado == atual:
        dizer("[OK    ] Hash igual ao registrado: é a mesma base do grupo.")
    else:
        dizer(f"[FALHA ] Hash diferente do registrado ({registrado[:16]}...). "
              f"Não é a base do grupo: use a do repositório (git checkout -- dados/{Path(caminho).name}).")
        falhas.append({"verificacao": "hash"})

    dizer()
    if falhas:
        dizer(f"Resultado: NÃO USE esta base ({len(falhas)} falha(s)).")
    elif alertas:
        dizer(f"Resultado: base aceita com {len(alertas)} alerta(s). Confira antes de seguir.")
    else:
        dizer("Resultado: base validada. Pode seguir para o piloto.")

    ev = Path(a.evidencia) if a.evidencia else RAIZ / "evidencias" / (
        "validacao_base.txt" if a.base == "adult" else f"validacao_acsincome_{a.estado.upper()}_{a.ano}.txt")
    ev.parent.mkdir(parents=True, exist_ok=True)
    ev.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    print(f"\n(Evidência gravada em {ev})")
    return 1 if falhas else 0


if __name__ == "__main__":
    sys.exit(main())
