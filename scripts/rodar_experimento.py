"""Executa o protocolo completo e gera os arquivos do artigo (Plano, Partes 4 e 5).

Exemplos (na raiz do projeto):
    python scripts/rodar_experimento.py --rapido --fonte sintetica      # teste de fumaça, ~1 min
    python scripts/rodar_experimento.py --sementes 2 --nome piloto       # piloto na base Adult
    python scripts/rodar_experimento.py --nome principal                 # 20 sementes, cenário principal
    python scripts/rodar_experimento.py --restricao equalized_odds --nome robustez_eo
    python scripts/rodar_experimento.py --usar-sexo --nome robustez_com_sexo
    python scripts/rodar_experimento.py --sementes 10 --nome estabilidade_10
    python scripts/rodar_experimento.py --restricao true_positive_rate_parity --nome limiar_tpr
    python scripts/rodar_experimento.py --remover-colunas relationship,marital-status --nome sem_proxies
    python scripts/rodar_experimento.py --fonte acsincome --estado CA --ano 2018 --nome acs_ca_2018

Saída em resultados/<nome>/ (CSV com ; e vírgula decimal, para abrir no Excel).
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import pandas as pd  # noqa: E402

from motor.config import RESTRICOES, Config  # noqa: E402
from persistencia.repositorio import Repositorio  # noqa: E402
from servicos import relatorio, sanidade, tabelas  # noqa: E402
from servicos.auditoria_service import ServicoAuditoria  # noqa: E402
from servicos.validacao_base import hash_registrado  # noqa: E402


def csv(df: pd.DataFrame, caminho: Path):
    df.to_csv(caminho, sep=";", decimal=",", index=False, encoding="utf-8-sig")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fonte", choices=["adult", "acsincome", "sintetica"], default="adult")
    p.add_argument("--estado", default="CA", help="ACSIncome: sigla do estado dos EUA (padrão CA)")
    p.add_argument("--ano", type=int, default=2018, help="ACSIncome: ano do ACS PUMS 1 ano (padrão 2018)")
    p.add_argument("--caminho", default=str(RAIZ / "dados" / "adult.csv"), help="cópia local da base Adult")
    p.add_argument("--sementes", type=int, default=20)
    p.add_argument("--restricao", choices=list(RESTRICOES), default="demographic_parity")
    p.add_argument("--remover-colunas", default="", help="colunas extras a remover, separadas por vírgula "
                   "(ex.: relationship,marital-status para testar os proxies do sexo)")
    p.add_argument("--usar-sexo", action="store_true", help="inclui o sexo nas variáveis de entrada")
    p.add_argument("--rapido", action="store_true", help="grade reduzida e 50 árvores (só para teste)")
    p.add_argument("--nome", default=None, help="nome da pasta de saída")
    p.add_argument("--saida", default=str(RAIZ / "resultados"))
    p.add_argument("--execucoes", default=str(RAIZ / "execucoes"))
    a = p.parse_args(argv)

    cfg = Config(sementes=list(range(a.sementes)), restricao=a.restricao, usar_sexo_como_entrada=a.usar_sexo)
    extras = tuple(c.strip() for c in a.remover_colunas.split(",") if c.strip())
    if extras:
        cfg.colunas_removidas = tuple(cfg.colunas_removidas) + extras
    if a.rapido:
        cfg.sementes = list(range(min(a.sementes, 2)))
        cfg.grades = {"regressao_logistica": {"C": [1.0]}, "random_forest": {"max_depth": [10], "min_samples_leaf": [5]}}
        cfg.rf_n_estimators, cfg.n_bootstrap = 50, 500

    servico = ServicoAuditoria(Repositorio(a.execucoes), a.caminho, acs_estado=a.estado, acs_ano=a.ano)
    print(f"Carregando dados ({a.fonte})...")
    df, info = servico.carregar_dados(a.fonte)
    print(f"  {info['registros']} registros · origem: {info['origem']} · hash {info['hash'][:16]}")
    faltam = [c for c in extras if c not in df.columns]
    if faltam:
        sys.exit(f"ERRO: colunas inexistentes em --remover-colunas: {', '.join(faltam)}")
    if a.fonte in ("adult", "acsincome"):
        registrado = hash_registrado(info["caminho"])
        if registrado and registrado != info["hash"]:
            sys.exit(f"ERRO: a base não é a registrada pelo grupo (hash diferente de {Path(info['caminho']).with_suffix('.sha256').name}). "
                     "Rode scripts/validar_base.py e use a base do repositório.")
        if registrado is None:
            print("  Aviso: hash da base ainda não registrado (scripts/validar_base.py --registrar).")

    inicio = time.perf_counter()

    def progresso(i, total, msg):
        decorrido = time.perf_counter() - inicio
        print(f"  [{i}/{total}] {msg} · {decorrido / 60:.1f} min")

    res = servico.executar(cfg, a.fonte, progresso, df=df, info=info)
    total = time.perf_counter() - inicio

    nome = a.nome or f"{time.strftime('%Y%m%d-%H%M')}_{res['id']}"
    out = Path(a.saida) / nome
    out.mkdir(parents=True, exist_ok=True)
    numerica, t1 = tabelas.tabela1(res["agregado"])
    csv(t1, out / "tabela1_formatada.csv")
    numerica.to_csv(out / "tabela1_numerica.csv", sep=";", decimal=",", encoding="utf-8-sig")
    csv(tabelas.tabela2(res["agregado"], res["grupos"]), out / "tabela2_por_grupo.csv")
    csv(tabelas.tabela3(res["pareado"]), out / "tabela3_diferencas_pareadas.csv")
    csv(pd.DataFrame(res["agregado"]), out / "agregado_completo.csv")
    csv(pd.DataFrame(res["pareado"]), out / "pareado_completo.csv")
    csv(pd.DataFrame([{"semente": r["semente"], "modelo": r["modelo"], "cenario": r["cenario"],
                       "hiperparametros": str(r["hiperparametros"]), "auc_validacao": r["auc_validacao"],
                       **r["metricas"]} for r in res["registros"]]), out / "por_semente.csv")
    tabelas.figura_tradeoff(res["agregado"], out / "figura_tradeoff.png")
    tabelas.figura_tradeoff(res["agregado"], out / "figura_tradeoff.pdf")
    from motor.acs import REGISTROS_CONHECIDOS
    esperado = REGISTROS_CONHECIDOS.get((a.estado.upper(), a.ano)) if a.fonte == "acsincome" else None
    checks = sanidade.checar(df, res["agregado"], base_adult=(a.fonte == "adult"), registros_esperados=esperado)
    csv(pd.DataFrame(checks), out / "sanidade.csv")
    ex = servico.repo.carregar(res["id"])
    (out / "relatorio.html").write_text(relatorio.gerar_html(ex), encoding="utf-8")
    (out / "execucao.json").write_text(relatorio.exportar_json(ex), encoding="utf-8")
    shutil.copy(Path(a.execucoes) / "previsoes" / f"{res['id']}.csv.gz", out / "previsoes.csv.gz")

    print(f"\nExecução {res['id']} concluída em {total / 60:.1f} min "
          f"({total / len(cfg.sementes):.0f} s por semente). Arquivos em {out}")
    print("\nChecagens de sanidade:")
    for c in checks:
        print(f"  [{c['situacao']:6}] {c['verificacao']}: {c['valor']} (referência: {c['referencia']})")
    if a.fonte == "sintetica":
        print("\nATENÇÃO: base sintética. Estes números NÃO podem ser usados no artigo.")
    return res


if __name__ == "__main__":
    main()
