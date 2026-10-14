"""T10 (hash), RF20 (previsões armazenadas), RF21/EX09 (modo Avaliador), EX08 (arquivo alterado), RF18."""
import gzip
import os
import stat

import pandas as pd
import pytest

from motor import dados
from persistencia.repositorio import ModoSomenteLeitura, Repositorio
from servicos.auditoria_service import ServicoAuditoria


def test_t10_hash_muda_com_um_valor(tmp_path, df_vies):
    a = tmp_path / "a.csv"; df_vies.to_csv(a, index=False)
    b = tmp_path / "b.csv"; d2 = df_vies.copy(); d2.loc[0, "age"] += 1; d2.to_csv(b, index=False)
    assert dados.hash_arquivo(a) != dados.hash_arquivo(b)
    assert dados.hash_arquivo(a) == dados.hash_arquivo(a)


@pytest.fixture
def execucao(tmp_path, df_vies, cfg_rapido):
    serv = ServicoAuditoria(Repositorio(tmp_path / "exec"))
    res = serv.executar(cfg_rapido, "sintetica", df=df_vies, info={"hash": dados.hash_dataframe(df_vies)})
    return serv, res


def test_rf20_metricas_recalculadas_conferem(execucao):
    serv, res = execucao
    ex = serv.repo.carregar(res["id"])
    assert ex["alerta_previsoes"] is None
    assert serv.verificar_execucao(ex) == []


def test_ex08_alerta_quando_previsoes_sao_alteradas(execucao):
    serv, res = execucao
    arq = serv.repo.pasta_prev / f"{res['id']}.csv.gz"
    os.chmod(arq, stat.S_IWRITE | stat.S_IREAD)
    df = pd.read_csv(arq); df.loc[0, "y_pred"] = 1 - df.loc[0, "y_pred"]
    df.to_csv(arq, index=False, compression="gzip")
    assert "alterado" in serv.repo.carregar(res["id"])["alerta_previsoes"]


def test_rnf11_arquivo_de_previsoes_somente_leitura(execucao):
    serv, res = execucao
    arq = serv.repo.pasta_prev / f"{res['id']}.csv.gz"
    assert not os.access(arq, os.W_OK) or os.geteuid() == 0   # root ignora permissões
    assert not (os.stat(arq).st_mode & stat.S_IWUSR)


def test_rf21_modo_avaliador_recusa_gravacao(execucao):
    serv, res = execucao
    leitor = Repositorio(serv.repo.pasta, somente_leitura=True)
    assert res["id"] in leitor.listar()["id"].tolist()
    assert leitor.carregar(res["id"])["id"] == res["id"]
    with pytest.raises(ModoSomenteLeitura):
        leitor.salvar(res, "treino", "x")


def test_rf18_validacao_do_arquivo_importado():
    ok = pd.DataFrame({"y": [0, 1, 1, 0], "p": [0, 1, 0, 0], "sexo": ["F", "M", "F", "M"], "prob": [.1, .9, .4, .2]})
    assert ServicoAuditoria.validar_arquivo(ok, "y", "p", "sexo", "prob") == []
    assert "sexo registrado" in ServicoAuditoria.validar_arquivo(ok.drop(columns="sexo"), "y", "p", "sexo")[0]
    ruim = ok.assign(p=[0, 2, 1, 0])
    assert "1 linhas" in ServicoAuditoria.validar_arquivo(ruim, "y", "p", "sexo")[0]
    assert "sexo registrado" in ServicoAuditoria.validar_arquivo(ok, "y", "p", None)[0]
    assert "valores diferentes" in ServicoAuditoria.validar_arquivo(
        pd.DataFrame({"y": [0, 1] * 10, "p": [0, 1] * 10, "prob": [i / 20 for i in range(20)]}), "y", "p", "prob")[0]


def test_importacao_registra_e_calcula(tmp_path):
    serv = ServicoAuditoria(Repositorio(tmp_path / "exec"))
    df = pd.DataFrame({"y": [0, 1, 1, 0] * 25, "p": [0, 1, 0, 0] * 25, "sexo": ["F", "M", "F", "M"] * 25,
                       "prob": [.1, .9, .4, .2] * 25})
    res = serv.importar_previsoes(df, "y", "p", "sexo", "prob", "Externo", "x.csv")
    ex = serv.repo.carregar(res["id"])
    assert ex["modo"] == "importacao" and serv.verificar_execucao(ex) == []
