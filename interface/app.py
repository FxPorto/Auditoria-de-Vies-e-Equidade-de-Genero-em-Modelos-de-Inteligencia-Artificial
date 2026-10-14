"""Painel de auditoria de viés e equidade (interface web).

Rodar na raiz do projeto:
    streamlit run interface/app.py                    # modo Pesquisador
    MODO=avaliador streamlit run interface/app.py     # modo Avaliador (somente leitura)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from motor import dados as mdados, metricas  # noqa: E402
from motor.config import CENARIOS, MODELOS, RESTRICOES, Config  # noqa: E402
from persistencia.repositorio import Repositorio  # noqa: E402
from servicos import relatorio, tabelas  # noqa: E402
from servicos.auditoria_service import FONTES, ServicoAuditoria  # noqa: E402

AVALIADOR = os.environ.get("MODO", "pesquisador").lower() == "avaliador"
PASTA = os.environ.get("PASTA_EXECUCOES", str(RAIZ / "execucoes"))
BASE = os.environ.get("CAMINHO_BASE", str(RAIZ / "dados" / "adult.csv"))
AVISO = ("Os resultados descrevem diferenças estatísticas entre grupos. A ferramenta não classifica "
         "um modelo como justo ou injusto; essa leitura depende do contexto de uso.")

st.set_page_config(page_title="Auditoria de viés e equidade", page_icon="⚖️", layout="wide")


@st.cache_resource
def servico() -> ServicoAuditoria:
    return ServicoAuditoria(Repositorio(PASTA, somente_leitura=AVALIADOR), BASE)


@st.cache_data(show_spinner="Carregando a base…")
def carregar(fonte: str):
    return servico().carregar_dados(fonte)


def abrir(eid: str):
    st.session_state["execucao"] = servico().repo.carregar(eid)


def nome(m):
    return MODELOS.get(m, m)


# ====================================================================== cabeçalho
st.title("Auditoria de viés e equidade")
if AVALIADOR:
    st.info("Modo Avaliador · somente leitura. Consulte resultados, histórico e relatórios; a execução está desabilitada.")
else:
    st.caption("Compare modelos de classificação por desempenho e pelas diferenças entre grupos de sexo registrado.")

# ====================================================================== barra lateral (Pesquisador)
if not AVALIADOR:
    with st.sidebar:
        st.header("Nova auditoria")
        fonte = st.selectbox("Base de dados", list(FONTES), format_func=FONTES.get,
                             help="A base Adult é baixada do OpenML na primeira vez e guardada em dados/adult.csv.")
        n_sem = int(st.number_input("Repetições (sementes)", 1, 50, 20,
                                    help="O protocolo do artigo usa 20. Use 2 para um teste rápido."))
        modelos = st.multiselect("Modelos", list(MODELOS), default=list(MODELOS), format_func=MODELOS.get)
        tecnicas = st.multiselect("Técnicas de mitigação", ["reponderacao", "limiar_por_grupo"],
                                  default=["reponderacao", "limiar_por_grupo"], format_func=CENARIOS.get)
        restricao = st.selectbox("Restrição do limiar por grupo", list(RESTRICOES), format_func=RESTRICOES.get)
        usar_sexo = st.checkbox("Incluir o sexo como variável de entrada", value=False,
                                help="No cenário principal o sexo é usado só para medir e mitigar.")
        if fonte == "sintetica":
            st.warning("Base sintética: serve só para testar a ferramenta. Não use estes números no artigo.")
        if st.button("Auditar modelos", type="primary", disabled=not modelos):
            cfg = Config(sementes=list(range(n_sem)), modelos=modelos, tecnicas=tecnicas,
                         restricao=restricao, usar_sexo_como_entrada=usar_sexo)
            barra = st.progress(0.0, text="Preparando…")
            try:
                df, info = carregar(fonte)
                res = servico().executar(cfg, fonte, lambda i, t, m: barra.progress(i / t, text=m), df=df, info=info)
                abrir(res["id"])
                barra.empty()
                st.success(f"Auditoria {res['id']} concluída e registrada no histórico.")
            except Exception as e:  # RNF12: sem detalhes internos
                barra.empty()
                st.error(f"Não foi possível concluir a auditoria ({type(e).__name__}). Nenhuma execução foi registrada.")

        with st.expander("Importar previsões (CSV)"):
            arq = st.file_uploader("Arquivo de previsões", type="csv",
                                   help="Uma linha por pessoa: rótulo real (0/1), previsão (0/1), sexo e, se houver, probabilidade.")
            if arq is not None:
                dfi = pd.read_csv(arq)
                cols = dfi.columns.tolist()
                opcoes = [None] + cols

                def sugerir(*nomes):  # pré-seleciona pelo nome; sem correspondência fica "(selecione)" (RF18)
                    return next((opcoes.index(c) for c in cols if c.strip().lower() in nomes), 0)

                rotulo = lambda c: "(selecione)" if c is None else c  # noqa: E731
                c_y = st.selectbox("Rótulo real", opcoes, sugerir("y_true", "y", "rotulo", "label", "income"),
                                   format_func=rotulo)
                c_p = st.selectbox("Previsão (0/1)", opcoes, sugerir("y_pred", "previsao", "pred", "prediction"),
                                   format_func=rotulo)
                c_s = st.selectbox("Sexo registrado", opcoes, sugerir("sexo", "sex", "sexo_registrado", "gender"),
                                   format_func=rotulo)
                c_sc = st.selectbox("Probabilidade (opcional)", opcoes, sugerir("score", "prob", "probabilidade"),
                                    format_func=lambda c: "(nenhuma)" if c is None else c)
                nm = st.text_input("Nome do modelo", "Modelo externo")
                if st.button("Auditar previsões"):
                    try:
                        res = servico().importar_previsoes(dfi, c_y, c_p, c_s, c_sc, nm, arq.name)
                        abrir(res["id"])
                        st.success(f"Previsões auditadas na execução {res['id']}.")
                    except ValueError as e:
                        st.error(str(e))

ex = st.session_state.get("execucao")
aba_dados, aba_res, aba_sim, aba_hist = st.tabs(["Dados", "Resultados", "Simulador", "Histórico e relatório"])

# ====================================================================== Dados (UC01)
with aba_dados:
    fonte_d = st.radio("Base", list(FONTES), format_func=FONTES.get, horizontal=True, key="fonte_dados")
    try:
        df, info = carregar(fonte_d)
        r = mdados.resumo(df)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Registros", f"{r['registros']:,}".replace(",", "."))
        c2.metric("Variáveis", r["variaveis"])
        c3.metric("Renda alta", f"{r['prevalencia'] * 100:.1f}%".replace(".", ","))
        c4.metric("Hash", info["hash"][:12], help=f"SHA-256 completo: {info['hash']}")
        st.caption(f"Origem: {info['origem']} · versão: {info.get('versao', '')}")
        g = pd.DataFrame(r["por_grupo"]).rename(columns={"sex": "Sexo registrado", "registros": "Registros",
                                                         "taxa_base": "Taxa-base de renda alta", "proporcao": "Proporção"})
        st.subheader("Por sexo registrado")
        st.dataframe(g, hide_index=True, column_config={
            "Taxa-base de renda alta": st.column_config.NumberColumn(format="%.3f"),
            "Proporção": st.column_config.NumberColumn(format="%.3f")})
        if r["ausentes"]:
            st.subheader("Valores ausentes")
            st.dataframe(pd.Series(r["ausentes"], name="Ausentes").to_frame())
        with st.expander("Amostra das primeiras 200 linhas"):
            st.dataframe(df.head(200), hide_index=True)
    except Exception as e:
        st.error(f"Não foi possível carregar a base ({type(e).__name__}). Verifique a conexão ou a cópia local em {BASE}.")

# ====================================================================== Resultados (UC05–UC08)
with aba_res:
    if ex is None:
        st.info("Nenhuma execução carregada. Abra uma no Histórico" + ("." if AVALIADOR else " ou execute uma auditoria pela barra lateral."))
    else:
        st.subheader(f"Execução {ex['id']}")
        st.caption(f"{ex['criado_em']} · {ex['fonte']} · hash dos dados {ex['dados'].get('hash', '')[:12]} · "
                   f"{len(ex['config'].get('sementes', []))} repetição(ões)")
        if ex.get("alerta_previsoes"):
            st.warning(ex["alerta_previsoes"])
        ag = pd.DataFrame(ex["agregado"]).set_index(["modelo", "cenario", "metrica"])
        for m in dict.fromkeys(ag.index.get_level_values(0)):
            st.markdown(f"**{nome(m)}**")
            cens = list(dict.fromkeys(ag.loc[m].index.get_level_values(0)))
            cols = st.columns(len(cens))
            for col, c in zip(cols, cens):
                a, p = ag.loc[(m, c, "acuracia")], ag.loc[(m, c, "dif_paridade")]
                col.metric(f"{CENARIOS.get(c, c)} · acurácia", tabelas.num(a["media"]),
                           help=f"IC 95%: {tabelas.num(a['ic_inf'])} a {tabelas.num(a['ic_sup'])}. {metricas.DESCRICOES['acuracia'][1]}")
                col.metric("Dif. de paridade demográfica", tabelas.num(p["media"]),
                           help=f"IC 95%: {tabelas.num(p['ic_inf'])} a {tabelas.num(p['ic_sup'])}. {metricas.DESCRICOES['dif_paridade'][1]}")
        st.subheader("Desempenho e equidade (média e IC 95%)")
        st.dataframe(tabelas.tabela1(ex["agregado"])[1], hide_index=True)

        st.subheader("Por sexo registrado")
        opc = {"selecao": "Taxa de seleção", "tpr": "TPR", "fpr": "FPR", "taxa_base": "Taxa-base"}
        k = st.radio("Métrica", list(opc), format_func=opc.get, horizontal=True,
                     help="Seleção: % classificado como renda alta. TPR: acerto entre quem tem renda alta. "
                          "FPR: erro entre quem não tem. Taxa-base: proporção real de renda alta.")
        linhas = []
        for (m, c, met), row in ag.iterrows():
            if met.startswith(f"{k}_"):
                linhas.append({"Cenário": f"{nome(m)} · {CENARIOS.get(c, c)}", "Grupo": met[len(k) + 1:], "valor": row["media"]})
        if linhas:
            st.bar_chart(pd.DataFrame(linhas).pivot(index="Cenário", columns="Grupo", values="valor"), stack=False)
        st.dataframe(tabelas.tabela2(ex["agregado"], ex["grupos"]), hide_index=True)

        if len({c for _, c, _ in ag.index}) > 1:
            st.subheader("Compromisso entre acurácia e paridade")
            st.pyplot(tabelas.figura_tradeoff(ex["agregado"]))
            st.caption("Cada linha tracejada liga o modelo original a uma versão mitigada. Para baixo: menos disparidade; "
                       "para a esquerda: menos acurácia. As barras mostram o IC 95%.")
        if ex.get("pareado"):
            with st.expander("Diferenças pareadas entre cenários (P2 a P4)"):
                st.dataframe(tabelas.tabela3(ex["pareado"]), hide_index=True)
        with st.expander("O que cada métrica mede"):
            for n_, d_ in metricas.DESCRICOES.values():
                st.markdown(f"- **{n_}**: {d_}")
        st.caption(AVISO)

# ====================================================================== Simulador (UC11)
with aba_sim:
    prev = None if ex is None else ex.get("previsoes")
    if prev is None or prev["score"].isna().all():
        st.info("O simulador precisa de uma execução com probabilidades. Abra ou execute uma auditoria.")
    else:
        com_score = prev[prev["score"].notna()]
        c1, c2, c3 = st.columns(3)
        mod = c1.selectbox("Modelo", sorted(com_score["modelo"].unique()), format_func=nome)
        cen = c2.selectbox("Cenário", sorted(com_score[com_score.modelo == mod]["cenario"].unique()),
                           format_func=lambda c: CENARIOS.get(c, c))
        sem = c3.selectbox("Repetição (semente)", sorted(com_score["semente"].unique()))
        d = com_score[(com_score.modelo == mod) & (com_score.cenario == cen) & (com_score.semente == sem)]
        grupos = sorted(d["sexo"].astype(str).unique())
        cols = st.columns(len(grupos))
        lim = {g: cols[i].slider(f"Limiar para {g}", 0.05, 0.95, 0.50, 0.01, key=f"lim_{g}") for i, g in enumerate(grupos)}
        yp = (d["score"].to_numpy() >= d["sexo"].astype(str).map(lim).to_numpy()).astype(int)
        m = metricas.calcular(d["y_true"], yp, d["sexo"].astype(str), None, ex["config"].get("grupo_referencia"))
        k1, k2, k3 = st.columns(3)
        k1.metric("Acurácia", tabelas.num(m["desempenho"]["acuracia"]))
        k2.metric("Dif. de paridade demográfica", tabelas.num(m["equidade"]["dif_paridade"]))
        k3.metric("Dif. de igualdade de oportunidades", tabelas.num(m["equidade"]["dif_ig_oport"]))
        st.bar_chart(pd.DataFrame({g: {"Taxa de seleção": v["selecao"], "TPR": v["tpr"]} for g, v in m["por_grupo"].items()}).T,
                     stack=False)
        st.caption("Simulação exploratória: não altera a auditoria registrada (RN09). Reduzir uma diferença costuma "
                   "custar acurácia ou aumentar outra diferença; a escolha do ponto de equilíbrio é humana.")

# ====================================================================== Histórico e relatório (UC09, UC10)
with aba_hist:
    hist = servico().repo.listar()
    if hist.empty:
        st.info("Nenhuma execução registrada ainda.")
    else:
        st.dataframe(hist.assign(hash_dados=hist["hash_dados"].str[:12]), hide_index=True)
        c1, c2 = st.columns([3, 1])
        escolhido = c1.selectbox("Execução", hist["id"])
        if c2.button("Abrir execução"):
            abrir(escolhido)
            st.rerun()
    if ex is not None:
        st.subheader(f"Exportar a execução {ex['id']}")
        c1, c2, c3, c4 = st.columns(4)
        c1.download_button("Relatório (HTML)", relatorio.gerar_html(ex), f"relatorio_{ex['id']}.html", "text/html")
        c2.download_button("Tabela 1 (CSV)", tabelas.tabela1(ex["agregado"])[1].to_csv(sep=";", index=False).encode("utf-8-sig"),
                           f"tabela1_{ex['id']}.csv", "text/csv")
        c3.download_button("Execução completa (JSON)", relatorio.exportar_json(ex), f"execucao_{ex['id']}.json", "application/json")
        if ex.get("previsoes") is not None:
            c4.download_button("Previsões (CSV)", ex["previsoes"].to_csv(index=False).encode(), f"previsoes_{ex['id']}.csv", "text/csv")
        st.caption("O relatório HTML pode ser salvo como PDF pelo navegador (Ctrl+P).")
        if st.button("Verificar execução"):
            problemas = servico().verificar_execucao(ex)
            if problemas:
                st.error(f"{len(problemas)} divergências entre o registro e o arquivo de previsões.")
                st.write(problemas[:20])
            else:
                st.success("As métricas recalculadas a partir do arquivo de previsões conferem com as gravadas.")
