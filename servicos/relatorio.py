"""Relatório da execução em HTML imprimível em PDF (RF12) e exportações (RF13)."""
from __future__ import annotations

import base64
import io
import json
from html import escape

import pandas as pd

from motor.config import CENARIOS, MODELOS, RESTRICOES
from motor.metricas import DESCRICOES
from servicos import tabelas

LIMITACOES = (
    "Os resultados descrevem diferenças estatísticas entre grupos; a ferramenta não classifica um modelo "
    "como justo ou injusto, nem emite parecer jurídico. A base Adult (Censo dos EUA, 1994) registra o sexo "
    "apenas como masculino ou feminino, o que não corresponde à identidade de gênero. Critérios de equidade "
    "diferentes podem ser incompatíveis quando as taxas-base dos grupos diferem. As repetições compartilham "
    "parte dos dados, o que torna o IC por bootstrap um pouco otimista; o IC pelo t corrigido é mais conservador."
)


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def gerar_html(ex: dict) -> str:
    cfg = ex["config"]
    _, t1 = tabelas.tabela1(ex["agregado"])
    t2 = tabelas.tabela2(ex["agregado"], ex["grupos"])
    t3 = tabelas.tabela3(ex["pareado"]) if ex.get("pareado") else pd.DataFrame()
    cen = {c for c in pd.DataFrame(ex["agregado"]).cenario}
    fig = _png(tabelas.figura_tradeoff(ex["agregado"])) if len(cen) > 1 else None
    resumo_cfg = {
        "Modelos": ", ".join(MODELOS.get(m, m) for m in cfg.get("modelos", [])) or "importado",
        "Técnicas": ", ".join(CENARIOS.get(t, t) for t in cfg.get("tecnicas", [])) or "—",
        "Restrição do limiar": RESTRICOES.get(cfg.get("restricao"), "—"),
        "Repetições (sementes)": len(cfg.get("sementes", [])),
        "Proporções treino/validação/teste": "/".join(str(p) for p in cfg.get("proporcoes", [])) or "—",
        "Sexo como entrada": "sim" if cfg.get("usar_sexo_como_entrada") else "não",
        "Colunas removidas": ", ".join(cfg.get("colunas_removidas", [])) or "—",
    }
    desc = "".join(f"<li><b>{escape(n)}</b>: {escape(d)}</li>" for n, d in DESCRICOES.values())
    css = ("body{font-family:Georgia,serif;max-width:900px;margin:40px auto;padding:0 20px;line-height:1.5;color:#0b0b0b}"
           "table{border-collapse:collapse;font-size:12px;width:100%;margin:8px 0 20px}"
           "td,th{border:1px solid #bbb;padding:4px 6px;text-align:left}th{background:#ececec}"
           "pre{background:#f4f4f4;padding:10px;font-size:12px;white-space:pre-wrap}")
    return f"""<!doctype html><html lang="pt-BR"><meta charset="utf-8">
<title>Relatório de auditoria {escape(str(ex.get('id', '')))}</title><style>{css}</style>
<h1>Relatório de auditoria de viés e equidade</h1>
<p>Execução <b>{escape(str(ex.get('id', '')))}</b> · {escape(str(ex.get('criado_em', '')))} ·
fonte: {escape(str(ex.get('fonte', ex.get('dados', {}).get('origem', ''))))} ·
hash dos dados: <code>{escape(str(ex.get('dados', {}).get('hash', ''))[:16])}</code></p>
<h2>Configuração</h2>{pd.Series(resumo_cfg).to_frame('Valor').to_html()}
<h2>Desempenho e equidade (média e IC 95%)</h2>{t1.to_html(index=False)}
{f'<h2>Compromisso entre acurácia e paridade</h2><img src="data:image/png;base64,{fig}" width="100%">' if fig else ''}
<h2>Resultados por sexo registrado</h2>{t2.to_html(index=False)}
{('<h2>Diferenças pareadas entre cenários</h2>' + t3.to_html(index=False)) if len(t3) else ''}
<h2>O que cada métrica mede</h2><ul>{desc}</ul>
<h2>Limitações</h2><p>{escape(LIMITACOES)}</p>
<h2>Versões</h2><pre>{escape(json.dumps(ex.get('versoes', {}), ensure_ascii=False, indent=2))}</pre>
</html>"""


def exportar_json(ex: dict) -> str:
    corpo = {k: v for k, v in ex.items() if k != "previsoes"}
    return json.dumps(corpo, ensure_ascii=False, indent=2, default=float)
