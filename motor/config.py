"""Configuração do protocolo experimental (Documentação de Software, Seção 3.4).

Tudo que muda os números fica aqui, e é gravado com cada execução (RN05).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

MODELOS = {"regressao_logistica": "Regressão logística", "random_forest": "Random Forest"}
CENARIOS = {"original": "Original", "reponderacao": "Reponderação", "limiar_por_grupo": "Limiar por grupo"}
RESTRICOES = {"demographic_parity": "Paridade demográfica", "equalized_odds": "Equalized odds"}

GRADES_PADRAO = {
    "regressao_logistica": {"C": [0.1, 1.0, 10.0]},
    "random_forest": {"max_depth": [10, 20, None], "min_samples_leaf": [1, 5]},
}


@dataclass
class Config:
    # protocolo
    sementes: list = field(default_factory=lambda: list(range(20)))
    proporcoes: tuple = (0.6, 0.2, 0.2)               # treino, validação, teste
    modelos: list = field(default_factory=lambda: list(MODELOS))
    tecnicas: list = field(default_factory=lambda: ["reponderacao", "limiar_por_grupo"])
    restricao: str = "demographic_parity"
    # dados
    alvo: str = "income"
    sensivel: str = "sex"
    grupo_referencia: str = "Male"                    # para as diferenças com sinal
    usar_sexo_como_entrada: bool = False              # decisão da seção 2 do plano
    colunas_removidas: tuple = ("fnlwgt", "education")
    # modelos
    grades: dict = field(default_factory=lambda: {k: dict(v) for k, v in GRADES_PADRAO.items()})
    rf_n_estimators: int = 300
    n_jobs: int = -1
    # estatística
    n_bootstrap: int = 2000
    limite_acuracia: float = 0.01                     # relevância prática (seção 2 do plano)
    limite_paridade: float = 0.05

    def validar(self) -> None:
        erros = []
        if not self.sementes:
            erros.append("Informe ao menos uma semente.")
        if abs(sum(self.proporcoes) - 1) > 1e-9 or len(self.proporcoes) != 3:
            erros.append("As proporções de treino, validação e teste devem somar 1.")
        desconhecidos = [m for m in self.modelos if m not in MODELOS]
        if not self.modelos or desconhecidos:
            erros.append("Selecione ao menos um modelo válido.")
        if any(t not in ("reponderacao", "limiar_por_grupo") for t in self.tecnicas):
            erros.append("Técnica de mitigação desconhecida.")
        if self.restricao not in RESTRICOES:
            erros.append("Restrição deve ser demographic_parity ou equalized_odds.")
        if erros:
            raise ValueError(" ".join(erros))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["proporcoes"] = list(self.proporcoes)
        d["colunas_removidas"] = list(self.colunas_removidas)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        d = dict(d)
        d["proporcoes"] = tuple(d.get("proporcoes", (0.6, 0.2, 0.2)))
        d["colunas_removidas"] = tuple(d.get("colunas_removidas", ()))
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def cenarios(self) -> list:
        return ["original"] + [t for t in ("reponderacao", "limiar_por_grupo") if t in self.tecnicas]
