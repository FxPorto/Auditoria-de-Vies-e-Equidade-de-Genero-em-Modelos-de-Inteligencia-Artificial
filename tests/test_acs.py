"""ACSIncome: filtro e definição iguais ao folktables (DING et al., 2021) e nomes equivalentes aos da Adult."""
import pandas as pd

from motor.acs import COLUNAS, preparar_acs


def test_filtro_alvo_e_colunas_do_acsincome():
    bruto = pd.DataFrame({
        "AGEP": [16, 30, 45, 50, 60], "COW": [1, 2, 3, 1, None], "SCHL": [16, 21, 22, 18, 19],
        "MAR": [5, 1, 1, 3, 2], "OCCP": [10, 20, 30, 40, 50], "POBP": [6, 6, 210, 6, 6],
        "RELP": [2, 0, 1, 0, 0], "WKHP": [20, 40, 50, 0, 35], "SEX": [1, 1, 2, 2, 2],
        "RAC1P": [1, 2, 6, 1, 9], "PINCP": [5000, 80000, 50000, 70000, 50001], "PWGTP": [10, 10, 10, 10, 10],
    })
    df = preparar_acs(bruto)
    # idade 16 e horas 0 saem; restam 3 pessoas
    assert len(df) == 3
    assert list(df.columns) == list(COLUNAS.values()) + ["income"]
    assert df["income"].tolist() == [1, 0, 1]          # > 50 mil (50.000 exato não conta)
    assert df["sex"].tolist() == ["Male", "Female", "Female"]
    assert df["race"].tolist() == ["Black", "Asian", "Two-or-more"]
    assert df["relationship"].iloc[0] == "rel_0" and df["workclass"].iloc[2] == "wor_-1"


def test_relshipp_de_2019_em_diante_vira_relationship():
    bruto = pd.DataFrame({"AGEP": [30], "COW": [1], "SCHL": [21], "MAR": [1], "OCCP": [10], "POBP": [6],
                          "RELSHIPP": [20], "WKHP": [40], "SEX": [2], "RAC1P": [1], "PINCP": [60000], "PWGTP": [5]})
    assert preparar_acs(bruto)["relationship"].tolist() == ["rel_20"]
