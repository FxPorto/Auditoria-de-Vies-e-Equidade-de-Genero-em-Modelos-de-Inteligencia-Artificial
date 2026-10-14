import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motor.config import Config  # noqa: E402
from motor.sintetico import gerar_adult_sintetico  # noqa: E402


@pytest.fixture
def cfg_rapido():
    """Configuração pequena para os testes rodarem em segundos."""
    return Config(sementes=[0, 1], rf_n_estimators=30, n_bootstrap=300, n_jobs=1,
                  grades={"regressao_logistica": {"C": [0.1, 1.0]},
                          "random_forest": {"max_depth": [5, 10], "min_samples_leaf": [5]}})


@pytest.fixture
def df_vies():
    return gerar_adult_sintetico(4000, seed=1, vies=2.0)


@pytest.fixture
def df_sem_vies():
    return gerar_adult_sintetico(4000, seed=2, vies=0.0)
