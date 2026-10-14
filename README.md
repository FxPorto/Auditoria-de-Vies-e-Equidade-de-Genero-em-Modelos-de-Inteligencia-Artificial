# Painel de Auditoria de Viés e Equidade

Software do TCC **Auditoria de viés e equidade de gênero em modelos de Inteligência Artificial**
(Engenharia de Software, Universidade São Judas Tadeu).

O sistema treina regressão logística e Random Forest na base Adult Census Income, mede o desempenho e as
diferenças entre grupos de sexo registrado, aplica duas técnicas de mitigação (reponderação e limiar por
grupo) e compara os cenários com média, desvio-padrão e IC 95% em 20 repetições.

> **Base sintética:** a opção "Sintética (somente teste)" existe para testar a ferramenta sem internet.
> Os números dela **nunca** entram no artigo.

## 1. Instalar (uma vez)

Requer Python 3.13 (ou Docker, seção 6).

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate      Linux/Mac:  source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Rodar os testes (antes de qualquer experimento)

```bash
python -m pytest -q
```

Os 21 testes cobrem T01 a T10 do plano de testes, além de persistência, modo Avaliador, importação e validação da base.
Todos precisam passar antes de rodar na base real.

| Teste | Arquivo | O que garante |
| --- | --- | --- |
| T01, T02 | tests/test_metricas.py | Métricas certas (exemplo à mão e conferência com scikit-learn/Fairlearn) |
| T03 | tests/test_protocolo.py | Detecta viés conhecido e não inventa viés onde não há |
| T04 | tests/test_protocolo.py | Sem vazamento: rótulos embaralhados dão AUC perto de 0,5 |
| T05 | tests/test_protocolo.py | Divisão 60/20/20 estratificada, sem sobreposição |
| T06 | tests/test_protocolo.py | O conjunto de teste não influencia hiperparâmetros nem limiares |
| T07, T08 | tests/test_protocolo.py | Reponderação e limiar por grupo fazem o que prometem |
| T09 | tests/test_protocolo.py | Mesma semente, mesmos números |
| T10 e outros | tests/test_persistencia.py | Hash, previsões armazenadas, modo Avaliador, importação |
| Base | tests/test_validacao_base.py | Reconhece a versão certa da Adult e detecta troca de base pelo hash |

## 3. Rodar o experimento do artigo

Na primeira vez, a base Adult é baixada do OpenML (id 1590) e salva em `dados/adult.csv` (precisa de internet).

### 3.1 Validar a base (antes do piloto)

```bash
python scripts/validar_base.py              # baixa se faltar, confere e mostra o hash
python scripts/validar_base.py --registrar  # só a primeira pessoa, uma vez: grava dados/adult.sha256
```

Use somente a versão com **48.842 registros** (OpenML 1590). A versão de 32.561 registros, comum no Kaggle, é só
o conjunto de treino e o script recusa. Depois do `--registrar`, façam commit de `dados/adult.csv` e
`dados/adult.sha256`: a partir daí o `rodar_experimento.py` se recusa a rodar com uma base de hash diferente.

### 3.2 Experimentos

```bash
# 1) teste de fumaça (1 min, base sintética)
python scripts/rodar_experimento.py --rapido --fonte sintetica --nome fumaca

# 2) piloto na base real: confira as checagens de sanidade e o tempo
python scripts/rodar_experimento.py --sementes 2 --nome piloto

# 3) experimento principal (20 sementes) - estes são os números do artigo
python scripts/rodar_experimento.py --nome principal

# 4) variações de robustez
python scripts/rodar_experimento.py --restricao equalized_odds --nome robustez_eo
python scripts/rodar_experimento.py --usar-sexo --nome robustez_com_sexo
python scripts/rodar_experimento.py --sementes 10 --nome estabilidade_10
```

Tempo de referência: cerca de 40 s por semente em uma máquina de 2 núcleos, ou seja, 15 a 30 minutos
para as 20 sementes. Anotem o tempo real e a configuração da máquina para o RNF03.

### O que sai em `resultados/<nome>/`

| Arquivo | Uso no artigo |
| --- | --- |
| `tabela1_formatada.csv` | Tabela 1: média e IC 95% por modelo e cenário |
| `tabela2_por_grupo.csv` | Tabela 2: taxa-base, seleção, TPR e FPR por sexo registrado |
| `tabela3_diferencas_pareadas.csv` | Tabela 3: respostas a P2, P3 e P4 (diferenças pareadas com IC) |
| `figura_tradeoff.png` / `.pdf` | Figura: acurácia × diferença de paridade, com IC |
| `sanidade.csv` | Checagens da seção 5.3 do plano (todas devem estar OK) |
| `por_semente.csv`, `agregado_completo.csv`, `pareado_completo.csv` | Dados completos, para conferência |
| `relatorio.html`, `execucao.json`, `previsoes.csv.gz` | Registro completo da execução |

Os CSV usam `;` e vírgula decimal, para abrir direto no Excel.

**Congelamento:** depois do experimento principal, façam commit da pasta `resultados/principal` e criem uma
tag (`git tag resultados-artigo-v1`). O artigo usa somente esses arquivos. Se algo der errado, corrijam o
código, rodem os testes e repitam a execução inteira; nunca descartem uma semente isolada.

### Como ler a Tabela 3

- **Consistente (bootstrap / t):** o IC 95% da diferença não contém zero. O IC pelo t corrigido de Nadeau e
  Bengio (2003) é mais conservador, porque considera que as repetições compartilham dados. Quando os dois
  discordam, relatem com cautela.
- **Relevante:** consistente e acima dos limites práticos definidos em `DECISOES.md` (acurácia: 1 ponto
  percentual; paridade: 0,05).

## 4. Abrir o painel

```bash
streamlit run interface/app.py
```

Abre em http://localhost:8501 com quatro áreas: **Dados**, **Resultados**, **Simulador** e
**Histórico e relatório**. Execuções feitas pelo script também aparecem no histórico.

## 5. Modo Avaliador (somente leitura)

```bash
# Linux/Mac
MODO=avaliador streamlit run interface/app.py --server.port 8502
# Windows (PowerShell)
$env:MODO="avaliador"; streamlit run interface/app.py --server.port 8502
```

Os comandos de execução ficam ocultos e o banco é aberto apenas para leitura (RF21).

## 6. Docker

```bash
docker compose up --build
```

Pesquisador em http://localhost:8501 e Avaliador em http://localhost:8502. As pastas `dados`,
`execucoes` e `resultados` ficam no seu computador (volumes).

## 7. Estrutura

```
interface/app.py                  Telas (Streamlit), modo Pesquisador ou Avaliador
servicos/auditoria_service.py     Validação, orquestração, importação e verificação
servicos/tabelas.py               Tabelas 1 a 3 e figura do artigo
servicos/relatorio.py             Relatório HTML e exportação JSON
servicos/sanidade.py              Checagens de sanidade na base real
servicos/validacao_base.py        Validação da base (versão, conteúdo e hash registrado)
motor/config.py                   Configuração do protocolo (o que muda os números)
motor/dados.py                    Carga, hash e preparação da base
motor/modelos.py                  Pré-processamento, modelos e ajuste na validação
motor/mitigacao.py                Reponderação e limiar por grupo
motor/metricas.py                 Métricas de desempenho, por grupo e de equidade
motor/estatistica.py              Média, desvio-padrão, IC 95% e diferenças pareadas
motor/protocolo.py                Divisão 60/20/20, laço de sementes e agregação
motor/sintetico.py                Base sintética para testes
persistencia/repositorio.py       SQLite (tabela execucoes) e arquivos de previsões
scripts/rodar_experimento.py      Linha de comando para o experimento do artigo
scripts/validar_base.py           Linha de comando para validar a base e registrar o hash
tests/                            Testes T01 a T10, persistência e validação da base
```
