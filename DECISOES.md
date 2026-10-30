# Decisões do protocolo

Registrar aqui, ANTES da primeira execução na base real, cada decisão que muda os números.
Depois de rodar o experimento principal, nada aqui muda sem uma nova execução completa.

| # | Decisão | Valor adotado no código | Situação | Data | Quem decidiu |
| --- | --- | --- | --- | --- | --- |
| D1 | Sexo como variável de entrada | Não (`usar_sexo_como_entrada=False`); com sexo é variação de robustez | Confirmada | 13/10/2026 | Fernando e Thiago |
| D2 | Proxies do sexo (relationship, marital-status) | Mantidos; o efeito é discutido no artigo | Confirmada | 13/10/2026 | Fernando e Thiago |
| D3 | Colunas removidas | fnlwgt (peso amostral) e education (duplica education-num) | Confirmada | 13/10/2026 | Fernando e Thiago |
| D4 | Divisão | 60/20/20 estratificada pelo alvo | Definida na documentação |  |  |
| D5 | Repetições | 20 sementes (0 a 19) | Definida na documentação |  |  |
| D6 | Grades de hiperparâmetros | LR: C ∈ {0,1; 1; 10}. RF: 300 árvores, max_depth ∈ {10; 20; sem limite}, min_samples_leaf ∈ {1; 5} | Definida na documentação |  |  |
| D7 | Critério de escolha | Maior AUC na validação | Definida na documentação |  |  |
| D8 | Restrição principal do limiar | Paridade demográfica; equalized odds como robustez | Definida na documentação |  |  |
| D9 | Limites de relevância prática | Acurácia: 0,01 (1 ponto percentual); paridade: 0,05 | Confirmada | 13/10/2026 | Fernando e Thiago |
| D10 | Grupo de referência das diferenças com sinal | Male (diferença = homens menos mulheres) | Confirmada | 13/10/2026 | Fernando e Thiago |
| D11 | IC | Bootstrap percentil (2.000 reamostragens) e t corrigido de Nadeau e Bengio (2003) | Confirmada | 13/10/2026 | Fernando e Thiago |
| D12 | Base | OpenML 1590 (Adult, 48.842 registros), SHA-256 b5802f938fa90b9e203d9cc165f5f6eb91fcd32fd3edb7ccf29ba6ee550fa9d7 | Definida | 13/10/2026 | Fernando |
| D13 | Diferença de TPR com sinal na Tabela 1 e na Tabela 3 | Incluída após a execução 2cdb13cafd para mostrar a inversão escondida pela métrica sem sinal; só acrescenta relato, modelos e sementes iguais (execução 5162e18bb3) | Definida | 18/10/2026 | Grupo |
| D14 | Testes complementares | robustez_eo, robustez_com_sexo, estabilidade_10, limiar_tpr, sem_proxies e analises_extras (sexo × raça, calibração, fronteira) são análises complementares e não substituem P1 a P4 | Definida | 18/10/2026 | Grupo |
| D15 | Replicação em base recente | ACSIncome, ACS PUMS 2018 (1 ano), Califórnia (195.665 registros, SHA-256 ef4850b1d74f42bb069eb292dc5b1abaea5193ea8e186a1e27417ce35b4577ef); mesmo protocolo do principal (20 sementes) | Definida | 27/10/2026 | Grazielle |

Os valores ficam em `motor/config.py`. Se o grupo mudar alguma decisão, altere lá e registre aqui.
