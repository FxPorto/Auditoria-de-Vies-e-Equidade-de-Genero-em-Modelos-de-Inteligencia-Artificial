# Pasta de dados

`adult.csv` é criado automaticamente na primeira execução, a partir do OpenML (id 1590, 48.842 registros).
Depois disso, o sistema usa sempre esta cópia local, identificada pelo hash SHA-256.

1. Rode `python scripts/validar_base.py`: baixa a base (se faltar), confere e mostra o hash.
2. Se não houver FALHA, a primeira pessoa do grupo roda `python scripts/validar_base.py --registrar`, que grava
   `adult.sha256`.
3. Façam commit de `adult.csv` e `adult.sha256`. Todos usam essa cópia; ninguém baixa de novo.

Versionem o `adult.csv` no repositório (cerca de 4 MB): é a "base de dados" exigida na entrega do software.
O `.gitattributes` impede o git de trocar as quebras de linha do arquivo, o que mudaria o hash no Windows.
