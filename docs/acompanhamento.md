# Acompanhamento

Atualizar a cada tarefa concluída ou decisão tomada. Status: `a fazer`, `fazendo`, `feito`, `bloqueado`.

**Apresentação: sexta-feira, 2026-11-27.** Trabalho individual.

## Tarefas

| ID | Tarefa | Fase | Requisito | Prazo | Status | Observações |
|---|---|---|---|---|---|---|
| T01 | Montar estrutura do repositório e docs de planejamento | 1 | — | 23/11 | feito | |
| T02 | Confirmar a leitura do tema com o professor | 1 | todos | 23/11 | feito | P1: confirmado, qualquer tarefa |
| T03 | Escolher o dataset (domínio principal + domínio para generalização) | 1 | R5, R7 | 23/11 | feito | B2W-Reviews01 + IMDB-pt (D4) |
| T04 | Confirmar stack e modelo (D1, D2) | 1 | R2–R4 | 23/11 | feito | |
| T05 | Escrever `docs/fundamentos.md` | 2 | R1 | 23/11 | feito | Rascunho; números ⚠ dependem da T06 |
| T06 | Levantar e conferir as referências | 2 | R1, R8 | 23/11 | fazendo | Resumos de [1], [2], [4], [5] e card de [3] lidos; falta o corpo dos artigos |
| T07 | Escrever o código com os blocos marcados | 3 | R2–R4, R6 | 24/11 | feito | `code/ajuste_fino.py` (E1–E6) e `code/notebook_colab.ipynb` |
| T08 | Teste rápido do código em CPU | 3 | R2–R4 | 24/11 | feito | E1–E6 com `--rapido` rodaram sem erro (torch 2.15, transformers 5.24, datasets 5.2) |
| T09 | Rodar E1–E5 no Colab e salvar em `results/` | 4 | R5–R7 | 24/11 | a fazer | |
| T10 | Gerar figuras (curvas de perda, matriz de confusão) | 5 | R6, R7 | 25/11 | a fazer | |
| T11 | Escrever `docs/analise-codigo.md` | 5 | R2–R4 | 25/11 | a fazer | |
| T12 | Escrever `docs/sobreajuste-generalizacao.md` | 5 | R6, R7 | 25/11 | a fazer | |
| T13 | Escrever `docs/aplicacoes-vantagens-limitacoes.md` | 5 | R8, R9 | 25/11 | a fazer | |
| T14 | Montar os slides em Beamer | 6 | todos | 26/11 | a fazer | |
| T15 | Ensaio cronometrado e ajuste do roteiro | 7 | R10 | 26/11 | a fazer | |

## Decisões tomadas

| Data | Decisão | Motivo |
|---|---|---|
| 2026-11-23 | Repositório recomeçado do zero, com histórico apagado (local e GitHub) | O trabalho anterior (outro tema) não tem relação com este |
| 2026-11-23 | Mantidas as regras gerais: pt-BR, sem demo ao vivo, resultados pré-gerados em `results/`, Colab T4, slides em Beamer, números com fonte | Combinado |
| 2026-11-23 | Tema: ajuste fino supervisionado de modelo pré-treinado, em qualquer tarefa | Confirmado pelo professor (P1) |
| 2026-11-23 | O código do exemplo é escrito por mim | Permitido pelo professor (P5) |
| 2026-11-23 | Datasets carregados pela revisão `refs/convert/parquet` do Hub | Os dois usam scripts de carregamento, que o `datasets` atual não executa |
| 2026-11-23 | Baseline do E1 com `scikit-learn` (TF-IDF + regressão logística) | O `evaluate` já depende do scikit-learn; não é biblioteca nova na prática |
| 2026-11-23 | Aprovados: stack `transformers` + `datasets` + `evaluate`, modelo BERTimbau base, datasets B2W-Reviews01 e IMDB-pt (D1, D2, D4) | Aprovado pelo usuário |

## Pendências e perguntas

| ID | Pergunta | Para quem | Resposta | Status |
|---|---|---|---|---|
| P1 | O "método" é mesmo o ajuste fino supervisionado de um modelo pré-treinado? Pode ser qualquer tarefa? | Professor | Sim e sim | fechada |
| P2 | Qual a data da apresentação? | Professor | Na próxima sexta-feira (2026-11-27) | fechada |
| P3 | O trabalho é individual ou em grupo? | Professor | Individual | fechada |
| P4 | Qual dataset usar (principal e outro domínio)? | Professor | Qualquer um | fechada; escolha em T03 |
| P5 | O "código pronto" pode ser escrito por nós ou precisa ser de uma fonte externa? | Professor | Escrito por nós | fechada |
