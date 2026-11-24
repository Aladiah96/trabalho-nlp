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
| T09 | Rodar E1–E5 no Colab e salvar em `results/` | 4 | R5–R7 | 24/11 | feito | E1–E6 rodados em 23/11 numa T4; ver `results/tabela_resultados.md` |
| T10 | Gerar figuras (curvas de perda, matriz de confusão) | 5 | R6, R7 | 25/11 | feito | `code/gerar_figuras.py` → `results/figuras/` (PDF e PNG) |
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

## Primeiros achados (para T11–T13)

Todos os números vêm de `results/tabela_resultados.md`. Teste da B2W: 11.089 avaliações.

1. **O ajuste fino ganha, mas o baseline clássico é forte.** F1 macro: classe majoritária 41,4%, TF-IDF + regressão logística 91,9%, BERTimbau 95,7%. Com os mesmos 20 mil exemplos, o erro cai de 6,7% para 3,6% de acurácia (quase a metade). Chutar "positivo" acerta 70,6%, por isso o F1 macro é a métrica certa.
2. **Com 1.000 exemplos o BERTimbau já supera o TF-IDF treinado com 20 mil** (93,6% × 91,9% de F1, E6). Os ganhos diminuem depois disso: 3.000 → 94,5%, 10.000 → 95,4%, 20.000 → 95,7%.
3. **Com 100 exemplos o modelo não aprende nada útil:** prevê sempre "positivo" (F1 41,5%, igual ao chute). Com batch 32 e 3 épocas são só 12 atualizações dos pesos, poucas para sair do chute inicial. Então parte do problema é o número de passos de treino, não só a quantidade de dados. Explicar assim.
4. **O sobreajuste aparece na perda, não tanto no F1 (E3).** A perda de validação é mínima na época 2 (0,162) e sobe até ~0,33 na época 6, enquanto a do treino vai a 0,0001 e a acurácia no treino chega a 100%. O F1 de validação, porém, fica entre 92,5% e 95%. O modelo continua acertando, mas fica confiante demais quando erra, e é isso que pesa na entropia cruzada.
5. **O early stopping (E4) entrega o mesmo F1 com 1/4 das épocas.** Ele para na época 5 e volta para o checkpoint da época 2: F1 93,4% × 93,6% do E3, perda no teste 0,166 × 0,361 e 85 s × 150 s de treino. O weight decay de 0,01 quase não muda nada (as curvas do E3 e do E4 são praticamente iguais até a época 5). Dizer isso em vez de atribuir o efeito a ele.
6. **Generalização para outro domínio: F1 cai de 95,7% para 81,9% no IMDB-pt.** O erro é assimétrico: 30% das críticas negativas viram "positivo", contra 6% no sentido contrário. Possíveis causas: o treino tem 70% de positivos, críticas de filme são longas e misturam elogio e crítica, e o texto é traduzido.
7. **Frases novas (E5):** 6 de 7 frases com sentimento claro acertadas. A ironia ("Ótimo, mais um carregador que não carrega.") sai positiva com 99,6% de confiança. A frase neutra é forçada para "positivo" (72,7%), porque o modelo só conhece duas classes. "Não é ruim." sai positiva com confiança baixa (60,7%).
8. **Custo:** o E2 treinou em 242 s numa T4, com pico de 3,01 GB de memória na GPU (estimativa do planejamento: ~1,7 GB + ativações).

## Pendências e perguntas

| ID | Pergunta | Para quem | Resposta | Status |
|---|---|---|---|---|
| P1 | O "método" é mesmo o ajuste fino supervisionado de um modelo pré-treinado? Pode ser qualquer tarefa? | Professor | Sim e sim | fechada |
| P2 | Qual a data da apresentação? | Professor | Na próxima sexta-feira (2026-11-27) | fechada |
| P3 | O trabalho é individual ou em grupo? | Professor | Individual | fechada |
| P4 | Qual dataset usar (principal e outro domínio)? | Professor | Qualquer um | fechada; escolha em T03 |
| P5 | O "código pronto" pode ser escrito por nós ou precisa ser de uma fonte externa? | Professor | Escrito por nós | fechada |
