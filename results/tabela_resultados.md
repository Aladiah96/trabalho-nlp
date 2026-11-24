# Tabela de resultados

Gerada por `code/gerar_figuras.py` a partir dos JSON em `results/`. Colab, GPU Tesla T4.
Teste da B2W: 11.089 avaliações. Teste do IMDB-pt: 5.000 críticas.

## Comparação no teste da B2W

| Experimento | Exemplos de treino | Épocas | Acurácia | F1 macro | Perda no teste | Tempo de treino | Pico de memória GPU |
|---|---|---|---|---|---|---|---|
| E1 classe majoritária | 20.000 | — | 70,6% | 41,4% | — | — | — |
| E1 TF-IDF + regressão logística | 20.000 | — | 93,3% | 91,9% | — | — | — |
| E2 ajuste fino completo | 20.000 | 2 | 96,4% | 95,7% | 0,123 | 242 s | 3,01 GB |
| E3 sem regularização | 500 | 20 | 94,7% | 93,6% | 0,361 | 150 s | 2,24 GB |
| E4 weight decay + early stopping | 500 | 5 | 94,5% | 93,4% | 0,166 | 85 s | 2,24 GB |

## Generalização (E5, modelo do E2)

| Conjunto | Acurácia | F1 macro | Perda |
|---|---|---|---|
| B2W teste (mesmo domínio) | 96,4% | 95,7% | 0,123 |
| IMDB-pt teste (outro domínio, até 512 tokens) | 82,2% | 81,9% | 0,442 |

## Curva de aprendizado (E6, 3 épocas)

| Exemplos de treino | Acurácia | F1 macro |
|---|---|---|
| 100 | 70,6% | 41,5% |
| 300 | 89,4% | 86,9% |
| 1.000 | 94,7% | 93,6% |
| 3.000 | 95,4% | 94,5% |
| 10.000 | 96,2% | 95,4% |
