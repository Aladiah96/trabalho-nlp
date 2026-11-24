# Sobreajuste e generalização

Material dos blocos 6 e 7 do roteiro (R5, R6, R7). Os números vêm de `results/tabela_resultados.md` e dos JSON de cada experimento. As figuras estão em `results/figuras/`.

## 1. Conceitos

- **Generalização:** acertar em dados que o modelo não viu no treino. É o que importa na prática.
- **Sobreajuste (overfitting):** o modelo se ajusta tanto ao treino que passa a decorar detalhes e ruído dele, e piora (ou para de melhorar) em dados novos.
- **Como detectar:** comparar o erro no treino com o erro na validação ao longo das épocas. O sinal típico é o treino continuar melhorando enquanto a validação piora.
- **Por que o ajuste fino é sensível a isso:** o modelo tem ~109 M parâmetros e pode ser ajustado com algumas centenas de exemplos. Há capacidade de sobra para decorar.

## 2. O sobreajuste no nosso exemplo (E3 × E4)

Figura: `results/figuras/curva_sobreajuste.pdf`.

**E3:** 500 avaliações, 20 épocas, taxa 5e-5, sem weight decay e sem early stopping. Por época (`results/e3_sobreajuste/log_history.json`):

| Época | Perda no treino | Acurácia no treino | Perda na validação | F1 na validação |
|---|---|---|---|---|
| 1 | 0,351 | 83,4% | 0,376 | 66,4% |
| **2** | 0,086 | 97,4% | **0,162** (mínimo) | 94,0% |
| 4 | 0,010 | 99,8% | 0,281 | 92,5% |
| 6 | 0,0004 | 100% | 0,329 | 93,2% |
| 20 | 0,0001 | 100% | 0,287 | 95,1% |

A perda no treino é medida com o modelo em modo de avaliação, ao fim de cada época (callback `AvaliarNoTreino` no código), para comparar com a validação nas mesmas condições.

### 2.1 O que a curva mostra

1. **Na época 2 o modelo já está no seu melhor na validação.** Depois disso, o treino continua a cair até praticamente zero e a acurácia no treino chega a 100% na época 6: o modelo decorou as 500 avaliações.
2. **A perda de validação dobra** (0,162 → 0,329) entre as épocas 2 e 6.
3. **Mas o F1 de validação quase não muda** (fica entre 92,5% e 95,1%). Esse é o ponto mais interessante para a apresentação.

### 2.2 Por que a perda sobe e o F1 não

A entropia cruzada de um exemplo é −log *p*<sub>correta</sub>. Ela não olha só se o modelo acertou: também pune a **confiança errada**, e sem limite. Um erro com 99,9% de certeza custa −log(0,001) ≈ 6,9, enquanto um erro com 60% custa −log(0,4) ≈ 0,9.

Ao decorar o treino, o modelo aprende a dar probabilidades extremas. Nos exemplos de validação que ele já acertava, nada muda. Nos que ele erra, o erro passa a ser confiante, e a perda dispara. O F1, que só conta acertos, quase não percebe.

**Conclusão para o slide:** um modelo sobreajustado pode continuar acertando, mas fica **confiante demais quando erra**. A ironia da seção 3.2 de `analise-codigo.md` (errada com 99,6%) é o mesmo fenômeno visto de perto.

### 2.3 Contramedida: early stopping (E4)

O E4 é o E3 com weight decay de 0,01 e early stopping (paciência de 3 épocas, de olho na perda de validação).

| | E3 (sem) | E4 (com) |
|---|---|---|
| Épocas executadas | 20 | 5 (parou sozinho) |
| Checkpoint usado | o da época 20 | o da época 2 (o melhor na validação) |
| F1 macro no teste | 93,6% | 93,4% |
| Perda no teste | 0,361 | **0,166** |
| Tempo de treino | 150 s | **85 s** |

- O early stopping entrega praticamente o mesmo F1, com **menos da metade da perda** e **57% do tempo**, porque percebe na validação que a época 2 foi a melhor e volta para ela.
- **O weight decay quase não fez diferença:** as curvas do E3 e do E4 são praticamente iguais nas 5 épocas em comum (ex.: perda de validação na época 2 de 0,1619 × 0,1621). Com só 5 épocas e um valor pequeno, ele não tem tempo de agir. Não atribuir o resultado do E4 a ele.
- **Cuidado com a comparação:** a diferença de 0,2 ponto no F1 entre E3 e E4 está dentro da variação esperada entre sementes aleatórias. Com os mesmos hiperparâmetros, sementes diferentes já mudam bastante o resultado do ajuste fino [5]. Cada experimento aqui rodou com uma semente só.

### 2.4 Outras contramedidas (para citar)

| Medida | Como age | Onde aparece |
|---|---|---|
| Mais dados | Mais difícil decorar | E6 (seção 3.2) |
| Menos épocas / taxa menor | Menos tempo para decorar | E2 usa 2 épocas e taxa 2e-5 |
| Early stopping | Escolhe a época pela validação | E4 |
| Weight decay | Penaliza pesos grandes | E2, E4 (efeito pequeno aqui) |
| Dropout | Desliga neurônios ao acaso durante o treino | Já vem no BERT (0,1 [1, A.3]), não foi alterado |
| Congelar camadas | Menos parâmetros livres | `fundamentos.md`, seção 5 |

## 3. Generalização

### 3.1 Mesmo domínio: o teste da B2W

Os 11.089 exemplos do teste nunca foram usados no treino nem na escolha de hiperparâmetros. No E2: **F1 de 95,7%** e acurácia de 96,4%. O modelo generaliza bem **dentro da distribuição** em que foi treinado.

### 3.2 Quanto dado é preciso (E6)

Figura: `results/figuras/curva_aprendizado.pdf`. Mesmo teste, 3 épocas por tamanho.

| Exemplos de treino | F1 macro |
|---|---|
| 100 | 41,5% (igual ao chute "sempre positivo") |
| 300 | 86,9% |
| 1.000 | 93,6% |
| 3.000 | 94,5% |
| 10.000 | 95,4% |
| 20.000 (E2, 2 épocas) | 95,7% |
| *TF-IDF + regressão logística com 20.000* | *91,9%* |

- **Com 1.000 exemplos o BERTimbau já supera o modelo clássico treinado com 20 vezes mais dados.** É o mesmo efeito relatado pelo ULMFiT [2]: o pré-treino substitui boa parte dos dados rotulados.
- **Os ganhos diminuem rapidamente:** de 1.000 para 10.000 exemplos (10 vezes mais) o F1 sobe só 1,8 ponto.
- **Com 100 exemplos o modelo não sai do chute.** Com batch 32 e 3 épocas, são só 12 atualizações dos pesos. Então parte do problema é o número de passos, não só a quantidade de dados. Mais épocas provavelmente ajudariam (não testado).

### 3.3 Outro domínio: IMDB-pt (E5)

Figuras: `results/figuras/generalizacao.pdf` e `results/figuras/matrizes_confusao.pdf`. O mesmo modelo do E2, sem nenhum ajuste, em 5.000 críticas de filmes (metade positiva, metade negativa).

| | F1 macro | Acurácia |
|---|---|---|
| B2W (produtos, mesmo domínio) | 95,7% | 96,4% |
| IMDB-pt (filmes, outro domínio) | **81,9%** | 82,2% |

Matriz de confusão no IMDB-pt:

| Real \ Previsto | negativo | positivo |
|---|---|---|
| negativo (2.500) | 1.750 (70%) | **750 (30%)** |
| positivo (2.500) | 141 (6%) | 2.359 (94%) |

- **Cai 14 pontos de F1** só por mudar o assunto. A tarefa é a mesma (sentimento), mas a distribuição dos textos é outra. Isso é **mudança de domínio** (*domain shift*).
- **O erro é quase todo de um lado:** críticas negativas viram "positivo". Hipóteses (não testadas separadamente):
  1. **Desbalanceamento do treino:** 70% das avaliações da B2W são positivas, e o modelo herdou essa tendência.
  2. **Textos diferentes:** críticas de filme são longas (mediana de 944 caracteres, contra 95 na B2W) e costumam elogiar partes do filme antes de criticá-lo.
  3. **Tradução:** o IMDB-pt é uma versão em português de críticas escritas em inglês [9] e pode soar menos natural que um texto escrito em português.
- **Ainda assim, 82% sem nenhum exemplo de filme** mostra que o modelo aprendeu algo geral sobre sentimento, não só sobre produtos.

### 3.4 Como melhorar a generalização

- Incluir exemplos do domínio-alvo no ajuste fino, mesmo poucos (o E6 sugere que algumas centenas já ajudam muito).
- Continuar o pré-treino em texto do domínio antes do ajuste fino, como no BioBERT [10] e no LEGAL-BERT [12].
- Balancear as classes no treino, ou ajustar o limiar de decisão.
- Sempre avaliar fora do domínio antes de usar o modelo em produção: o número do teste da B2W (95,7%) seria otimista para filmes.

## 4. Proposta de slides (bloco 7, 3 min)

1. **Sobreajuste** (1,5 min): `curva_sobreajuste.pdf` + uma frase: "o treino vai a zero, a validação sobe depois da época 2; o early stopping volta para a época 2 e economiza 3/4 das épocas".
2. **Generalização** (1,5 min): `generalizacao.pdf` com a matriz do IMDB ao lado (30% × 6%) e a frase "mesma tarefa, outro domínio: −14 pontos".

A curva de aprendizado (`curva_aprendizado.pdf`) e a comparação com o TF-IDF (`comparacao_metodos.pdf`) vão no bloco 6 (exemplo e resultados).
