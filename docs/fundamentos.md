# Fundamentos: ajuste fino de modelos pré-treinados

Material do bloco 2 do roteiro (6 min, requisito R1). Os números entre colchetes remetem a `referencias.md`.

## 1. O problema

Treinar um modelo de linguagem do zero para cada tarefa exige muitos dados rotulados e muito processamento. Já é possível aprender bastante sobre a língua (vocabulário, sintaxe, parte do significado) com texto **sem rótulo**, que existe em grande quantidade. O ajuste fino aproveita isso: primeiro um modelo aprende a língua em geral, depois é adaptado a uma tarefa com poucos dados rotulados.

Esse esquema é um caso de **transferência de aprendizado** (*transfer learning*). Em visão computacional ele já era comum (pré-treinar na ImageNet e ajustar). Em NLP, ele se firmou com o ULMFiT [2] e o BERT [1].

## 2. As duas fases

| | Pré-treino | Ajuste fino |
|---|---|---|
| Dados | Muito texto sem rótulo | Poucos exemplos rotulados da tarefa |
| Objetivo | Tarefa auto-supervisionada (no BERT: prever palavras mascaradas e se uma frase segue a outra [1]) | A tarefa de interesse (ex.: sentimento positivo/negativo) |
| Quem faz | Quem publica o modelo (ex.: NeuralMind, no BERTimbau [3]) | Nós |
| Custo | Alto (dias de TPU/GPU) | Baixo (minutos a horas numa GPU) |
| Resultado | Pesos genéricos do encoder | Encoder ajustado + cabeça de tarefa |

Resumo do BERT [1]: o modelo pré-treinado "pode ser ajustado com apenas uma camada de saída adicional" para várias tarefas, "sem modificações substanciais na arquitetura".

## 3. O que muda no modelo

Para classificação de texto com um modelo do tipo BERT:

```
texto ──► tokenizador ──► [CLS] tok1 tok2 ... [SEP]
                                   │
                         encoder Transformer (12 camadas, pré-treinado)
                                   │
                         vetor do [CLS] (768 dimensões)
                                   │
                         cabeça de classificação (nova, pesos aleatórios)
                                   │
                         logits (1 por classe) ──► softmax ──► probabilidades
```

- O **encoder** vem do pré-treino.
- A **cabeça** é nova. Na classe `BertForSequenceClassification` do `transformers` ela é formada pelo *pooler* (camada densa + tanh sobre o `[CLS]`, que já vem do pré-treino), um dropout e uma camada linear de 768 → número de classes [6]. Só essa camada linear começa com pesos aleatórios. Com 2 classes ela tem 768 × 2 + 2 = 1.538 parâmetros, contra os 108,9 M do modelo inteiro (ver `planejamento.md`).

## 4. Como o treino funciona

Para um exemplo com rótulo verdadeiro *y* e probabilidades previstas *p* = softmax(**W h**<sub>[CLS]</sub> + **b**), a perda é a **entropia cruzada**:

  L = − log *p*<sub>y</sub>

O gradiente dessa perda passa pela cabeça e por **todas** as camadas do encoder. No ajuste fino completo, todos os ~109 M pesos são atualizados, mas com uma taxa de aprendizado pequena, para não "apagar" o que o pré-treino ensinou.

Hiperparâmetros sugeridos no artigo do BERT [1, apêndice A.3]: batch de 16 ou 32; taxa de aprendizado (Adam) de 5e-5, 3e-5 ou 2e-5; 2, 3 ou 4 épocas; dropout fixo em 0,1. Os autores observam que conjuntos grandes (100 mil exemplos ou mais) são bem menos sensíveis à escolha de hiperparâmetros que os pequenos.

## 5. Variações

| Variação | O que é treinado | Quando usar |
|---|---|---|
| Ajuste fino completo | Encoder + cabeça | Padrão; é o que o exemplo deste trabalho usa |
| Extração de características | Só a cabeça (encoder congelado) | Pouca memória ou pouquíssimos dados; costuma render menos |
| Descongelamento gradual e taxas por camada (ULMFiT [2]) | Camadas liberadas aos poucos, de cima para baixo | Reduzir o risco de esquecer o pré-treino |
| Ajuste eficiente (adapters, LoRA etc.) | Poucos parâmetros extras | Modelos muito grandes; fora do escopo, só uma menção |

## 6. Por que funciona com poucos dados

O ULMFiT reduziu o erro em 18–24% na maioria de seis conjuntos de classificação de texto. Com só 100 exemplos rotulados, ele igualou um modelo treinado do zero com 100 vezes mais dados [2, resumo]. O pré-treino já entrega representações úteis, e o ajuste fino só precisa "apontar" essas representações para a tarefa.

## 7. Cuidados (ponte para sobreajuste e generalização)

- **Instabilidade:** com os mesmos hiperparâmetros, sementes aleatórias diferentes podem levar a resultados bem diferentes. A inicialização da cabeça e a ordem dos dados pesam de forma parecida. Em conjuntos pequenos, muitas execuções divergem no meio do treino [5, resumo].
- **Causa da instabilidade:** Mosbach et al. mostram que nem o esquecimento catastrófico nem o tamanho pequeno do conjunto explicam a instabilidade. A causa é a dificuldade de otimização (gradientes que somem). O que sobra da variação vem de diferenças de generalização: modelos com a mesma perda de treino têm desempenhos diferentes no teste [4, resumo].
- **Sobreajuste:** com poucos exemplos e muitas épocas, o modelo decora o conjunto de treino. Esse é o tema do bloco 7 e dos experimentos E3/E4 (`planejamento.md`).

## 8. O modelo do exemplo: BERTimbau

O BERTimbau é um BERT pré-treinado para português do Brasil, no corpus brWaC. Ele existe em duas versões: Base (12 camadas, 110 M parâmetros) e Large (24 camadas, 335 M) [3, card do modelo]. Pelo cálculo feito a partir da configuração, o Base tem 108,9 M parâmetros (ver `planejamento.md`), o que bate com os ~110 M do card.

## Proposta de slides para o bloco (6 min)

1. **Do zero × reaproveitar** (1 min): o problema da seção 1.
2. **Pré-treino → ajuste fino** (1,5 min): a tabela da seção 2.
3. **O que muda no modelo** (1,5 min): o diagrama da seção 3, com o destaque de 1.538 parâmetros novos × 108,9 M ajustados.
4. **Perda e hiperparâmetros** (1 min): seção 4.
5. **Variações e por que funciona** (1 min): seções 5 e 6, com o número do ULMFiT.

A seção 7 vai para o bloco 7 do roteiro.
