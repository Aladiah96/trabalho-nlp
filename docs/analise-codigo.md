# Análise do código

Material dos blocos 3, 4 e 5 do roteiro (9 min, requisitos R2, R3 e R4). O código é `code/ajuste_fino.py`. As linhas citadas valem para o commit `da1dcb6`. Os números vêm de `results/`.

Visão geral do fluxo:

```
B2W (132.373 avaliações)
  │  [PREPARAÇÃO DOS DADOS]  limpar → rotular → dividir → tokenizar
  ▼
treino / validação / teste  (ids de tokens + rótulo)
  │  [TREINAMENTO]  BERTimbau + cabeça nova → Trainer → métricas
  ▼
modelo ajustado (salvo em disco)
  │  [INFERÊNCIA]  carregar → pipeline → rótulo + probabilidade
  ▼
previsões em textos novos e em outro domínio
```

---

## 1. Preparação dos dados (R2, 3 min)

### 1.1 Carga e rótulo (`carregar_b2w`, linhas 95–150)

```python
bruto = load_dataset(DATASET, revision=REVISAO_PARQUET, split="train")

ds = bruto.filter(
    lambda ex: ex["review_text"] is not None
    and ex["review_text"].strip() != ""
    and ex["overall_rating"] != 3
)
ds = ds.map(
    lambda ex: {"texto": ex["review_text"].strip(), "label": int(ex["overall_rating"] >= 4)},
    remove_columns=ds.column_names,
)
tabela = ds.to_pandas().drop_duplicates(subset="texto")
```

| Decisão | Por quê |
|---|---|
| Nota 1–2 → negativo, 4–5 → positivo | A B2W não tem rótulo de sentimento, só nota de 1 a 5. Transformar a nota em rótulo é a primeira decisão de modelagem. |
| Descartar a nota 3 | Avaliações medianas são ambíguas e deixariam o rótulo ruidoso. Em troca, o modelo nunca vê textos neutros, o que aparece no E5 (ver 3.3). |
| Remover textos vazios e duplicados | Uma duplicata que cai no treino e no teste ao mesmo tempo infla a métrica: o modelo "acerta" um texto que já viu. |

Resultado (`results/e2_ajuste_completo/dados.json`): das 132.373 avaliações sobram **110.882** depois da limpeza.

### 1.2 Divisão (linhas 121–124)

```python
div = ds.train_test_split(test_size=0.2, stratify_by_column="label", seed=SEMENTE)
resto = div["test"].train_test_split(test_size=0.5, stratify_by_column="label", seed=SEMENTE)
treino, validacao, teste = div["train"], resto["train"], resto["test"]
```

- **Três conjuntos com papéis diferentes:** o treino ajusta os pesos; a validação escolhe a época e decide o early stopping; o teste só é usado no fim, uma vez. Se a validação também servisse de teste, a métrica final estaria otimista, porque o modelo teria sido escolhido justamente por ir bem nela.
- **Estratificação:** a proporção de classes é a mesma nos três. No teste: 3.260 negativos e 7.829 positivos (70,6% positivos).
- **Semente fixa:** todos os experimentos usam o mesmo teste de 11.089 avaliações, então os números são comparáveis entre si.
- O tamanho do treino muda por experimento (500, 20 mil...). A validação usada durante o treino é reduzida para 2.000 exemplos para economizar tempo.

### 1.3 Tokenização (linhas 153–163)

```python
return dataset.map(
    lambda lote: tokenizador(lote["texto"], truncation=True, max_length=max_tokens),
    batched=True,
)
```

Exemplo real (`results/e2_ajuste_completo/tokenizacao.json`):

> "Produto confortável, ajusta bem ao bebê e não machuca e raramente dá alergia."
>
> `[CLS] Produ ##to confor ##tável , ajus ##ta bem ao bebê e não machu ##ca e raramente dá aler ##gia . [SEP]`

- **Subpalavras (WordPiece):** palavras menos comuns são quebradas em pedaços (`machu` + `##ca`). Assim o vocabulário fica fixo (29.794 tokens) e nenhuma palavra é desconhecida.
- **`[CLS]` e `[SEP]`:** tokens especiais. O vetor do `[CLS]` é o que a cabeça de classificação usa (ver `fundamentos.md`, seção 3).
- **Truncamento em 128 tokens:** mediana de 26 tokens por avaliação, percentil 95 de 95 tokens. Só **2,4%** dos textos são cortados. Para o IMDB, com textos bem mais longos, o E5 usa até 512.
- **Padding dinâmico:** não fazemos padding aqui. O `DataCollatorWithPadding` completa cada lote até o maior texto *daquele lote*. Com textos curtos, isso gasta bem menos memória e tempo do que completar tudo até 128.

**Slide sugerido:** o exemplo tokenizado acima, com setas para `[CLS]`, uma subpalavra e o rótulo, mais a frase "3 conjuntos, 3 papéis".

---

## 2. Treinamento (R3, 4 min)

### 2.1 Modelo (linhas 253–262)

```python
modelo = AutoModelForSequenceClassification.from_pretrained(
    MODELO,
    num_labels=len(ROTULOS),
    id2label=dict(enumerate(ROTULOS)),
    label2id={r: i for i, r in enumerate(ROTULOS)},
)
```

Uma linha carrega o encoder pré-treinado e cria a cabeça nova. Ao carregar, o `transformers` avisa que `classifier.weight` e `classifier.bias` não existem no checkpoint e foram criados com valores aleatórios. É o esperado: essa é a parte que ainda não sabe nada sobre a tarefa.

Contagem (`results/e2_ajuste_completo/parametros.json`):

| | Parâmetros |
|---|---|
| Total | 108.924.674 |
| Treináveis | 108.924.674 (todos) |
| Camada nova (classificador) | 1.538 |

### 2.2 Hiperparâmetros (linhas 273–296 e `EXPERIMENTOS`, linha 54)

| Argumento | E2 | Papel |
|---|---|---|
| `learning_rate` | 2e-5 | Pequena, para ajustar sem destruir o que o pré-treino aprendeu. O BERT sugere 5e-5, 3e-5 ou 2e-5 [1, A.3] |
| `num_train_epochs` | 2 | Poucas passadas bastam; mais épocas aumentam o risco de sobreajuste (E3) |
| `per_device_train_batch_size` | 32 | Exemplos por atualização de pesos |
| `warmup_steps` | 10% dos passos | A taxa começa em 0 e sobe; evita passos grandes enquanto a cabeça ainda é aleatória |
| `lr_scheduler_type` | `linear` | Depois do aquecimento, a taxa cai linearmente até 0 |
| `weight_decay` | 0,01 | Regularização: puxa os pesos para perto de zero |
| `eval_strategy` | a cada 10% do treino | Acompanha a validação durante o treino |
| `fp16` | sim (na GPU) | Meia precisão: mais rápido e usa menos memória |

### 2.3 O laço (linhas 306–326)

```python
trainer = Trainer(
    model=modelo,
    args=argumentos,
    train_dataset=treino_tok,
    eval_dataset=validacao_tok,
    processing_class=tokenizador,
    data_collator=DataCollatorWithPadding(tokenizador),
    compute_metrics=calcular_metricas,
    callbacks=callbacks,
)
trainer.train()
```

O `Trainer` esconde o laço clássico. Vale mostrar no slide o que ele faz a cada passo:

1. pega um lote e completa com padding;
2. passa pelo modelo e obtém os logits;
3. calcula a entropia cruzada com os rótulos;
4. retropropaga o gradiente por **todas** as 12 camadas e pela cabeça;
5. atualiza os pesos com AdamW e ajusta a taxa conforme a agenda;
6. de tempos em tempos, avalia na validação e chama `calcular_metricas`.

### 2.4 Métricas (linhas 195–202)

Acurácia e **F1 macro**. O F1 macro é a média do F1 das duas classes. Ele é necessário porque 70,6% do teste é positivo: um modelo que chuta sempre "positivo" tem 70,6% de acurácia, mas só 41,4% de F1 macro (`results/e1_referencia/metricas.json`).

### 2.5 Resultado e custo (E2)

| | Valor | Fonte |
|---|---|---|
| F1 macro no teste | 95,7% | `e2_ajuste_completo/metricas.json` |
| Acurácia no teste | 96,4% | idem |
| Tempo de treino | 242 s (20 mil exemplos, 2 épocas) | idem |
| Pico de memória da GPU | 3,01 GB de 15 GB da T4 | idem |

**Slide sugerido:** à esquerda o trecho do `Trainer`; à direita a lista de 6 passos do laço e a tabela de custo.

---

## 3. Inferência (R4, 2 min)

### 3.1 Carregar e classificar (`generalizacao`, linhas 399–420)

```python
tokenizador = AutoTokenizer.from_pretrained(caminho_modelo)
modelo = AutoModelForSequenceClassification.from_pretrained(caminho_modelo)

classificar = pipeline("text-classification", model=modelo, tokenizer=tokenizador, device=dispositivo)
saidas = classificar([texto for texto, _ in FRASES_NOVAS])
```

- O modelo salvo no E2 é uma pasta com pesos, configuração (incluindo `id2label`) e o **mesmo tokenizador** do treino. Usar outro tokenizador na inferência daria ids diferentes e previsões sem sentido.
- O `pipeline` repete a preparação (tokeniza), roda o modelo sem calcular gradientes, aplica softmax nos logits e devolve o rótulo com a probabilidade.
- Para muitos textos com rótulo (o teste do IMDB), o código usa `trainer.predict`, que processa em lotes e já calcula as métricas.

### 3.2 Saída real (`results/e5_generalizacao/frases_novas.json`)

| Frase | Esperado | Previsto | Probabilidade |
|---|---|---|---|
| Chegou antes do prazo e funciona perfeitamente. | positivo | positivo | 99,9% |
| Não recomendo, quebrou na primeira semana. | negativo | negativo | 99,1% |
| O filme é longo, mas vale cada minuto. | positivo | positivo | 99,4% |
| O atendimento do restaurante foi péssimo. | negativo | negativo | 87,2% |
| Não é ruim. | positivo | positivo | 60,7% |
| Ótimo, mais um carregador que não carrega. | negativo | **positivo** | 99,6% |
| Esperava mais pelo preço que paguei. | negativo | negativo | 77,5% |
| A entrega foi feita na terça-feira. | neutro | positivo | 72,7% |

### 3.3 O que a tabela mostra

- O modelo acerta 6 das 7 frases com sentimento claro, incluindo filme e restaurante, domínios que não estavam no treino.
- **Probabilidade não é certeza:** a ironia sai errada com 99,6%. A confiança do modelo não indica, sozinha, se ele acertou.
- **O modelo só conhece as classes do treino:** a frase neutra precisa virar "positivo" ou "negativo", porque a nota 3 foi descartada na preparação (seção 1.1). Uma decisão da preparação dos dados aparece na inferência.

**Slide sugerido:** o trecho do `pipeline` e a tabela, com as duas últimas linhas problemáticas destacadas (elas servem de ponte para o bloco de limitações).
