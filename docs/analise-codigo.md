# Análise do código

Roteiro do bloco 4 da apresentação. O professor pediu para identificar no código três coisas: **adaptadores**, **parâmetros treináveis** e **configuração de quantização**. O código está em `code/` e marca cada uma com os comentários `# [QUANTIZAÇÃO]`, `# [ADAPTADORES]` e `# [PARÂMETROS TREINÁVEIS]`.

## Escolhas do exemplo

- **Bibliotecas:** `transformers`, `peft`, `bitsandbytes`, `trl`, `datasets`, `accelerate`. Versões fixadas em `code/requirements.txt`.
- **Modelo:** `Qwen/Qwen2.5-0.5B-Instruct` (24 camadas, dimensão 896; atenção com GQA, então `k_proj`/`v_proj` saem com 128).
- **Dataset:** `timdettmers/openassistant-guanaco`, o mesmo usado no artigo do QLoRA para treinar o Guanaco. São 800 exemplos de treino e 100 de avaliação (split `test`), convertidos para o formato prompt/completion do `trl`, então a perda é calculada só na resposta do assistente.
- **Treino:** 100 passos, batch efetivo 16 (4 × 4 de acumulação), ≈ 2 épocas, `max_length=512`, lr 2e-4 com cosseno.
- **Configurações:** `lora16` (base em fp16), `qlora_r16`, `qlora_r4` e `qlora_r64`. Usamos `lora_alpha = 2·r` em todas, o que mantém α/r = 2 e isola o efeito do posto.
- **Ambiente:** Google Colab com GPU T4. Rodar `code/notebook_colab.ipynb` antes da apresentação e copiar a pasta `results/` para o repositório.

## Arquivos do código

| Arquivo | Conteúdo |
|---|---|
| `code/lora_qlora.py` | Pipeline completo, com os blocos `# [QUANTIZAÇÃO]`, `# [ADAPTADORES]` e `# [PARÂMETROS TREINÁVEIS]`. CLI: `--config`, `--medir-memoria`, `--teste-cpu`. |
| `code/gerar_figuras.py` | Lê `results/*/` e gera `results/curva_perda.png` e `results/tabela_resultados.md`. |
| `code/notebook_colab.ipynb` | Roda tudo no Colab e baixa `results.zip`. |
| `code/requirements.txt` | Versões fixadas (o `torch` é o do Colab). |

Cada configuração grava em `results/<config>/`: `metricas.json`, `log_history.json`, `parametros.txt`, `modelo_print.txt`, `respostas.json` e `adaptador/`.

## Trechos principais (de `code/lora_qlora.py`)

```python
# [QUANTIZAÇÃO] ------------------------------------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                     # pesos do modelo base guardados em 4 bits
    bnb_4bit_quant_type="nf4",             # tipo NormalFloat 4 (NF4), proposto no QLoRA
    bnb_4bit_use_double_quant=True,        # quantiza também as constantes de escala de cada bloco
    bnb_4bit_compute_dtype=torch.float16,  # dtype das multiplicações (a T4 não tem bf16)
)
model = AutoModelForCausalLM.from_pretrained(
    MODELO_ID, quantization_config=bnb_config, device_map={"": 0}
)
# [QUANTIZAÇÃO] prepara o modelo de 4 bits para treino: congela os pesos
# base, converte as normalizações para fp32 e liga o gradient checkpointing.
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True, ...)

# [ADAPTADORES] -----------------------------------------------------------
lora_config = LoraConfig(
    r=r,                     # posto das matrizes A (r x k) e B (d x r)
    lora_alpha=2 * r,        # escala da atualização = lora_alpha / r = 2
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# [PARÂMETROS TREINÁVEIS] ------------------------------------------------
model.print_trainable_parameters()
nomes_treinaveis = [n for n, p in model.named_parameters() if p.requires_grad]
treinaveis_grad = sum(p.numel() for p in model.parameters() if p.requires_grad)
# + conferência r·(d + k) somada sobre todas as camadas LoraLayer

training_args = SFTConfig(
    ...,
    save_strategy="no",                 # não grava checkpoints; só o adaptador no fim
    optim="paged_adamw_8bit",           # [QUANTIZAÇÃO] paged optimizer (só no QLoRA)
    fp16=True,                          # a T4 não tem bf16
)
trainer = SFTTrainer(model=model, args=training_args, train_dataset=..., eval_dataset=...,
                     processing_class=tokenizer)
trainer.train()
model.save_pretrained(".../adaptador")  # só lora_A e lora_B
```

Decisões que corrigem o primeiro esqueleto:

1. `bnb_4bit_compute_dtype=torch.float16`, coerente com `fp16=True`, porque a T4 não suporta bf16.
2. As respostas "antes" são geradas no modelo base, antes de `get_peft_model`. As respostas "depois" vêm do mesmo modelo depois do treino.
3. A memória 16 bits × 4 bits é medida à parte (`--medir-memoria`), carregando o modelo duas vezes.
4. A conta r·(d + k) é feita camada por camada, porque d ≠ k em `k_proj`, `v_proj` e no MLP.
5. `save_strategy="no"` evita gravar checkpoints grandes em `results/`.
6. A métrica quantitativa é o `eval_loss` antes e depois do treino, além da perda de treino.
7. As versões das bibliotecas estão fixadas em `requirements.txt`.

### Teste local (CPU, sem quantização)

`python code/lora_qlora.py --config lora16 --teste-cpu` roda 2 passos em CPU só para validar as APIs; a saída vai para `results/lora16_teste_cpu/` e **não deve ir para os slides**. Nesse teste, a contagem de parâmetros treináveis com r=16 deu 8.798.208 de 502.830.976 (1,75%), com as três formas de contagem batendo. Esse número não depende da GPU, mas o valor dos slides deve ser o de `results/qlora_r16/parametros.txt`.

## O que mostrar em cada slide

### Configuração de quantização (R5)
- O bloco `BitsAndBytesConfig` com uma linha de explicação por argumento.
- A ligação com a teoria: `nf4` → NF4, `use_double_quant` → double quantization, `optim="paged_adamw_8bit"` → paged optimizer.
- Memória do modelo: `model.get_memory_footprint()` carregado em 16 bits × carregado em 4 bits.

### Adaptadores (R3)
- O bloco `LoraConfig`: `r`, `lora_alpha`, `lora_dropout`, `target_modules`.
- Um trecho de `print(model)` mostrando uma camada adaptada, algo como:
  ```
  (q_proj): lora.Linear4bit(
    (base_layer): Linear4bit(in_features=..., out_features=...)
    (lora_dropout): ModuleDict(...)
    (lora_A): ModuleDict((default): Linear(in_features=..., out_features=16))
    (lora_B): ModuleDict((default): Linear(in_features=16, out_features=...))
  )
  ```
  A camada base é `Linear4bit` (quantizada, congelada); `lora_A` e `lora_B` são lineares comuns (treináveis).
- O tamanho da pasta `adapter/` salva, comparado com o tamanho do modelo inteiro.

### Parâmetros treináveis (R4)
- A saída real de `print_trainable_parameters()`.
- Conferir a conta: soma de r·(d + k) em todas as camadas adaptadas = número de parâmetros treináveis (`parametros.txt` traz a quebra por tipo de camada e o campo `contagens_batem`).
- Comentar que 1,75% é bem mais que o "menos de 1%" citado para modelos grandes: num modelo pequeno, as camadas são estreitas, e r·(d + k) pesa mais em relação a d·k.
- Os nomes dos parâmetros com `requires_grad=True`: todos contêm `lora_A` ou `lora_B`.

### Exemplo (R6)
- Curva de perda (`trainer.state.log_history`) salva como imagem em `results/`.
- As três perguntas de `respostas.json`, respondidas antes e depois do fine-tuning. O Qwen-Instruct já responde bem, então a mudança tende a ser de estilo (mais próximo do Guanaco). O `eval_loss` é o argumento quantitativo.
- Tempo de treino e pico de memória (`torch.cuda.max_memory_allocated()`).

## Detalhe para comentar: fusão dos pesos

`model.merge_and_unload()` funde o adaptador no modelo base. Com o modelo base em 4 bits, a fusão em cima dos pesos quantizados perde precisão. O caminho recomendado é recarregar o modelo base em 16 bits, carregar o adaptador e fundir.

## Comparação entre configurações

As quatro configurações (`lora16`, `qlora_r16`, `qlora_r4` e `qlora_r64`) preenchem a tabela de [exemplos-e-tradeoffs.md](exemplos-e-tradeoffs.md). Full fine-tuning não foi rodado; os números dele vêm dos artigos.

Ao comparar `lora16` com `qlora_r16`, lembrar que o otimizador é diferente (`adamw_torch` × `paged_adamw_8bit`). Como os estados do otimizador só existem para os parâmetros do adaptador (poucos MB), o efeito disso na memória é pequeno. A diferença vem principalmente dos pesos base em 4 bits.
