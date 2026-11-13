# Requisitos

Enunciado do professor decomposto em itens que dá para conferir antes da entrega.

| # | Requisito | Critério de pronto |
|---|---|---|
| R1 | Explicar LoRA | Slides mostram a decomposição ΔW = B·A, o posto `r`, o fator `α/r`, os pesos base congelados e a conta de parâmetros treináveis. |
| R2 | Explicar QLoRA | Slides mostram NF4, double quantization, paged optimizers e o fato de o LoRA ser treinado sobre o modelo base quantizado em 4 bits. |
| R3 | Adaptadores no código | Slide com o trecho de `LoraConfig` + `get_peft_model` e a saída de `print(model)` destacando `lora_A` e `lora_B`. |
| R4 | Parâmetros treináveis no código | Slide com a saída de `print_trainable_parameters()` e a explicação de onde vem o número. |
| R5 | Configuração de quantização no código | Slide com o trecho de `BitsAndBytesConfig`, explicando cada argumento, e a memória ocupada pelo modelo antes e depois de quantizar. |
| R6 | Exemplo | Treino curto executado e documentado em `results/`: comparação de saídas do modelo antes e depois do fine-tuning e curva de perda. |
| R7 | Exemplos de aplicação | Pelo menos três casos de uso reais ou plausíveis, com um motivo para usar LoRA/QLoRA em cada. |
| R8 | Compromissos e vantagens/limitações | Tabela qualidade × memória × custo (full FT × LoRA × QLoRA) e listas de vantagens e limitações. |
| R9 | Duração ≤ 25 min | Ensaio cronometrado abaixo de 25 min, deixando margem para perguntas, se o professor quiser. |

## Pontos ainda em aberto

- Formato dos slides (PowerPoint, Google Slides, LaTeX Beamer etc.).
- Trabalho individual ou em grupo e, se for em grupo, quem apresenta cada bloco.
- Data da apresentação.
- Se as perguntas contam dentro dos 25 minutos.
