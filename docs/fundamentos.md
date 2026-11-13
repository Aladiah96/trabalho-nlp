# Fundamentos: LoRA e QLoRA

Base teórica dos blocos 1 a 3 da apresentação. Fontes em [referencias.md](referencias.md).

## 1. O problema: fine-tuning completo é caro

No fine-tuning completo (full fine-tuning), todos os pesos do modelo são atualizados. Com Adam e precisão mista, a memória por parâmetro fica em torno de **16 bytes** (conta do artigo do ZeRO, Rajbhandari et al., 2020):

| Item | Bytes por parâmetro |
|---|---|
| Pesos em fp16/bf16 | 2 |
| Gradientes em fp16/bf16 | 2 |
| Cópia mestre dos pesos em fp32 | 4 |
| Estados do Adam (momento 1 e 2, fp32) | 8 |
| **Total** | **~16** |

Para um modelo de 7B parâmetros: 7 × 10⁹ × 16 B ≈ **112 GB**, sem contar as ativações. Isso exige várias GPUs de datacenter. Além disso, cada tarefa gera uma cópia inteira do modelo (14 GB em fp16 para 7B).

PEFT (*parameter-efficient fine-tuning*) é a família de técnicas que treina só uma pequena parte dos parâmetros. LoRA é a mais usada.

## 2. LoRA (Low-Rank Adaptation)

Hu et al., 2021.

### Ideia

A hipótese do artigo é que a mudança nos pesos durante a adaptação tem **posto intrínseco baixo**. Em vez de aprender uma matriz ΔW inteira, o LoRA aprende duas matrizes pequenas cujo produto aproxima essa mudança.

Para uma camada linear com peso W₀ ∈ ℝ^(d×k):

```
h = W₀·x + ΔW·x = W₀·x + (α/r)·B·A·x

A ∈ ℝ^(r×k)    B ∈ ℝ^(d×r)    r ≪ min(d, k)
```

- **W₀ fica congelado.** Só A e B recebem gradiente.
- **Inicialização:** A aleatória, **B = 0**. No início do treino ΔW = 0, então o modelo começa exatamente igual ao original.
- **Escala α/r:** `lora_alpha` controla a magnitude da atualização. Mudar `r` sem mexer em `α` muda a escala efetiva.
- **Onde aplicar (`target_modules`):** o artigo original aplicou nas projeções de atenção (W_q, W_v). Hoje é comum aplicar em todas as camadas lineares (q, k, v, o, gate, up, down).

### Conta de parâmetros

Parâmetros treináveis por camada adaptada: **r·(d + k)**, em vez de d·k.

Exemplo com d = k = 4096 e r = 8:

- Matriz completa: 4096 × 4096 = 16.777.216
- LoRA: 8 × (4096 + 4096) = 65.536 → **0,39%** da matriz

### Inferência

Depois do treino dá para **fundir** o adaptador: W = W₀ + (α/r)·B·A. O modelo resultante tem o mesmo tamanho e a mesma latência do original. A alternativa é manter o adaptador separado (arquivos de poucos MB) e trocar adaptadores sobre o mesmo modelo base.

### Resultados do artigo

- No GPT-3 175B: cerca de **10.000× menos parâmetros treináveis** e **3× menos memória de GPU** que o fine-tuning completo.
- Qualidade igual ou melhor que o fine-tuning completo nos benchmarks testados (RoBERTa, DeBERTa, GPT-2, GPT-3).
- Sem latência extra na inferência (com fusão dos pesos).

## 3. QLoRA (Quantized LoRA)

Dettmers et al., 2023.

### Ideia

Mesmo com LoRA, os pesos congelados W₀ ainda ocupam memória em 16 bits (7B → 14 GB). O QLoRA **quantiza o modelo base para 4 bits** e treina adaptadores LoRA em 16 bits por cima dele. O gradiente passa pelos pesos de 4 bits (dequantizados na hora do cálculo) até chegar nos adaptadores.

### Os componentes

1. **NF4 (4-bit NormalFloat):** tipo de dado de 4 bits pensado para pesos com distribuição aproximadamente normal, que é o caso de redes neurais treinadas. Os 16 níveis são escolhidos pelos quantis da distribuição normal, então cada nível é usado com a mesma frequência. A quantização é feita em blocos (64 pesos), cada um com sua própria constante de escala.
2. **Double quantization:** as constantes de escala de cada bloco também são quantizadas (para 8 bits). Isso reduz o custo dessas constantes de ~0,5 para ~0,127 bit por parâmetro, cerca de 3 GB a menos num modelo de 65B.
3. **Paged optimizers:** usam a memória unificada da NVIDIA para mover estados do otimizador para a RAM da CPU quando a GPU enche (picos de memória, por exemplo com sequências longas). Evitam erro de falta de memória (OOM) sem travar o treino.
4. **Compute dtype:** os pesos ficam guardados em NF4, mas as multiplicações são feitas em bf16 (ou fp16). Cada bloco é dequantizado na hora do cálculo.

### Resultados do artigo

- Fine-tuning de um modelo de **65B numa única GPU de 48 GB**.
- Qualidade equivalente ao fine-tuning completo em 16 bits, desde que o LoRA seja aplicado em **todas as camadas lineares**.
- Família de modelos Guanaco: 99,3% do nível do ChatGPT no benchmark Vicuna, com 24 horas de treino numa GPU.

## 4. Comparação

Valores aproximados para um modelo de 7B, sem ativações. Os números do nosso exemplo vão para [exemplos-e-tradeoffs.md](exemplos-e-tradeoffs.md).

| | Full fine-tuning | LoRA | QLoRA |
|---|---|---|---|
| Pesos do modelo base | 16 bits, treináveis | 16 bits, congelados | 4 bits (NF4), congelados |
| Parâmetros treináveis | 100% | ~0,1–1% | ~0,1–1% |
| Memória dos pesos (7B) | ~112 GB (pesos + gradientes + Adam) | ~14 GB (fp16) + adaptador | ~3,5 GB (4 bits) + constantes + adaptador |
| Velocidade de treino | referência | mais rápido por passo (sem gradiente dos pesos base) | mais lento que LoRA (dequantização a cada passo) |
| Qualidade | referência | próxima ou igual | próxima ou igual (artigo); pode cair um pouco pela quantização |
| Artefato salvo por tarefa | modelo inteiro | adaptador (MB) | adaptador (MB) |

> A linha de memória conta só pesos e estados do otimizador (7 × 10⁹ × 2 B e 7 × 10⁹ × 0,5 B). As ativações dependem do tamanho da sequência, do batch e do gradient checkpointing, e não entram aí. Para números medidos, usar a tabela do nosso exemplo (0,5B) em [exemplos-e-tradeoffs.md](exemplos-e-tradeoffs.md).
