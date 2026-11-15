# Exemplos de aplicação e compromissos

Base dos blocos 5 e 6 da apresentação.

## Exemplos de aplicação

1. **Assistente de domínio específico.** Ajustar um modelo aberto para responder sobre documentação interna, jurídica ou médica. Com QLoRA isso roda em uma GPU de consumidor, e os dados não precisam sair da empresa.
2. **Adaptação de idioma ou estilo.** Melhorar respostas em português ou fazer o modelo seguir um formato (tom formal, JSON, respostas curtas).
3. **Tarefas de NLP clássicas com LLM.** Classificação de sentimento, extração de entidades ou sumarização, treinando um adaptador por tarefa.
4. **Vários adaptadores sobre um modelo base.** Um único modelo carregado na memória e vários adaptadores de poucos MB trocados por requisição (um por cliente ou por tarefa).
5. **Modelos de imagem.** LoRA também é muito usado em modelos de difusão (Stable Diffusion) para ensinar um estilo ou personagem com poucas imagens. Serve para mostrar que a técnica não é só de texto.
6. **Pesquisa com pouco recurso.** O próprio artigo do QLoRA treinou a família Guanaco em uma GPU, em 24 horas.

## Compromissos: qualidade × memória × custo

| Escolha | Qualidade | Memória | Custo/tempo de treino |
|---|---|---|---|
| Full FT → LoRA | igual ou um pouco menor | cai muito (sem gradiente e otimizador dos pesos base) | menor |
| LoRA → QLoRA | pode cair um pouco pela quantização em 4 bits | cai ~4× nos pesos base | passo mais lento (dequantização), mas roda em GPU mais barata |
| `r` maior | mais capacidade; ganho diminui a partir de certo ponto | aumenta pouco | aumenta pouco |
| Mais `target_modules` (todas as lineares) | melhor; o artigo do QLoRA mostrou que é necessário para igualar o full FT | aumenta | aumenta |
| Gradient checkpointing | não muda | reduz ativações | mais lento (recalcula ativações no backward) |
| Paged optimizer | não muda | evita OOM em picos | lento só quando precisa paginar |

Ponto central para a fala: **QLoRA troca tempo de computação por memória.** O treino fica mais lento por passo, mas cabe numa GPU muito mais barata, e o custo total costuma cair.

## Vantagens

- Treina uma fração pequena dos parâmetros: menos de 1% num modelo de 7B (conta r·(d + k) em [fundamentos.md](fundamentos.md)). Num modelo pequeno a fração é maior: 1,75% no nosso exemplo (0,5B, r=16).
- Memória muito menor; QLoRA permite treinar modelos grandes em uma GPU.
- Adaptadores pequenos: fáceis de guardar, versionar e distribuir.
- Vários adaptadores sobre um mesmo modelo base.
- Sem latência extra na inferência quando os pesos são fundidos.
- O modelo base não é alterado, então dá para voltar ao original a qualquer momento.

## Limitações

- O posto baixo limita o quanto o modelo consegue mudar. Para tarefas muito diferentes do pré-treino (um domínio ou idioma novo), o full FT pode ser melhor.
- Hiperparâmetros a mais para ajustar: `r`, `lora_alpha`, `target_modules`, taxa de aprendizado.
- QLoRA é mais lento por passo que LoRA por causa da dequantização.
- A quantização em 4 bits pode perder qualidade, principalmente em modelos pequenos.
- Fundir o adaptador num modelo de 4 bits perde precisão; é preciso o modelo base em 16 bits para fundir direito.
- Dependência do `bitsandbytes`, que é focado em GPUs NVIDIA (o suporte a outros hardwares é mais limitado).
- Fine-tuning não ensina conhecimento novo de forma confiável. Para fatos que mudam, RAG costuma ser melhor.

## Tabela de resultados do exemplo

Preencher com os números de `results/tabela_resultados.md`, gerada por `code/gerar_figuras.py` depois de rodar `code/notebook_colab.ipynb` no Colab.

Modelo: Qwen2.5-0.5B-Instruct, 100 passos, batch efetivo 16, α/r = 2 em todas as linhas.

| Configuração | Parâmetros treináveis | % do total | Pico de memória (GB) | Tempo de treino | Perda final | eval_loss antes → depois | Tamanho do adaptador |
|---|---|---|---|---|---|---|---|
| LoRA (base 16 bits, r=16) | 8.798.208 | 1,75% | 1,51 | 264 s (2,64 s/passo) | 1,438 | 1,807 → 1,768 | 35,2 MB |
| QLoRA (base NF4, r=16) | 8.798.208 | 1,75% | 1,33 | 348 s (3,48 s/passo) | 1,491 | 1,903 → 1,842 | 35,2 MB |
| QLoRA (r=4) | 2.199.552 | 0,44% | 1,27 | 349 s (3,49 s/passo) | 1,633 | 1,903 → 1,839 | 8,9 MB |
| QLoRA (r=64) | 35.192.832 | 6,65% | 1,60 | 353 s (3,53 s/passo) | 1,149 | 1,903 → 1,891 | 140,8 MB |
| Full FT | não rodado | 100% | ver [fundamentos.md §1](fundamentos.md) e artigos | | | | modelo inteiro |

Fonte: `results/tabela_resultados.md` e `results/*/metricas.json`, GPU Tesla T4 no Colab, 15/11/2026. Os pesos do modelo base ocupam 0,99 GB em fp16 e 0,45 GB em NF4, 2,19× menos (`results/memoria_modelo.json`). As três contagens de parâmetros treináveis batem em todas as configurações, e todos os parâmetros treináveis são `lora_A`/`lora_B`.

### O que os resultados mostram

1. **A quantização custa qualidade antes mesmo do treino.** O `eval_loss` do modelo base sobe de 1,807 (fp16) para 1,903 (NF4), cerca de 5%. As respostas "antes" do QLoRA já são diferentes das do LoRA (`respostas.json`). Depois do treino, o QLoRA r=16 (1,842) não alcança o LoRA r=16 (1,768). Num modelo de 0,5B a perda pela quantização aparece bem; o artigo do QLoRA, com modelos de 7B a 65B, relata qualidade equivalente.
2. **QLoRA é mais lento por passo:** 3,48 s × 2,64 s, cerca de 32% a mais, por causa da dequantização.
3. **A economia de memória é pequena num modelo pequeno.** O pico caiu de 1,51 GB para 1,33 GB (−12%), não 2× nem 4×. Há três motivos:
   - só as camadas lineares vão para 4 bits;
   - as embeddings (151.936 × 896 ≈ 136M parâmetros, compartilhadas com o `lm_head`) são uma parte grande de um modelo de 0,5B;
   - o `prepare_model_for_kbit_training` converte essas camadas não quantizadas para fp32. Por isso o modelo QLoRA ocupa 0,72 GB no treino, contra 0,45 GB medidos logo depois de carregar.

   Num modelo de 7B, as lineares dominam e a redução se aproxima de 4× nos pesos (ver [fundamentos.md §4](fundamentos.md)).
4. **`r` maior não é melhor por si só.** O r=64 teve a menor perda de treino (1,149), mas o `eval_loss` piorou depois do passo 50 (1,847 → 1,891). Com 35M parâmetros e 2 épocas sobre 800 exemplos, ele decorou o treino (a queda brusca da perda no passo 50 é o início da 2ª época). O r=4 (0,44% dos parâmetros) chegou ao mesmo `eval_loss` que o r=16. Isso está de acordo com a hipótese do posto baixo.
5. **Os adaptadores são pequenos:** 8,9 a 141 MB, contra 0,99 GB do modelo em fp16. Foram salvos em fp32 (4 bytes por parâmetro); em fp16 teriam a metade.
6. **As respostas mudam pouco.** O Qwen-Instruct já responde bem, e 100 passos no Guanaco mudam mais o estilo que o conteúdo. O argumento quantitativo é o `eval_loss` e a curva em `results/curva_perda.png`.
