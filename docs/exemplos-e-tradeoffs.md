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
| LoRA (base 16 bits, r=16) | | | | | | | |
| QLoRA (base NF4, r=16) | | | | | | | |
| QLoRA (r=4) | | | | | | | |
| QLoRA (r=64) | | | | | | | |
| Full FT | não rodado | 100% | ver [fundamentos.md §1](fundamentos.md) e artigos | | | | modelo inteiro |
