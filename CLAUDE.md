# CLAUDE.md

Contexto e regras para trabalhar neste repositório.

## O trabalho

- Disciplina de NLP. Tema: **Adaptação eficiente com LoRA e QLoRA**.
- Entrega principal: apresentação de **até 25 minutos** com fundamentos, análise de um código pronto, exemplos de aplicação, vantagens e limitações.
- O enunciado completo está em `README.md`; os requisitos verificáveis estão em `docs/requisitos.md`.

## Regras

- Escrever tudo em **português do Brasil** (docs, comentários do código, slides). Nomes de variáveis e APIs ficam como estão.
- Stack fixa: **Hugging Face `transformers` + `peft` + `bitsandbytes` + `trl` (`SFTTrainer`)**. Não trocar de biblioteca sem combinar antes.
- **Sem demo ao vivo.** O código roda antes e os resultados vão para `results/`. Os slides mostram trechos de código e saídas já geradas.
- O código precisa deixar explícitos, com comentários marcados, os três elementos que o professor pediu:
  1. `# [QUANTIZAÇÃO]`: `BitsAndBytesConfig` e `prepare_model_for_kbit_training`.
  2. `# [ADAPTADORES]`: `LoraConfig`, `get_peft_model` e os módulos `lora_A`/`lora_B`.
  3. `# [PARÂMETROS TREINÁVEIS]`: `print_trainable_parameters()` e a contagem por `requires_grad`.
- O exemplo deve caber numa GPU gratuita (Colab T4, 15 GB): modelo de 0,5B a 1,5B parâmetros, poucos passos de treino.
- Toda afirmação numérica dos slides (memória, % de parâmetros, resultados dos artigos) precisa ter fonte em `docs/referencias.md` ou vir de `results/`.
- O roteiro em `docs/roteiro-apresentacao.md` soma 25 min. Ao mexer num bloco, ajustar outro para não passar do limite.

## Onde está cada coisa

- Teoria: `docs/fundamentos.md`
- Análise do código: `docs/analise-codigo.md`
- Aplicações e trade-offs: `docs/exemplos-e-tradeoffs.md`
- Roteiro dos slides: `docs/roteiro-apresentacao.md`
- Referências: `docs/referencias.md`
- Código: `code/` (`lora_qlora.py`, `gerar_figuras.py`, `notebook_colab.ipynb`)
- Resultados (gerados no Colab): `results/`
- Slides (a criar): `slides/`
