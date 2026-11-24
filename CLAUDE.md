# CLAUDE.md

Contexto e regras para trabalhar neste repositório.

## O trabalho

- Disciplina de NLP. Tema: **ajuste fino (fine-tuning) supervisionado de um modelo pré-treinado para uma tarefa**.
- O professor pede: explicar o método; analisar um código destacando **preparação dos dados, treinamento e inferência**; apresentar um exemplo de adaptação a uma tarefa; discutir **sobreajuste e generalização**.
- Entrega principal: apresentação de **até 25 minutos** com fundamentos, análise de um código pronto, exemplos de aplicação, vantagens e limitações.
- O enunciado completo está em `README.md`; os requisitos verificáveis estão em `docs/requisitos.md`.

## Regras

- Escrever tudo em **português do Brasil** (docs, comentários do código, slides). Nomes de variáveis e APIs ficam como estão.
- Stack fixa: **Hugging Face `transformers` (`Trainer`) + `datasets` + `evaluate`**, modelo **BERTimbau base**, datasets **B2W-Reviews01** (principal) e **IMDB-pt** (fora do domínio). Não trocar sem combinar antes.
- **Sem demo ao vivo.** O código roda antes e os resultados vão para `results/`. Os slides mostram trechos de código e saídas já geradas.
- O código precisa deixar explícitas, com comentários marcados, as partes que o professor pediu:
  1. `# [PREPARAÇÃO DOS DADOS]`: carga do dataset, divisão treino/validação/teste, tokenização.
  2. `# [TREINAMENTO]`: modelo com cabeça de tarefa, `TrainingArguments`, `Trainer`, métricas.
  3. `# [INFERÊNCIA]`: predição em textos novos (ex.: `pipeline`).
  4. `# [SOBREAJUSTE]`: registro das perdas de treino e validação, regularização, early stopping.
- O exemplo precisa caber numa GPU gratuita (Colab T4, 15 GB). Os modelos do tamanho do BERT base (cerca de 110 M parâmetros) cabem com folga no ajuste fino completo.
- Toda afirmação numérica dos slides precisa ter fonte em `docs/referencias.md` ou vir de `results/`.
- O roteiro em `docs/roteiro-apresentacao.md` soma 25 min. Ao mexer num bloco, ajustar outro para não passar do limite.
- Ao concluir ou decidir algo, atualizar `docs/acompanhamento.md` (tarefas, decisões, pendências) e o status no checklist do `README.md`.

## Onde está cada coisa

- Requisitos: `docs/requisitos.md`
- Planejamento (escopo, decisões, riscos, fases): `docs/planejamento.md`
- Acompanhamento (tarefas, decisões, pendências): `docs/acompanhamento.md`
- Roteiro dos slides: `docs/roteiro-apresentacao.md`
- Referências: `docs/referencias.md`
- Fundamentos (teoria): `docs/fundamentos.md`
- Código: `code/ajuste_fino.py` (experimentos E1–E6; `--rapido` testa em CPU, saída em `results_teste_cpu/`, fora do git), `code/notebook_colab.ipynb`
- Figuras e tabela: `code/gerar_figuras.py` (lê `results/`, grava `results/figuras/` e `results/tabela_resultados.md`)
- Resultados (Colab T4, 23/11): `results/`; achados resumidos em `docs/acompanhamento.md`
- A criar: `docs/analise-codigo.md`, `docs/sobreajuste-generalizacao.md`, `docs/aplicacoes-vantagens-limitacoes.md`, `slides/` (Beamer; compilar com `latexmk -pdf` dentro de `slides/`)
