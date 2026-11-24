# Ajuste fino de modelos pré-treinados para uma tarefa

Trabalho da disciplina de NLP: apresentação de até 25 minutos sobre como adaptar um modelo pré-treinado a uma tarefa, com análise de um código pronto.

## Enunciado do professor

> **Instruções:** Explique o método e analise um código, destacando preparação dos dados, treinamento e inferência. Apresente um exemplo de adaptação a uma tarefa e discuta sobreajuste e generalização.
>
> Cada apresentação deve durar até 25 minutos, contemplando os fundamentos da técnica, a análise de um código pronto, exemplos de aplicação e uma discussão sobre suas vantagens e limitações.

O método é o **ajuste fino (fine-tuning) supervisionado de um modelo pré-treinado**, em qualquer tarefa (confirmado pelo professor). Trabalho individual; apresentação na sexta-feira, 2026-11-27. Ver [docs/planejamento.md](docs/planejamento.md).

## Entregáveis

1. Apresentação (slides) de até 25 minutos.
2. Código de exemplo, com preparação dos dados, treinamento e inferência marcados nos comentários.
3. Resultados gerados antes da apresentação (métricas, curvas, logs). Não haverá execução ao vivo.

## Checklist de requisitos

Detalhes e critérios de "pronto" em [docs/requisitos.md](docs/requisitos.md). O andamento está em [docs/acompanhamento.md](docs/acompanhamento.md).

| # | Requisito | Onde será atendido | Status |
|---|---|---|---|
| R1 | Explicar o método | [docs/fundamentos.md](docs/fundamentos.md) | ◐ rascunho |
| R2 | Código: preparação dos dados | `code/`, `docs/analise-codigo.md` | ☐ |
| R3 | Código: treinamento | `code/`, `docs/analise-codigo.md` | ☐ |
| R4 | Código: inferência | `code/`, `docs/analise-codigo.md` | ☐ |
| R5 | Exemplo de adaptação a uma tarefa | `code/`, [results/tabela_resultados.md](results/tabela_resultados.md) | ◐ resultados gerados |
| R6 | Discussão de sobreajuste | `docs/sobreajuste-generalizacao.md`, `results/` | ◐ resultados gerados |
| R7 | Discussão de generalização | `docs/sobreajuste-generalizacao.md`, `results/` | ◐ resultados gerados |
| R8 | Exemplos de aplicação | `docs/aplicacoes-vantagens-limitacoes.md` | ☐ |
| R9 | Vantagens e limitações | `docs/aplicacoes-vantagens-limitacoes.md` | ☐ |
| R10 | Duração máxima de 25 minutos | [docs/roteiro-apresentacao.md](docs/roteiro-apresentacao.md) | ☐ |

Os docs de análise (`analise-codigo.md` etc.) ainda não existem. Eles serão criados nas fases seguintes do [planejamento](docs/planejamento.md).

## Como rodar

No Colab (GPU T4), abrir `code/notebook_colab.ipynb` e executar as células. Localmente, para testar o código em CPU com poucos exemplos:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu torch
uv pip install --python .venv/bin/python -r code/requirements.txt
.venv/bin/python code/ajuste_fino.py e2 --rapido   # saídas em results_teste_cpu/
```

## Estrutura do repositório

```
trabalho-nlp/
├── README.md                    # este arquivo
├── CLAUDE.md                    # regras e contexto para sessões com o Claude Code
├── code/
│   ├── ajuste_fino.py           # experimentos E1–E6, com os blocos marcados
│   ├── gerar_figuras.py         # figuras e tabela a partir de results/
│   ├── notebook_colab.ipynb     # roda tudo no Colab e gera results.zip
│   └── requirements.txt
├── results/                     # saídas do Colab (T4): JSON por experimento, figuras/, tabela_resultados.md
└── docs/
    ├── requisitos.md            # enunciado quebrado em requisitos verificáveis
    ├── fundamentos.md           # teoria do ajuste fino (R1)
    ├── planejamento.md          # escopo, decisões, riscos e fases
    ├── acompanhamento.md        # tarefas, decisões tomadas e pendências
    ├── roteiro-apresentacao.md  # blocos da apresentação e tempos
    └── referencias.md           # fontes
```
