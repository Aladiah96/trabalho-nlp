# Adaptação eficiente com LoRA e QLoRA

Trabalho da disciplina de NLP: apresentação de até 25 minutos sobre LoRA e QLoRA, com análise de um código pronto.

## Enunciado do professor

> **Tema:** Adaptação eficiente com LoRA e QLoRA
>
> **Instruções:** Explique as duas técnicas e identifique, no código, os adaptadores, os parâmetros treináveis e a configuração de quantização. Apresente um exemplo e discuta os compromissos entre qualidade, memória e custo de treinamento.
>
> A apresentação deve durar até 25 minutos, contemplando os fundamentos da técnica, a análise de um código pronto, exemplos de aplicação e uma discussão sobre suas vantagens e limitações.

## Entregáveis

1. Apresentação (slides) de até 25 minutos.
2. Código de exemplo com LoRA/QLoRA (Hugging Face `transformers` + `peft` + `bitsandbytes` + `trl`).
3. Resultados gerados antes da apresentação (prints, métricas, logs). Não haverá execução ao vivo.

## Checklist de requisitos

Detalhes e critérios de "pronto" em [docs/requisitos.md](docs/requisitos.md).

| # | Requisito | Onde é atendido | Status |
|---|---|---|---|
| R1 | Explicar LoRA | [docs/fundamentos.md](docs/fundamentos.md) | ☐ |
| R2 | Explicar QLoRA | [docs/fundamentos.md](docs/fundamentos.md) | ☐ |
| R3 | Identificar os adaptadores no código | [docs/analise-codigo.md](docs/analise-codigo.md) | ☐ |
| R4 | Identificar os parâmetros treináveis no código | [docs/analise-codigo.md](docs/analise-codigo.md) | ☐ |
| R5 | Identificar a configuração de quantização no código | [docs/analise-codigo.md](docs/analise-codigo.md) | ☐ |
| R6 | Apresentar um exemplo | [docs/analise-codigo.md](docs/analise-codigo.md), `code/`, `results/` | ☐ |
| R7 | Exemplos de aplicação | [docs/exemplos-e-tradeoffs.md](docs/exemplos-e-tradeoffs.md) | ☐ |
| R8 | Compromissos qualidade × memória × custo; vantagens e limitações | [docs/exemplos-e-tradeoffs.md](docs/exemplos-e-tradeoffs.md) | ☐ |
| R9 | Duração máxima de 25 minutos | [docs/roteiro-apresentacao.md](docs/roteiro-apresentacao.md) | ☐ |

## Estrutura do repositório

```
trabalho-nlp/
├── README.md                     # este arquivo
├── CLAUDE.md                     # regras e contexto para sessões com o Claude Code
├── docs/
│   ├── requisitos.md             # enunciado quebrado em requisitos verificáveis
│   ├── fundamentos.md            # teoria de LoRA e QLoRA
│   ├── analise-codigo.md         # roteiro da análise do código pronto
│   ├── exemplos-e-tradeoffs.md   # aplicações, compromissos, vantagens e limitações
│   ├── roteiro-apresentacao.md   # slides e tempos
│   └── referencias.md            # artigos e documentação
├── code/
│   ├── lora_qlora.py             # pipeline LoRA/QLoRA com os blocos marcados
│   ├── gerar_figuras.py          # curva de perda e tabela a partir de results/
│   ├── notebook_colab.ipynb      # roda tudo no Colab (T4)
│   └── requirements.txt          # versões fixadas
├── results/                      # (gerado no Colab) métricas, logs e prints
└── slides/                       # (a criar) arquivo da apresentação
```

## Próximos passos

- [x] Escolher modelo e dataset do exemplo: Qwen2.5-0.5B-Instruct + openassistant-guanaco.
- [x] Escrever o código em `code/` (testado em CPU com `--teste-cpu`).
- [ ] Rodar `code/notebook_colab.ipynb` no Colab (T4) e copiar `results/` para o repositório.
- [ ] Preencher a tabela de resultados em [docs/exemplos-e-tradeoffs.md](docs/exemplos-e-tradeoffs.md).
- [ ] Montar os slides seguindo [docs/roteiro-apresentacao.md](docs/roteiro-apresentacao.md).
- [ ] Ensaiar e cronometrar (máximo de 25 min).
