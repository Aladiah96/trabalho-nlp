# Planejamento

## Escopo

**Tema.** O trabalho trata do **ajuste fino (fine-tuning) supervisionado de um modelo de linguagem pré-treinado para uma tarefa específica**. A tarefa pode ser qualquer uma (confirmado pelo professor, P1). O trabalho é individual, a apresentação é na sexta-feira, 2026-11-27, e o código pode ser escrito por mim.

**Entra:**
- Fundamentos: transferência de aprendizado, pré-treino auto-supervisionado → ajuste fino supervisionado, cabeça de classificação, perda de entropia cruzada, ajuste fino completo × congelar camadas.
- Um código pronto e comentado com as quatro partes marcadas: preparação dos dados, treinamento, inferência e sobreajuste.
- Um exemplo rodado antes no Colab, com resultados salvos em `results/`.
- Discussão de sobreajuste e generalização apoiada nos resultados próprios.
- Aplicações, vantagens e limitações.

**Não entra:** demo ao vivo; métodos de ajuste eficiente (LoRA, adapters, prompt tuning) além de uma menção de contexto; treino de LLMs grandes.

## Decisões

| # | Decisão | Proposta | Motivo | Status |
|---|---|---|---|---|
| D1 | Stack | Hugging Face `transformers` (`Trainer`) + `datasets` + `evaluate` | É o padrão para código pronto de ajuste fino e mostra as três etapas de forma clara | Aprovada |
| D2 | Modelo | BERTimbau base (`neuralmind/bert-base-portuguese-cased`) | Modelo treinado em português, do tamanho do BERT base, cabe na T4 | Aprovada |
| D3 | Tarefa | Classificação de sentimento em pt-BR (positivo/negativo) | Fácil de explicar em pouco tempo; métricas simples; o resultado dá para entender olhando os exemplos | Proposta |
| D4 | Dataset | Principal: **B2W-Reviews01** (`ruanchaves/b2w-reviews01`), avaliações de produtos. Outro domínio: **IMDB em português** (`maritaca-ai/imdb_pt`), críticas de filmes | Os dois estão no Hub e são públicos, em pt-BR, com domínios diferentes (produto × filme), o que serve para o R7 | Aprovada |
| D5 | Ambiente | Colab com T4 (15 GB), resultados exportados para `results/` | GPU gratuita; sem demo ao vivo | Proposta |
| D6 | Slides | LaTeX Beamer em `slides/` | Mesmo formato combinado para o curso | Proposta |

### Datasets candidatos

Verificado em 2026-11-23 pela API do Hugging Face.

| Dataset | Conteúdo | Divisões | Licença | Como vira rótulo binário |
|---|---|---|---|---|
| `ruanchaves/b2w-reviews01` | 132.373 avaliações de produtos (texto, nota 1–5, `recommend_to_a_friend`, categoria) | Só `train`: a divisão treino/validação/teste fica no código, o que é bom para mostrar a preparação dos dados (R2) | CC BY 4.0 | Nota 1–2 → negativo, 4–5 → positivo, nota 3 descartada |
| `maritaca-ai/imdb_pt` | Críticas de filmes do IMDB traduzidas para o português (`text`, `label`) | `train` 25.000, `test` 5.000, `test_all` 25.000 | Não informada no card; conferir antes de usar | Já é binário |

Usar só o `test` do IMDB, e apenas como avaliação fora do domínio. O modelo não treina nele.

As colunas `site_category_lv1` da B2W permitem um segundo teste de generalização sem trocar de dataset: treinar em algumas categorias de produto e avaliar em outras.

### Tamanho dos modelos candidatos

Verificado em 2026-11-23 a partir dos arquivos de configuração no Hugging Face.

| Modelo | Parâmetros | Origem do número |
|---|---|---|
| BERTimbau base | 108,9 M (108.924.674 com a cabeça de 2 classes) | Calculado do `config.json`: vocabulário 29.794, dimensão 768, 12 camadas |
| BERTimbau large | 334,4 M | Calculado do `config.json`: dimensão 1024, 24 camadas |
| DistilBERT multilíngue (`distilbert/distilbert-base-multilingual-cased`) | 135,4 M | Metadado da página do modelo |

Estimativa de memória (não medida): no ajuste fino completo com AdamW em fp32, cada parâmetro gasta uns 16 bytes (pesos, gradientes e os dois estados do otimizador). Para o base são cerca de 1,7 GB, e para o large cerca de 5,4 GB, antes das ativações. Os dois cabem na T4. O número real vai ser medido no Colab e salvo em `results/`.

O DistilBERT multilíngue tem **mais** parâmetros que o BERTimbau base, por causa do vocabulário multilíngue. Ele só serve de plano B se o problema for tempo de execução (6 camadas em vez de 12), não memória.

## Experimentos planejados

| # | Experimento | Requisito | Saída esperada em `results/` |
|---|---|---|---|
| E1 | Referência sem ajuste fino (baseline simples, ex.: TF-IDF + regressão logística, ou classe majoritária) | R5 | Métricas no teste |
| E2 | Ajuste fino completo com o conjunto todo | R3, R5 | Log do treino, métricas no teste, matriz de confusão |
| E3 | Treino longo com um subconjunto pequeno (ex.: algumas centenas de exemplos, várias épocas) sem early stopping | R6 | Curvas de perda treino × validação mostrando o sobreajuste |
| E4 | O mesmo que E3, mas com early stopping e/ou weight decay | R6 | Curvas comparadas com E3 |
| E5 | Avaliação do modelo do E2 em outro domínio e em frases escritas à mão | R4, R7 | Métricas fora do domínio e predições comentadas |
| E6 (opcional) | Curva de aprendizado: métrica no teste em função do tamanho do conjunto de treino | R7 | Gráfico |

## Riscos

| Risco | Efeito | Mitigação |
|---|---|---|
| Prazo curto (4 dias até 27/11) | Sem tempo para slides e ensaio | Seguir o cronograma por dia; o E6 é opcional e sai primeiro se atrasar |
| Licença do IMDB-pt não informada | Problema ao citar o dataset | Conferir o card; plano B: teste entre categorias de produto da própria B2W |
| Sessão do Colab cair ou ficar sem GPU | Experimentos incompletos | Experimentos curtos e independentes, salvando resultados a cada etapa |
| Dataset sem licença clara ou difícil de baixar | Atraso | Escolher entre datasets já no Hugging Face Hub |
| A tarefa ficar "fácil demais" e o sobreajuste não aparecer | R6 fica fraco | Usar o subconjunto pequeno do E3; aumentar épocas |
| Passar de 25 min | Corte na apresentação | Roteiro com tempos, ensaio cronometrado |

## Fases

Apresentação na sexta-feira, 2026-11-27.

| Dia | Fases |
|---|---|
| Seg 23/11 | 1. **Planejamento** (este documento, requisitos, roteiro) · 2. **Fundamentos e referências** (`docs/fundamentos.md`, `referencias.md`) |
| Ter 24/11 | 3. **Código** (`code/` com os blocos marcados, teste em CPU) · 4. **Execução no Colab** (E1–E5, resultados em `results/`) |
| Qua 25/11 | 5. **Análise** (figuras, `analise-codigo.md`, `sobreajuste-generalizacao.md`, `aplicacoes-vantagens-limitacoes.md`) |
| Qui 26/11 | 6. **Slides** (`slides/apresentacao.tex`) · 7. **Ensaio** (cronometrar, ajustar o roteiro, revisar as fontes dos números) |
| Sex 27/11 | Apresentação |
