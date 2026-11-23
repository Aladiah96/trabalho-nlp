# Planejamento

## Escopo

**Premissa sobre o tema.** O enunciado não dá nome ao método. Pelos pedidos ("adaptação a uma tarefa", "preparação dos dados, treinamento e inferência", "sobreajuste e generalização"), o trabalho trata do **ajuste fino (fine-tuning) supervisionado de um modelo de linguagem pré-treinado para uma tarefa específica**. Essa leitura precisa ser confirmada com o professor (pendência P1 em `acompanhamento.md`).

**Entra:**
- Fundamentos: transferência de aprendizado, pré-treino auto-supervisionado → ajuste fino supervisionado, cabeça de classificação, perda de entropia cruzada, ajuste fino completo × congelar camadas.
- Um código pronto e comentado com as quatro partes marcadas: preparação dos dados, treinamento, inferência e sobreajuste.
- Um exemplo rodado antes no Colab, com resultados salvos em `results/`.
- Discussão de sobreajuste e generalização apoiada nos resultados próprios.
- Aplicações, vantagens e limitações.

**Não entra:** demo ao vivo; métodos de ajuste eficiente (LoRA, adapters, prompt tuning) além de uma menção de contexto; treino de LLMs grandes.

## Decisões propostas (a confirmar)

| # | Decisão | Proposta | Motivo | Status |
|---|---|---|---|---|
| D1 | Stack | Hugging Face `transformers` (`Trainer`) + `datasets` + `evaluate` | É o padrão para código pronto de ajuste fino e mostra as três etapas de forma clara | Proposta |
| D2 | Modelo | BERTimbau base (`neuralmind/bert-base-portuguese-cased`) | Modelo treinado em português, do tamanho do BERT base, cabe na T4 | Proposta |
| D3 | Tarefa | Classificação de sentimento em pt-BR (positivo/negativo) | Fácil de explicar em pouco tempo; métricas simples; o resultado dá para entender olhando os exemplos | Proposta |
| D4 | Dataset | A escolher: avaliações de produtos ou filmes em pt-BR, com um segundo domínio para testar a generalização | Ver pendência P4 | Em aberto |
| D5 | Ambiente | Colab com T4 (15 GB), resultados exportados para `results/` | GPU gratuita; sem demo ao vivo | Proposta |
| D6 | Slides | LaTeX Beamer em `slides/` | Mesmo formato combinado para o curso | Proposta |

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
| A leitura do tema estar errada | Retrabalho grande | Confirmar com o professor antes de escrever código (P1) |
| Sessão do Colab cair ou ficar sem GPU | Experimentos incompletos | Experimentos curtos e independentes, salvando resultados a cada etapa |
| Dataset sem licença clara ou difícil de baixar | Atraso | Escolher entre datasets já no Hugging Face Hub |
| A tarefa ficar "fácil demais" e o sobreajuste não aparecer | R6 fica fraco | Usar o subconjunto pequeno do E3; aumentar épocas |
| Passar de 25 min | Corte na apresentação | Roteiro com tempos, ensaio cronometrado |

## Fases

Sem datas por enquanto, porque a data de entrega ainda não foi informada (P2).

1. **Planejamento**: este documento, requisitos, roteiro inicial. *(em andamento)*
2. **Fundamentos e referências**: `docs/fundamentos.md`, fontes em `referencias.md`.
3. **Código**: `code/` com os blocos marcados; teste rápido em CPU com poucos exemplos.
4. **Execução no Colab**: experimentos E1–E5, resultados em `results/`.
5. **Análise**: `docs/analise-codigo.md`, `docs/sobreajuste-generalizacao.md`, `docs/aplicacoes-vantagens-limitacoes.md`, figuras.
6. **Slides**: `slides/apresentacao.tex`.
7. **Ensaio**: cronometrar, ajustar o roteiro, revisar fontes dos números.
