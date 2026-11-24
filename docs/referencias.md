# Referências

Toda afirmação numérica dos slides precisa vir de uma fonte desta lista ou de `results/`. Status: `a conferir` (citação ainda não verificada), `resumo conferido` (só o abstract foi lido; serve para números do resumo) ou `conferida` (lida, com página/seção anotada).

## Fundamentos

| Ref | Fonte | Uso previsto | Status |
|---|---|---|---|
| [1] | Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", NAACL 2019 | Pré-treino + ajuste fino; cabeça de classificação | conferida (resumo + apêndice A.3: hiperparâmetros e dropout) |
| [2] | Howard e Ruder, "Universal Language Model Fine-tuning for Text Classification" (ULMFiT), ACL 2018 | Transferência de aprendizado em NLP; ajuste gradual | resumo conferido |
| [3] | Souza, Nogueira e Lotufo, "BERTimbau: Pretrained BERT Models for Brazilian Portuguese", BRACIS 2020 | Modelo usado no exemplo | card do modelo conferido (110 M / 335 M, brWaC) |

## Sobreajuste, estabilidade e generalização

| Ref | Fonte | Uso previsto | Status |
|---|---|---|---|
| [4] | Mosbach, Andriushchenko e Klakow, "On the Stability of Fine-tuning BERT", ICLR 2021 | Instabilidade do ajuste fino com poucos dados | resumo conferido |
| [5] | Dodge et al., "Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping", 2020 | Variação entre sementes; early stopping | resumo conferido |

## Aplicações e limitações

| Ref | Fonte | Uso previsto | Status |
|---|---|---|---|
| [10] | Lee et al., "BioBERT: a pre-trained biomedical language representation model for biomedical text mining", Bioinformatics 2020 (arXiv 1901.08746) | Aplicação: biomedicina; pré-treino no domínio | resumo conferido |
| [11] | Araci, "FinBERT: Financial Sentiment Analysis with Pre-trained Language Models", 2019 (arXiv 1908.10063) | Aplicação: finanças | resumo conferido |
| [12] | Chalkidis et al., "LEGAL-BERT: The Muppets straight out of Law School", Findings of EMNLP 2020 (arXiv 2010.02559) | Aplicação: direito; estratégias de adaptação ao domínio | resumo conferido |
| [13] | Ouyang et al., "Training language models to follow instructions with human feedback" (InstructGPT), NeurIPS 2022 (arXiv 2203.02155) | Aplicação: assistentes; ajuste fino supervisionado em LLMs | resumo conferido |
| [14] | Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022 (arXiv 2106.09685) | Limitação: custo do ajuste completo em modelos grandes | resumo conferido |
| [15] | Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks", PNAS 2017 (arXiv 1612.00796) | Limitação: esquecimento catastrófico | resumo conferido |

## Documentação

| Ref | Fonte | Uso previsto | Status |
|---|---|---|---|
| [6] | Hugging Face, documentação de `transformers` (guia de classificação de texto e `Trainer`) | Base do código | a conferir |
| [7] | Hugging Face, documentação de `datasets` e `evaluate` | Preparação dos dados e métricas | a conferir |

## Datasets

| Ref | Fonte | Uso previsto | Status |
|---|---|---|---|
| [8] | Real, Oshiro e Mafra, "B2W-Reviews01: an open product reviews corpus", STIL 2019. Hub: `ruanchaves/b2w-reviews01` (CC BY 4.0) | Dataset principal | conferida (citação do card) |
| [9] | Versão em português do Large Movie Review Dataset (Maas et al., "Learning Word Vectors for Sentiment Analysis", ACL 2011). Hub: `maritaca-ai/imdb_pt`; o card não informa licença nem como foi feita a tradução | Avaliação fora do domínio | conferida (citação do script do dataset) |
