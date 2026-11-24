# Roteiro da apresentação

Limite: **25 minutos**. Ao mudar o tempo de um bloco, compensar em outro. Slides em `slides/apresentacao.pdf`. A numeração abaixo é a do rodapé dos slides (x/19); as páginas de seção não contam.

| # | Bloco | Slides | Conteúdo | Requisito | Tempo (min) |
|---|---|---|---|---|---|
| 1 | Abertura | 1 | Treinar do zero × reaproveitar; a pergunta da apresentação | — | 1 |
| 2 | Fundamentos | 2–5 | Duas fases (1,5); o que muda no modelo (1,5); perda e hiperparâmetros (1,5); variações e ULMFiT (1,5) | R1 | 6 |
| 3 | Código: preparação dos dados | 6–8 | O exemplo e os experimentos (1); rótulo e limpeza (1); divisão e tokenização (1) | R2 | 3 |
| 4 | Código: treinamento | 9–10 | Modelo e hiperparâmetros (2); o que o `Trainer` faz e o custo (2) | R3 | 4 |
| 5 | Código: inferência | 11 | `pipeline` e as 8 frases novas | R4 | 2 |
| 6 | Exemplo e resultados | 12–13 | Comparação com as referências (2); curva de aprendizado (2) | R5 | 4 |
| 7 | Sobreajuste e generalização | 14–15 | Curva do E3 e E3 × E4 (1,5); IMDB-pt e matrizes de confusão (1,5) | R6, R7 | 3 |
| 8 | Aplicações, vantagens e limitações | 16–17 | Aplicações (0,5); vantagens × limitações (1) | R8, R9 | 1,5 |
| 9 | Fechamento | 18 | Resumo e perguntas (o slide 19, de referências, fica na tela sem ser apresentado) | — | 0,5 |
| | **Total** | | | R10 | **25** |

## Pontos para não esquecer na fala

- **Slide 3:** só 1.538 parâmetros são novos, mas **todos** os 108,9 M são ajustados.
- **Slide 7:** a nota 3 sai na preparação; isso volta no slide 11 (frase neutra).
- **Slide 10:** explicar por que F1 macro: chutar "positivo" dá 70,6% de acurácia.
- **Slide 12:** o TF-IDF é forte; o ganho é real, mas não é mágica (erro de 6,7% → 3,6%).
- **Slide 13:** com 100 exemplos são só 12 atualizações dos pesos, então não é só falta de dados.
- **Slide 14:** a perda sobe e o F1 não, porque a entropia cruzada pune a confiança errada. O weight decay quase não fez diferença, e foi uma semente por experimento.
- **Slide 15:** as causas do erro assimétrico são hipóteses, não foram testadas.
