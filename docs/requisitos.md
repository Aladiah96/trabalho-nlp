# Requisitos

O enunciado quebrado em itens que dá para verificar. Um requisito só está "pronto" quando o critério da última coluna é cumprido.

## Enunciado

> Explique o método e analise um código, destacando preparação dos dados, treinamento e inferência. Apresente um exemplo de adaptação a uma tarefa e discuta sobreajuste e generalização.
>
> Cada apresentação deve durar até 25 minutos, contemplando os fundamentos da técnica, a análise de um código pronto, exemplos de aplicação e uma discussão sobre suas vantagens e limitações.

## Itens

| # | Requisito | Origem no enunciado | Critério de "pronto" |
|---|---|---|---|
| R1 | Explicar o método | "Explique o método"; "fundamentos da técnica" | Slides cobrem pré-treino × ajuste fino, a cabeça de tarefa sobre o encoder, a função de perda e quais pesos são atualizados. Cada afirmação tem fonte em `referencias.md`. |
| R2 | Código: preparação dos dados | "destacando preparação dos dados" | Bloco `# [PREPARAÇÃO DOS DADOS]` no código, com divisão treino/validação/teste, tokenização (truncamento, padding) e mapeamento de rótulos. Há slide com o trecho e um exemplo tokenizado gerado antes. |
| R3 | Código: treinamento | "destacando ... treinamento" | Bloco `# [TREINAMENTO]` com os hiperparâmetros explicados (taxa de aprendizado, épocas, batch, weight decay, warmup). Log do treino salvo em `results/`. |
| R4 | Código: inferência | "destacando ... inferência" | Bloco `# [INFERÊNCIA]` que carrega o modelo salvo e classifica textos novos. As saídas ficam salvas em `results/`. |
| R5 | Exemplo de adaptação a uma tarefa | "Apresente um exemplo de adaptação a uma tarefa"; "análise de um código pronto" | O modelo é treinado na tarefa escolhida e as métricas no teste (acurácia, F1) ficam em `results/`, comparadas com uma referência (modelo sem ajuste ou baseline simples). |
| R6 | Discutir sobreajuste | "discuta sobreajuste" | Curva de perda de treino × validação mostrando o ponto onde a validação piora. Mostra o efeito de pelo menos uma contramedida (early stopping, weight decay ou mais dados). |
| R7 | Discutir generalização | "discuta ... generalização" | Comparação do desempenho no teste do mesmo domínio com dados de outro domínio ou textos novos, e/ou desempenho em função do tamanho do conjunto de treino. |
| R8 | Exemplos de aplicação | "exemplos de aplicação" | Pelo menos 3 aplicações reais de ajuste fino citadas, com fonte. |
| R9 | Vantagens e limitações | "vantagens e limitações" | Lista equilibrada (ex.: custo, quantidade de dados, esquecimento catastrófico, instabilidade, viés), ligada aos resultados quando der. |
| R10 | Até 25 minutos | "deve durar até 25 minutos" | `roteiro-apresentacao.md` soma no máximo 25 min, e um ensaio cronometrado confirma. |
