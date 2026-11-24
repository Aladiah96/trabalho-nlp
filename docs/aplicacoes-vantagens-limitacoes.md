# Aplicações, vantagens e limitações

Material do bloco 8 do roteiro (1,5 min, requisitos R8 e R9). As referências entre colchetes estão em `referencias.md`. Os números do nosso exemplo vêm de `results/`.

## 1. Exemplos de aplicação (R8)

| Área | Exemplo | O que foi ajustado | Fonte |
|---|---|---|---|
| Comércio eletrônico | Classificar avaliações de clientes por sentimento | BERTimbau na B2W (este trabalho): 95,7% de F1 | `results/` |
| Português em geral | Reconhecimento de entidades, similaridade de frases e inferência textual | BERTimbau ajustado em cada tarefa | [3] |
| Biomedicina | Entidades biomédicas, relações entre elas e perguntas e respostas | BioBERT: BERT com pré-treino extra em textos biomédicos, depois ajustado. No resumo: melhoria de 2,80% no F1 de extração de relações e de 12,24% no MRR de perguntas e respostas | [10] |
| Finanças | Sentimento em notícias e textos financeiros | FinBERT: supera o estado da arte da época em dois conjuntos, mesmo com menos dados e ajustando só parte do modelo | [11] |
| Direito | Classificação e extração em textos jurídicos | LEGAL-BERT: compara usar o BERT direto, continuar o pré-treino em textos jurídicos ou pré-treinar do zero | [12] |
| Assistentes conversacionais | Fazer um modelo de linguagem seguir instruções | InstructGPT: ajuste fino supervisionado com demonstrações humanas, seguido de RLHF. Respostas do modelo de 1,3 B foram preferidas às do GPT-3 de 175 B | [13] |

**Padrão comum:** todos partem de um modelo pré-treinado genérico e o adaptam com dados rotulados (ou demonstrações) da tarefa. Os de domínio especializado (biomedicina, direito) acrescentam uma etapa de pré-treino no texto do domínio antes do ajuste fino.

## 2. Vantagens (R9)

| Vantagem | Evidência |
|---|---|
| **Precisa de poucos dados rotulados** | Com 1.000 exemplos, 93,6% de F1, acima do TF-IDF com 20.000 (91,9%) (E6). O ULMFiT relata o mesmo efeito: 100 exemplos igualaram 100 vezes mais dados treinando do zero [2] |
| **Melhor desempenho que métodos clássicos** | 95,7% × 91,9% de F1 com os mesmos dados; o erro de acurácia cai de 6,7% para 3,6% (E1 × E2) |
| **Barato perto do pré-treino** | 4 minutos e 3,01 GB numa GPU gratuita (E2). O pré-treino do modelo é feito uma vez por quem o publica |
| **Mesma receita para várias tarefas** | Trocar a cabeça e os dados; a arquitetura não muda [1] |
| **Aprende algo além do domínio de treino** | 82% de acurácia em críticas de filmes sem nenhum exemplo de filme (E5) |

## 3. Limitações (R9)

| Limitação | Evidência |
|---|---|
| **Sobreajuste com poucos dados** | Com 500 exemplos, o treino chega a 100% de acurácia e a perda de validação dobra após a época 2 (E3). Exige validação separada e early stopping |
| **Confiança enganosa** | Ironia classificada errado com 99,6% de confiança (E5). A perda no teste do modelo sobreajustado é o dobro da do modelo com early stopping, com o mesmo F1 (E3 × E4) |
| **Sensível à mudança de domínio** | F1 cai de 95,7% para 81,9% em filmes, e 30% das críticas negativas viram positivas (E5) |
| **Herda vieses dos dados** | O treino é 70% positivo, e o modelo erra quase sempre para o lado positivo fora do domínio (E5) |
| **Presa às classes do treino** | Uma frase neutra precisa virar positiva ou negativa (E5), porque a nota 3 foi descartada na preparação |
| **Resultados instáveis** | Sementes aleatórias diferentes dão resultados bem diferentes com os mesmos hiperparâmetros [5]. A causa é a otimização (gradientes que somem), não só o tamanho do conjunto [4] |
| **Custo cresce com o modelo** | No ajuste completo, cada tarefa gera uma cópia de todos os pesos. Para o GPT-3 de 175 B, isso é proibitivo, e daí vêm os métodos eficientes como o LoRA, que reduz em 10.000 vezes os parâmetros treináveis e em 3 vezes a memória da GPU [14] |
| **Pode esquecer o pré-treino** | Ajustar todos os pesos numa tarefa pode apagar o que foi aprendido antes (esquecimento catastrófico) [15]. Taxa de aprendizado pequena e poucas épocas reduzem o risco |

## 4. Proposta de slides (bloco 8, 1,5 min)

1. **Aplicações** (0,5 min): a tabela da seção 1 reduzida a 4 linhas (B2W, BioBERT, FinBERT, InstructGPT), com um ícone ou palavra por área.
2. **Vantagens × limitações** (1 min): duas colunas, 3 itens cada, cada um com um número do nosso exemplo:
   - Vantagens: poucos dados (1.000 > TF-IDF com 20.000), melhor que o clássico (−46% de erro), barato (4 min, 3 GB).
   - Limitações: sobreajuste (validação piora após a época 2), mudança de domínio (−14 pontos), confiança enganosa (ironia com 99,6%).
