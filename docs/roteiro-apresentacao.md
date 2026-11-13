# Roteiro da apresentação

Limite: **25 minutos**. O roteiro abaixo usa os 25. Se as perguntas contarem dentro do tempo, cortar 1 minuto do bloco 4 e 1 do bloco 6.

| Bloco | Conteúdo | Tempo | Acumulado | Base |
|---|---|---|---|---|
| 1 | Abertura e motivação | 2 min | 2 | [fundamentos.md §1](fundamentos.md) |
| 2 | LoRA | 5 min | 7 | [fundamentos.md §2](fundamentos.md) |
| 3 | QLoRA | 4 min | 11 | [fundamentos.md §3](fundamentos.md) |
| 4 | Análise do código | 6 min | 17 | [analise-codigo.md](analise-codigo.md) |
| 5 | Exemplo e aplicações | 3 min | 20 | [analise-codigo.md](analise-codigo.md), [exemplos-e-tradeoffs.md](exemplos-e-tradeoffs.md) |
| 6 | Compromissos, vantagens e limitações | 3 min | 23 | [exemplos-e-tradeoffs.md](exemplos-e-tradeoffs.md) |
| 7 | Conclusão e perguntas | 2 min | 25 | — |

22 slides, pouco mais de 1 min por slide em média.

## Slides

### Bloco 1: Abertura e motivação (2 min)
1. **Título:** Adaptação eficiente com LoRA e QLoRA. Nome(s), disciplina, data.
2. **Por que não treinar tudo?** Conta dos 16 bytes por parâmetro; 7B → ~112 GB. Pergunta para o público: "como adaptar esse modelo numa GPU de 16 GB?"

### Bloco 2: LoRA (5 min)
3. **Hipótese do posto baixo.** A mudança nos pesos cabe num subespaço pequeno.
4. **Diagrama:** W₀ congelado em paralelo com B·A treinável; h = W₀x + (α/r)BAx.
5. **Conta de parâmetros:** 4096×4096 vs r=8 → 0,39%.
6. **Detalhes:** B = 0 no início, escala α/r, `target_modules`, fusão dos pesos na inferência.
7. **Resultados do artigo:** 10.000× menos parâmetros, 3× menos memória, sem latência extra.

### Bloco 3: QLoRA (4 min)
8. **Ideia:** modelo base em 4 bits + LoRA em 16 bits por cima.
9. **NF4 e double quantization:** níveis pelos quantis da normal; quantizar as constantes de escala.
10. **Paged optimizers e compute dtype.**
11. **Resultados do artigo:** 65B em uma GPU de 48 GB; Guanaco.

### Bloco 4: Análise do código (6 min)
12. **Visão geral do pipeline:** carregar quantizado → preparar → adicionar LoRA → treinar → salvar adaptador.
13. **[QUANTIZAÇÃO]** `BitsAndBytesConfig` + memória 16 bits × 4 bits.
14. **[ADAPTADORES]** `LoraConfig` + trecho de `print(model)` com `lora_A`/`lora_B`.
15. **[PARÂMETROS TREINÁVEIS]** saída de `print_trainable_parameters()` e conferência da conta.
16. **Treino e salvamento:** `SFTConfig` (`paged_adamw_8bit`), tamanho do adaptador salvo.

### Bloco 5: Exemplo e aplicações (3 min)
17. **Resultado do exemplo:** curva de perda; respostas antes × depois.
18. **Aplicações:** assistente de domínio, idioma/estilo, vários adaptadores num modelo base, difusão.

### Bloco 6: Compromissos, vantagens e limitações (3 min)
19. **Tabela qualidade × memória × custo** com os números medidos (tabela de [exemplos-e-tradeoffs.md](exemplos-e-tradeoffs.md)).
20. **Vantagens e limitações** em duas colunas.

### Bloco 7: Conclusão (2 min)
21. **Resumo em três frases:** LoRA reduz o que é treinado; QLoRA reduz o que é guardado; os dois juntos levam fine-tuning de LLM para uma GPU comum.
22. **Referências e perguntas.**

## Checklist antes de apresentar

- [ ] Todos os números dos slides têm fonte (artigo ou `results/`).
- [ ] Trechos de código legíveis (fonte grande, só as linhas relevantes).
- [ ] Ensaio cronometrado abaixo de 25 min.
- [ ] Slides exportados em PDF como reserva.
