# Referências

## Artigos

1. HU, Edward J. et al. **LoRA: Low-Rank Adaptation of Large Language Models.** arXiv:2106.09685, 2021. Publicado no ICLR 2022.
   https://arxiv.org/abs/2106.09685
2. DETTMERS, Tim et al. **QLoRA: Efficient Finetuning of Quantized LLMs.** arXiv:2305.14314, 2023. Publicado no NeurIPS 2023.
   https://arxiv.org/abs/2305.14314
3. DETTMERS, Tim et al. **LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale.** arXiv:2208.07339, 2022. Base da biblioteca `bitsandbytes`.
   https://arxiv.org/abs/2208.07339
4. AGHAJANYAN, Armen et al. **Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning.** arXiv:2012.13255, 2020. Motivação da hipótese de posto baixo do LoRA.
   https://arxiv.org/abs/2012.13255
5. RAJBHANDARI, Samyam et al. **ZeRO: Memory Optimizations Toward Training Trillion Parameter Models.** arXiv:1910.02054, 2019. Fonte da conta de ~16 bytes por parâmetro no treino com Adam e precisão mista.
   https://arxiv.org/abs/1910.02054

## Documentação

- Hugging Face PEFT: https://huggingface.co/docs/peft
- Guia de LoRA no PEFT: https://huggingface.co/docs/peft/developer_guides/lora
- Quantização com bitsandbytes no Transformers: https://huggingface.co/docs/transformers/quantization/bitsandbytes
- TRL (`SFTTrainer`): https://huggingface.co/docs/trl/sft_trainer
- Repositório bitsandbytes: https://github.com/bitsandbytes-foundation/bitsandbytes
- Repositório oficial do QLoRA: https://github.com/artidoro/qlora
- Repositório oficial do LoRA (Microsoft): https://github.com/microsoft/LoRA

## Modelo e dataset do exemplo

- Qwen2.5-0.5B-Instruct (card do modelo): https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
- openassistant-guanaco (subconjunto do OASST1 usado no artigo do QLoRA): https://huggingface.co/datasets/timdettmers/openassistant-guanaco

## Blog posts

- Hugging Face, **Making LLMs even more accessible with bitsandbytes, 4-bit quantization and QLoRA** (2023): https://huggingface.co/blog/4bit-transformers-bitsandbytes
