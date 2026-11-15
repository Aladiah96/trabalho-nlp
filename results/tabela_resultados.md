| Configuração | Parâmetros treináveis | % do total | Pico de memória no treino (GB) | Tempo de treino (s) | s/passo | Perda de treino final | eval_loss antes → depois | Tamanho do adaptador (MB) |
|---|---|---|---|---|---|---|---|---|
| LoRA (base 16 bits, r=16) | 8.798.208 | 1.75% | 1.513 | 263.8 | 2.64 | 1.438 | 1.807 → 1.768 | 35.24 |
| QLoRA (base NF4, r=16) | 8.798.208 | 1.75% | 1.331 | 348.2 | 3.48 | 1.491 | 1.903 → 1.842 | 35.24 |
| QLoRA (r=4) | 2.199.552 | 0.44% | 1.268 | 349.1 | 3.49 | 1.633 | 1.903 → 1.839 | 8.85 |
| QLoRA (r=64) | 35.192.832 | 6.65% | 1.597 | 352.7 | 3.53 | 1.149 | 1.903 → 1.891 | 140.82 |

Memória dos pesos do modelo base: fp16 = 0.988 GB, NF4 = 0.451 GB (2.19× menor).

Ambiente: python 3.13.16, torch 2.14.1+cu130, transformers 5.22.1, peft 0.21.2, trl 1.17.1, bitsandbytes 0.50.2, gpu Tesla T4
