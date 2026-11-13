"""Exemplo de fine-tuning com LoRA e QLoRA (transformers + peft + bitsandbytes + trl).

Modelo:  Qwen/Qwen2.5-0.5B-Instruct
Dataset: timdettmers/openassistant-guanaco (o mesmo do artigo do QLoRA), subconjunto pequeno.
Alvo:    Google Colab com GPU T4 (15 GB, sem bf16).

Os três elementos pedidos pelo professor estão marcados com comentários:
    # [QUANTIZAÇÃO]            -> BitsAndBytesConfig e prepare_model_for_kbit_training
    # [ADAPTADORES]            -> LoraConfig, get_peft_model e os módulos lora_A / lora_B
    # [PARÂMETROS TREINÁVEIS]  -> print_trainable_parameters() e contagem por requires_grad

Uso:
    python code/lora_qlora.py --config qlora_r16
    python code/lora_qlora.py --medir-memoria       # memória do modelo em 16 bits x 4 bits
    python code/lora_qlora.py --config lora16 --teste-cpu   # teste rápido de API, sem GPU

Cada execução grava em results/<config>/:
    metricas.json, log_history.json, parametros.txt, modelo_print.txt, respostas.json, adaptador/
"""

import argparse
import contextlib
import gc
import io
import json
import platform
import re
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft.tuners.lora import LoraLayer
from trl import SFTConfig, SFTTrainer

MODELO_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_ID = "timdettmers/openassistant-guanaco"
RAIZ = Path(__file__).resolve().parent.parent
PASTA_RESULTADOS = RAIZ / "results"
SEMENTE = 42

# As quatro configurações do experimento. Só muda a quantização e o posto r.
# lora_alpha = 2·r mantém a escala α/r = 2 fixa, para isolar o efeito do posto.
CONFIGS = {
    "lora16": {"quantizar": False, "r": 16},
    "qlora_r16": {"quantizar": True, "r": 16},
    "qlora_r4": {"quantizar": True, "r": 4},
    "qlora_r64": {"quantizar": True, "r": 64},
}

# Hiperparâmetros de treino comuns a todas as configurações.
TREINO = {
    "n_treino": 800,          # exemplos de treino (subconjunto do split "train")
    "n_avaliacao": 100,       # exemplos de avaliação (subconjunto do split "test")
    "max_length": 512,        # tokens por exemplo (o resto é truncado)
    "batch": 4,
    "acumulacao": 4,          # batch efetivo = 4 x 4 = 16
    "max_steps": 100,         # 100 x 16 = 1600 exemplos ≈ 2 épocas
    "learning_rate": 2e-4,
    "eval_steps": 25,
}

# Perguntas fixas para comparar as respostas antes e depois do fine-tuning.
PROMPTS = [
    "Explique em poucas frases o que é aprendizado de máquina.",
    "Escreva um e-mail curto pedindo para remarcar uma reunião.",
    "What are three tips for staying focused while studying?",
]


# ---------------------------------------------------------------------------
# Carregamento do modelo
# ---------------------------------------------------------------------------

def carregar_modelo(quantizar: bool, gpu: bool = True):
    """Carrega o modelo base em 4 bits (QLoRA) ou em 16 bits (LoRA)."""
    if quantizar:
        # [QUANTIZAÇÃO] ------------------------------------------------------
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,                     # pesos do modelo base guardados em 4 bits
            bnb_4bit_quant_type="nf4",             # tipo NormalFloat 4 (NF4), proposto no QLoRA
            bnb_4bit_use_double_quant=True,        # quantiza também as constantes de escala de cada bloco
            bnb_4bit_compute_dtype=torch.float16,  # dtype das multiplicações (a T4 não tem bf16)
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODELO_ID, quantization_config=bnb_config, device_map={"": 0}
        )
        # [QUANTIZAÇÃO] prepara o modelo de 4 bits para treino: congela os pesos
        # base, converte as normalizações para fp32 e liga o gradient checkpointing.
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    else:
        # LoRA comum: modelo base em 16 bits (fp16), sem quantização.
        dtype = torch.float16 if gpu else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            MODELO_ID, dtype=dtype, device_map={"": 0} if gpu else None
        )
        for p in model.parameters():
            p.requires_grad = False  # pesos base congelados (o peft também faz isso)
        # Mesmo gradient checkpointing do QLoRA, para a comparação ser justa.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()

    tokenizer = AutoTokenizer.from_pretrained(MODELO_ID)
    return model, tokenizer


def medir_memoria() -> dict:
    """Memória ocupada pelos pesos do modelo em 16 bits e em 4 bits (slide de [QUANTIZAÇÃO])."""
    resultado = {}

    model = AutoModelForCausalLM.from_pretrained(MODELO_ID, dtype=torch.float16, device_map={"": 0})
    resultado["fp16_bytes"] = model.get_memory_footprint()
    del model
    _liberar_gpu()

    # [QUANTIZAÇÃO] mesma configuração usada no treino do QLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODELO_ID, quantization_config=bnb_config, device_map={"": 0}
    )
    resultado["nf4_bytes"] = model.get_memory_footprint()
    # Camadas lineares que ficaram em 4 bits (embeddings e lm_head continuam em 16 bits)
    resultado["exemplo_camada_4bit"] = repr(model.model.layers[0].self_attn.q_proj)
    del model
    _liberar_gpu()

    resultado["fp16_gb"] = round(resultado["fp16_bytes"] / 1e9, 3)
    resultado["nf4_gb"] = round(resultado["nf4_bytes"] / 1e9, 3)
    resultado["reducao_x"] = round(resultado["fp16_bytes"] / resultado["nf4_bytes"], 2)
    return resultado


# ---------------------------------------------------------------------------
# Adaptadores e parâmetros treináveis
# ---------------------------------------------------------------------------

def adicionar_lora(model, r: int):
    """Envolve as camadas lineares do modelo com adaptadores LoRA."""
    # [ADAPTADORES] -----------------------------------------------------------
    lora_config = LoraConfig(
        r=r,                     # posto das matrizes A (r x k) e B (d x r)
        lora_alpha=2 * r,        # escala da atualização = lora_alpha / r = 2
        lora_dropout=0.05,
        target_modules=[         # todas as lineares do bloco transformer (recomendação do QLoRA)
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )
    # [ADAPTADORES] get_peft_model troca cada camada alvo por uma lora.Linear
    # (ou lora.Linear4bit), que guarda a camada original em base_layer e
    # acrescenta lora_A e lora_B. B começa em zero, então ΔW = B·A = 0 no início.
    model = get_peft_model(model, lora_config)
    return model


def exemplo_camada_adaptada(model) -> str:
    """Texto de uma camada q_proj adaptada, mostrando base_layer, lora_A e lora_B."""
    # [ADAPTADORES]
    for nome, modulo in model.named_modules():
        if nome.endswith("q_proj") and isinstance(modulo, LoraLayer):
            return f"{nome}:\n{modulo}"
    return "(nenhuma camada q_proj adaptada encontrada)"


def contar_parametros(model) -> dict:
    """Conta os parâmetros treináveis de três formas e confere se batem."""
    # [PARÂMETROS TREINÁVEIS] ------------------------------------------------
    # 1) Contagem do próprio peft
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        model.print_trainable_parameters()
    saida_peft = buffer.getvalue().strip()
    print(saida_peft)
    treinaveis_peft, total_peft = model.get_nb_trainable_parameters()

    # 2) Contagem manual por requires_grad
    nomes_treinaveis = [n for n, p in model.named_parameters() if p.requires_grad]
    treinaveis_grad = sum(p.numel() for p in model.parameters() if p.requires_grad)
    so_lora = all(("lora_A" in n) or ("lora_B" in n) for n in nomes_treinaveis)

    # 3) Conta da teoria: r·(d + k) por camada adaptada.
    #    No Qwen2.5, k_proj e v_proj têm saída menor (atenção com GQA), então d ≠ k.
    treinaveis_formula = 0
    por_tipo = {}
    for nome, modulo in model.named_modules():
        if isinstance(modulo, LoraLayer):
            r = modulo.r["default"]
            k, d = modulo.in_features, modulo.out_features
            n = r * (d + k)
            treinaveis_formula += n
            tipo = nome.split(".")[-1]
            por_tipo.setdefault(tipo, {"d": d, "k": k, "r": r, "por_camada": n, "camadas": 0})
            por_tipo[tipo]["camadas"] += 1

    return {
        "print_trainable_parameters": saida_peft,
        "treinaveis": treinaveis_peft,
        "total": total_peft,
        "percentual": round(100 * treinaveis_peft / total_peft, 4),
        "treinaveis_requires_grad": treinaveis_grad,
        "treinaveis_formula_r_d_mais_k": treinaveis_formula,
        "contagens_batem": treinaveis_peft == treinaveis_grad == treinaveis_formula,
        "so_lora_A_e_lora_B": so_lora,
        "n_tensores_treinaveis": len(nomes_treinaveis),
        "exemplos_nomes_treinaveis": nomes_treinaveis[:4],
        "por_tipo_de_camada": por_tipo,
    }


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------

def _texto_para_mensagens(texto: str) -> list[dict]:
    """Converte '### Human: ... ### Assistant: ...' em lista de mensagens de chat."""
    partes = re.split(r"### (Human|Assistant):", texto)
    mensagens = []
    for papel, conteudo in zip(partes[1::2], partes[2::2]):
        mensagens.append({
            "role": "user" if papel == "Human" else "assistant",
            "content": conteudo.strip(),
        })
    return mensagens


def _para_prompt_completion(exemplo: dict) -> dict:
    mensagens = _texto_para_mensagens(exemplo["text"])
    # Termina na última resposta do assistente
    while mensagens and mensagens[-1]["role"] != "assistant":
        mensagens.pop()
    # Formato prompt/completion do trl: a perda é calculada só na resposta final.
    return {"prompt": mensagens[:-1], "completion": mensagens[-1:]}


def _valido(exemplo: dict) -> bool:
    return len(exemplo["prompt"]) >= 1 and len(exemplo["completion"]) == 1


def preparar_dataset(n_treino: int, n_avaliacao: int):
    bruto = load_dataset(DATASET_ID)
    dados = {}
    for split, n in [("train", n_treino), ("test", n_avaliacao)]:
        ds = bruto[split].shuffle(seed=SEMENTE)
        ds = ds.map(_para_prompt_completion, remove_columns=ds.column_names).filter(_valido)
        dados[split] = ds.select(range(min(n, len(ds))))
    return dados["train"], dados["test"]


# ---------------------------------------------------------------------------
# Geração (antes x depois)
# ---------------------------------------------------------------------------

@torch.no_grad()
def gerar(model, tokenizer, prompts, max_new_tokens: int = 150) -> list[str]:
    model.eval()
    respostas = []
    for prompt in prompts:
        entrada = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)
        saida = model.generate(
            **entrada,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # geração gulosa: resultado reprodutível
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
        novos = saida[0, entrada["input_ids"].shape[1]:]
        respostas.append(tokenizer.decode(novos, skip_special_tokens=True).strip())
    return respostas


# ---------------------------------------------------------------------------
# Execução completa de uma configuração
# ---------------------------------------------------------------------------

def executar(nome: str, quantizar: bool, r: int, teste_cpu: bool = False) -> dict:
    set_seed(SEMENTE)
    gpu = not teste_cpu
    treino = dict(TREINO)
    if teste_cpu:
        # Teste rápido só para validar as APIs, sem GPU e sem quantização.
        treino.update(n_treino=8, n_avaliacao=4, max_steps=2, batch=2, acumulacao=1,
                      eval_steps=1, max_length=128)
    saida = PASTA_RESULTADOS / (nome + ("_teste_cpu" if teste_cpu else ""))
    saida.mkdir(parents=True, exist_ok=True)

    print(f"=== {nome}: quantizar={quantizar}, r={r} ===")
    model, tokenizer = carregar_modelo(quantizar, gpu=gpu)
    memoria_modelo = model.get_memory_footprint()

    # Respostas do modelo original, antes de adicionar os adaptadores
    respostas_antes = gerar(model, tokenizer, PROMPTS, max_new_tokens=20 if teste_cpu else 150)

    model = adicionar_lora(model, r)
    (saida / "modelo_print.txt").write_text(
        exemplo_camada_adaptada(model) + "\n\n" + "=" * 80 + "\n\n" + str(model), encoding="utf-8"
    )
    parametros = contar_parametros(model)
    (saida / "parametros.txt").write_text(
        json.dumps(parametros, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(exemplo_camada_adaptada(model))

    ds_treino, ds_avaliacao = preparar_dataset(treino["n_treino"], treino["n_avaliacao"])

    training_args = SFTConfig(
        output_dir=str(saida / "checkpoints"),
        per_device_train_batch_size=treino["batch"],
        per_device_eval_batch_size=treino["batch"],
        gradient_accumulation_steps=treino["acumulacao"],
        learning_rate=treino["learning_rate"],
        lr_scheduler_type="cosine",
        warmup_steps=5,
        max_steps=treino["max_steps"],
        max_length=treino["max_length"],
        logging_steps=1 if teste_cpu else 5,
        eval_strategy="steps",
        eval_steps=treino["eval_steps"],
        save_strategy="no",                 # não grava checkpoints; só o adaptador no fim
        # [QUANTIZAÇÃO] paged optimizer do QLoRA (estados em 8 bits, paginados para a CPU)
        optim="paged_adamw_8bit" if quantizar else "adamw_torch",
        fp16=gpu,                           # a T4 não tem bf16
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        seed=SEMENTE,
        use_cpu=teste_cpu,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds_treino,
        eval_dataset=ds_avaliacao,
        processing_class=tokenizer,
    )

    eval_antes = trainer.evaluate()["eval_loss"]
    if gpu:
        torch.cuda.reset_peak_memory_stats()
    inicio = time.time()
    saida_treino = trainer.train()
    tempo = time.time() - inicio
    pico = torch.cuda.max_memory_allocated() if gpu else None
    eval_depois = trainer.evaluate()["eval_loss"]

    respostas_depois = gerar(model, tokenizer, PROMPTS, max_new_tokens=20 if teste_cpu else 150)

    # Salva só o adaptador (lora_A e lora_B), não o modelo inteiro
    pasta_adaptador = saida / "adaptador"
    model.save_pretrained(pasta_adaptador)
    tamanho_adaptador = sum(f.stat().st_size for f in pasta_adaptador.rglob("*") if f.is_file())

    perdas_treino = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    metricas = {
        "config": nome,
        "quantizar": quantizar,
        "r": r,
        "lora_alpha": 2 * r,
        "treino": treino,
        "modelo": MODELO_ID,
        "dataset": DATASET_ID,
        "n_treino_real": len(ds_treino),
        "n_avaliacao_real": len(ds_avaliacao),
        "treinaveis": parametros["treinaveis"],
        "total": parametros["total"],
        "percentual_treinavel": parametros["percentual"],
        "contagens_batem": parametros["contagens_batem"],
        "memoria_modelo_base_gb": round(memoria_modelo / 1e9, 3),
        "pico_memoria_treino_gb": round(pico / 1e9, 3) if pico else None,
        "tempo_treino_s": round(tempo, 1),
        "segundos_por_passo": round(tempo / treino["max_steps"], 2),
        "perda_treino_final": perdas_treino[-1] if perdas_treino else None,
        "perda_treino_media": round(saida_treino.training_loss, 4),
        "eval_loss_antes": round(eval_antes, 4),
        "eval_loss_depois": round(eval_depois, 4),
        "tamanho_adaptador_mb": round(tamanho_adaptador / 1e6, 2),
        "ambiente": _ambiente(),
    }
    (saida / "metricas.json").write_text(json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8")
    (saida / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=2), encoding="utf-8")
    (saida / "respostas.json").write_text(
        json.dumps(
            [{"prompt": p, "antes": a, "depois": d} for p, a, d in zip(PROMPTS, respostas_antes, respostas_depois)],
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in metricas.items() if k not in ("treino", "ambiente")}, indent=2))
    return metricas


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def _liberar_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _ambiente() -> dict:
    import bitsandbytes
    import peft
    import transformers
    import trl

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "trl": trl.__version__,
        "bitsandbytes": bitsandbytes.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Exemplo de LoRA/QLoRA com Qwen2.5-0.5B.")
    parser.add_argument("--config", choices=sorted(CONFIGS), help="configuração a executar")
    parser.add_argument("--medir-memoria", action="store_true",
                        help="mede a memória do modelo em 16 bits e em 4 bits")
    parser.add_argument("--teste-cpu", action="store_true",
                        help="teste rápido sem GPU (só para validar as APIs; não usar nos slides)")
    args = parser.parse_args()

    if args.medir_memoria:
        PASTA_RESULTADOS.mkdir(exist_ok=True)
        resultado = medir_memoria()
        (PASTA_RESULTADOS / "memoria_modelo.json").write_text(
            json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(resultado, indent=2, ensure_ascii=False))
    if args.config:
        cfg = CONFIGS[args.config]
        if args.teste_cpu and cfg["quantizar"]:
            parser.error("--teste-cpu só funciona com --config lora16 (4 bits exige GPU)")
        executar(args.config, cfg["quantizar"], cfg["r"], teste_cpu=args.teste_cpu)
    if not (args.config or args.medir_memoria):
        parser.print_help()


if __name__ == "__main__":
    main()
