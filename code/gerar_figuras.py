"""Lê results/<config>/ e gera os artefatos agregados para os slides.

Saídas:
    results/curva_perda.png         perda de treino e de avaliação das 4 configurações
    results/tabela_resultados.md    tabela no formato de docs/exemplos-e-tradeoffs.md

Uso:
    python code/gerar_figuras.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
RESULTADOS = RAIZ / "results"

# Ordem fixa das configurações e cores categóricas fixas (a cor segue a configuração).
ORDEM = ["lora16", "qlora_r16", "qlora_r4", "qlora_r64"]
ROTULOS = {
    "lora16": "LoRA (base 16 bits, r=16)",
    "qlora_r16": "QLoRA (base NF4, r=16)",
    "qlora_r4": "QLoRA (r=4)",
    "qlora_r64": "QLoRA (r=64)",
}
CORES = {
    "lora16": "#2a78d6",
    "qlora_r16": "#eb6834",
    "qlora_r4": "#1baf7a",
    "qlora_r64": "#eda100",
}
TEXTO = "#0b0b0b"
TEXTO_SECUNDARIO = "#52514e"
GRADE = "#e4e3df"


def carregar():
    dados = {}
    for nome in ORDEM:
        pasta = RESULTADOS / nome
        if not (pasta / "metricas.json").exists():
            print(f"aviso: {pasta} sem metricas.json, pulando")
            continue
        dados[nome] = {
            "metricas": json.loads((pasta / "metricas.json").read_text(encoding="utf-8")),
            "log": json.loads((pasta / "log_history.json").read_text(encoding="utf-8")),
        }
    return dados


def _estilo(ax, titulo):
    ax.set_title(titulo, loc="left", color=TEXTO, fontsize=12)
    ax.set_xlabel("passo de treino", color=TEXTO_SECUNDARIO)
    ax.set_ylabel("perda (entropia cruzada)", color=TEXTO_SECUNDARIO)
    ax.grid(axis="y", color=GRADE, linewidth=0.8)
    ax.tick_params(colors=TEXTO_SECUNDARIO)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(GRADE)


def curva_perda(dados):
    fig, (ax_treino, ax_eval) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)
    for nome, d in dados.items():
        treino = [(h["step"], h["loss"]) for h in d["log"] if "loss" in h]
        avaliacao = [(h["step"], h["eval_loss"]) for h in d["log"] if "eval_loss" in h]
        # O primeiro ponto de avaliação é antes do treino (passo 0)
        avaliacao = [(0, d["metricas"]["eval_loss_antes"])] + avaliacao
        for ax, pontos in ((ax_treino, treino), (ax_eval, avaliacao)):
            if not pontos:
                continue
            x, y = zip(*pontos)
            ax.plot(x, y, color=CORES[nome], linewidth=2, label=ROTULOS[nome],
                    marker="o" if ax is ax_eval else None, markersize=4)
    _estilo(ax_treino, "Perda de treino")
    _estilo(ax_eval, "Perda de avaliação (100 exemplos do split test)")
    ax_eval.legend(frameon=False, labelcolor=TEXTO)
    fig.tight_layout()
    caminho = RESULTADOS / "curva_perda.png"
    fig.savefig(caminho, facecolor="white")
    print(f"gravado: {caminho}")


def _milhar(n: int) -> str:
    """12345678 -> '12.345.678' (separador de milhar em português)."""
    return f"{n:,}".replace(",", ".")


def tabela(dados):
    linhas = [
        "| Configuração | Parâmetros treináveis | % do total | Pico de memória no treino (GB) "
        "| Tempo de treino (s) | s/passo | Perda de treino final | eval_loss antes → depois "
        "| Tamanho do adaptador (MB) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for nome, d in dados.items():
        m = d["metricas"]
        linhas.append(
            f"| {ROTULOS[nome]} | {_milhar(m['treinaveis'])} | {m['percentual_treinavel']:.2f}% "
            f"| {m['pico_memoria_treino_gb']} | {m['tempo_treino_s']} | {m['segundos_por_passo']} "
            f"| {m['perda_treino_final']:.3f} | {m['eval_loss_antes']:.3f} → {m['eval_loss_depois']:.3f} "
            f"| {m['tamanho_adaptador_mb']} |"
        )
    memoria = RESULTADOS / "memoria_modelo.json"
    if memoria.exists():
        mem = json.loads(memoria.read_text(encoding="utf-8"))
        linhas += [
            "",
            f"Memória dos pesos do modelo base: fp16 = {mem['fp16_gb']} GB, "
            f"NF4 = {mem['nf4_gb']} GB ({mem['reducao_x']}× menor).",
        ]
    if dados:
        amb = next(iter(dados.values()))["metricas"]["ambiente"]
        linhas += ["", "Ambiente: " + ", ".join(f"{k} {v}" for k, v in amb.items())]
    caminho = RESULTADOS / "tabela_resultados.md"
    caminho.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    print(f"gravado: {caminho}")
    print("\n".join(linhas))


if __name__ == "__main__":
    dados = carregar()
    if not dados:
        raise SystemExit("nenhum resultado encontrado em results/")
    curva_perda(dados)
    tabela(dados)
