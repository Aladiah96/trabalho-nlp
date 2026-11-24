"""Gera as figuras e a tabela de resultados a partir de results/.

Uso:
    python code/gerar_figuras.py            # lê results/, grava em results/figuras/

Cada figura sai em PDF (vetorial, para o Beamer) e PNG (para os docs).
Não treina nada: só lê os JSON gerados por code/ajuste_fino.py no Colab.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

RESULTADOS = Path("results")
FIGURAS = RESULTADOS / "figuras"

# Paleta categórica (validada para daltonismo) e tons de texto/grade
AZUL = "#2a78d6"
LARANJA = "#eb6834"
TEXTO = "#0b0b0b"
TEXTO_2 = "#52514e"
GRADE = "#e4e3df"
FUNDO = "#ffffff"

plt.rcParams.update({
    "font.size": 12,
    "axes.edgecolor": GRADE,
    "axes.labelcolor": TEXTO_2,
    "axes.titlecolor": TEXTO,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRADE,
    "grid.linewidth": 0.8,
    "xtick.color": TEXTO_2,
    "ytick.color": TEXTO_2,
    "legend.frameon": False,
    "figure.facecolor": FUNDO,
    "axes.facecolor": FUNDO,
    "savefig.bbox": "tight",
    "savefig.dpi": 200,
})


def ler(caminho):
    return json.loads((RESULTADOS / caminho).read_text(encoding="utf-8"))


def salvar(fig, nome):
    FIGURAS.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURAS / f"{nome}.{ext}")
    plt.close(fig)
    print(f"figura: {FIGURAS / nome}.{{pdf,png}}")


def pct(x, _=None):
    return f"{x:.0%}"


def por_epoca(log, chave):
    """Valores de uma chave do log_history, um por época."""
    return {round(x["epoch"]): x[chave] for x in log if chave in x}


def curva_sobreajuste():
    """E3: perda no treino × validação por época, com o ponto de parada do E4."""
    log = ler("e3_sobreajuste/log_history.json")
    treino = por_epoca(log, "treino_loss")
    validacao = por_epoca(log, "eval_loss")
    e4 = ler("e4_regularizado/metricas.json")
    melhor = min(validacao, key=validacao.get)
    parada = round(e4["epocas_executadas"])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(list(treino), list(treino.values()), color=AZUL, lw=2, marker="o", ms=5, label="Treino")
    ax.plot(list(validacao), list(validacao.values()), color=LARANJA, lw=2, marker="o", ms=5, label="Validação")

    ax.axvline(melhor, color=TEXTO_2, lw=1, ls="--")
    ax.annotate(f"menor perda de validação\n(época {melhor})", (melhor, validacao[melhor]),
                xytext=(melhor + 0.6, 0.06), color=TEXTO_2, fontsize=10)
    ax.axvline(parada, color=TEXTO_2, lw=1, ls=":")
    ax.annotate(f"early stopping\npara o E4 (época {parada})", (parada, 0.2),
                xytext=(parada + 0.4, 0.19), color=TEXTO_2, fontsize=10)

    ultima = max(validacao)
    ax.text(ultima + 0.3, validacao[ultima], "Validação", color=TEXTO, va="center", fontsize=11)
    ax.text(ultima + 0.3, treino[ultima], "Treino", color=TEXTO, va="center", fontsize=11)

    ax.set_xlim(0.5, ultima + 2.2)
    ax.set_xticks(range(1, ultima + 1, 2))
    ax.set_ylim(0, None)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ",")))
    ax.set_xlabel("Época")
    ax.set_ylabel("Perda (entropia cruzada)")
    ax.set_title("E3: 500 exemplos, 20 épocas, sem regularização")
    ax.legend(loc="upper right", bbox_to_anchor=(1, 0.92))
    salvar(fig, "curva_sobreajuste")


def barras(ax, nomes, valores, cor=AZUL):
    y = range(len(nomes))
    ax.barh(list(y), valores, color=cor, height=0.55)
    ax.set_yticks(list(y), nomes)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(FuncFormatter(pct))
    for i, v in zip(y, valores):
        ax.text(v + 0.01, i, f"{v:.1%}".replace(".", ","), va="center", color=TEXTO, fontsize=11)


def comparacao_metodos():
    """E1 × E2: F1 macro no teste da B2W."""
    e1 = ler("e1_referencia/metricas.json")
    e2 = ler("e2_ajuste_completo/metricas.json")
    nomes = ["Chutar sempre \"positivo\"", "TF-IDF + regressão logística", "BERTimbau com ajuste fino"]
    valores = [e1["classe_majoritaria"]["f1_macro"], e1["tfidf_regressao_logistica"]["f1_macro"],
               e2["teste"]["teste_f1_macro"]]
    fig, ax = plt.subplots(figsize=(8, 3))
    barras(ax, nomes, valores)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("F1 macro no teste (11.089 avaliações)")
    ax.set_title("Mesmos 20 mil exemplos de treino")
    salvar(fig, "comparacao_metodos")


def generalizacao():
    """E5: o mesmo modelo no domínio do treino e em outro domínio."""
    e5 = ler("e5_generalizacao/metricas.json")
    nomes = ["B2W: produtos (mesmo domínio)", "IMDB-pt: filmes (outro domínio)"]
    valores = [e5["b2w_teste_mesmo_dominio"]["teste_f1_macro"], e5["imdb_pt_teste_outro_dominio"]["imdb_f1_macro"]]
    fig, ax = plt.subplots(figsize=(8, 2.4))
    barras(ax, nomes, valores)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("F1 macro")
    ax.set_title("Modelo do E2, sem nenhum ajuste para filmes")
    salvar(fig, "generalizacao")


def matrizes_confusao():
    """E2 (B2W) e E5 (IMDB-pt) lado a lado, em % de cada classe real."""
    e2 = ler("e2_ajuste_completo/metricas.json")["matriz_confusao"]
    e5 = ler("e5_generalizacao/metricas.json")["imdb_matriz_confusao"]
    rotulos = e2["linhas_real_colunas_previsto"]
    fig, eixos = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, (titulo, m) in zip(eixos, [("B2W (produtos)", e2["valores"]), ("IMDB-pt (filmes)", e5["valores"])]):
        linhas = [[v / sum(linha) for v in linha] for linha in m]
        ax.imshow(linhas, cmap="Blues", vmin=0, vmax=1)
        ax.grid(False)
        for i in range(2):
            for j in range(2):
                cor = "white" if linhas[i][j] > 0.6 else TEXTO
                ax.text(j, i, f"{linhas[i][j]:.0%}\n({m[i][j]:,})".replace(",", "."),
                        ha="center", va="center", color=cor, fontsize=11)
        ax.set_xticks([0, 1], rotulos)
        ax.set_yticks([0, 1], rotulos)
        ax.set_xlabel("Previsto")
        ax.set_title(titulo)
        for lado in ax.spines.values():
            lado.set_visible(False)
    eixos[0].set_ylabel("Real")
    salvar(fig, "matrizes_confusao")


def curva_aprendizado():
    """E6: F1 macro em função do tamanho do treino, com as referências do E1."""
    e6 = ler("e6_curva_aprendizado/metricas.json")
    e1 = ler("e1_referencia/metricas.json")
    n = [x["n_treino"] for x in e6]
    f1 = [x["teste_f1_macro"] for x in e6]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(n, f1, color=AZUL, lw=2, marker="o", ms=6)
    for xi, yi in zip(n, f1):
        # o primeiro ponto fica colado na linha de referência; o rótulo vai à direita
        desloc, alinh = ((10, -4), "left") if xi == n[0] else ((0, 9), "center")
        ax.annotate(f"{yi:.1%}".replace(".", ","), (xi, yi), xytext=desloc, textcoords="offset points",
                    ha=alinh, color=TEXTO, fontsize=10)
    ax.axhline(e1["tfidf_regressao_logistica"]["f1_macro"], color=LARANJA, lw=1.5, ls="--")
    ax.text(9_500, e1["tfidf_regressao_logistica"]["f1_macro"] - 0.04,
            "TF-IDF + regressão logística (20 mil exemplos)", color=TEXTO_2, fontsize=10, ha="right")
    ax.axhline(e1["classe_majoritaria"]["f1_macro"], color=TEXTO_2, lw=1, ls=":")
    ax.text(9_500, e1["classe_majoritaria"]["f1_macro"] + 0.02, "Chutar sempre \"positivo\"",
            color=TEXTO_2, fontsize=10, ha="right")

    ax.set_xscale("log")
    ax.set_xticks(n, [f"{v:,}".replace(",", ".") for v in n])
    ax.minorticks_off()
    ax.set_ylim(0.3, 1.02)
    ax.yaxis.set_major_formatter(FuncFormatter(pct))
    ax.set_xlabel("Exemplos de treino (escala log)")
    ax.set_ylabel("F1 macro no teste")
    ax.set_title("BERTimbau, 3 épocas para cada tamanho")
    salvar(fig, "curva_aprendizado")


def tabela():
    """Tabela-resumo em Markdown com todos os números usados nos slides."""
    e1 = ler("e1_referencia/metricas.json")
    e2 = ler("e2_ajuste_completo/metricas.json")
    e3 = ler("e3_sobreajuste/metricas.json")
    e4 = ler("e4_regularizado/metricas.json")
    e5 = ler("e5_generalizacao/metricas.json")
    e6 = ler("e6_curva_aprendizado/metricas.json")

    def f(v):
        return f"{v:.1%}".replace(".", ",")

    def n(v):
        return f"{v:,}".replace(",", ".")

    def d(v, casas=3):
        return f"{v:.{casas}f}".replace(".", ",")

    linhas = [
        "# Tabela de resultados",
        "",
        "Gerada por `code/gerar_figuras.py` a partir dos JSON em `results/`. Colab, GPU Tesla T4.",
        "Teste da B2W: 11.089 avaliações. Teste do IMDB-pt: 5.000 críticas.",
        "",
        "## Comparação no teste da B2W",
        "",
        "| Experimento | Exemplos de treino | Épocas | Acurácia | F1 macro | Perda no teste | Tempo de treino | Pico de memória GPU |",
        "|---|---|---|---|---|---|---|---|",
        f"| E1 classe majoritária | {n(e2['hiperparametros']['n_treino'])} | — | {f(e1['classe_majoritaria']['acuracia'])} | {f(e1['classe_majoritaria']['f1_macro'])} | — | — | — |",
        f"| E1 TF-IDF + regressão logística | {n(e2['hiperparametros']['n_treino'])} | — | {f(e1['tfidf_regressao_logistica']['acuracia'])} | {f(e1['tfidf_regressao_logistica']['f1_macro'])} | — | — | — |",
    ]
    for nome, m in [("E2 ajuste fino completo", e2), ("E3 sem regularização", e3), ("E4 weight decay + early stopping", e4)]:
        t = m["teste"]
        linhas.append(
            f"| {nome} | {n(m['hiperparametros']['n_treino'])} | {m['epocas_executadas']:.0f} | {f(t['teste_acuracia'])} | "
            f"{f(t['teste_f1_macro'])} | {d(t['teste_loss'])} | {m['tempo_treino_s']:.0f} s | {d(m['memoria_gpu_pico_gb'], 2)} GB |"
        )
    imdb = e5["imdb_pt_teste_outro_dominio"]
    linhas += [
        "",
        "## Generalização (E5, modelo do E2)",
        "",
        "| Conjunto | Acurácia | F1 macro | Perda |",
        "|---|---|---|---|",
        f"| B2W teste (mesmo domínio) | {f(e2['teste']['teste_acuracia'])} | {f(e2['teste']['teste_f1_macro'])} | {d(e2['teste']['teste_loss'])} |",
        f"| IMDB-pt teste (outro domínio, até {e5['imdb_max_tokens']} tokens) | {f(imdb['imdb_acuracia'])} | {f(imdb['imdb_f1_macro'])} | {d(imdb['imdb_loss'])} |",
        "",
        "## Curva de aprendizado (E6, 3 épocas)",
        "",
        "| Exemplos de treino | Acurácia | F1 macro |",
        "|---|---|---|",
    ]
    linhas += [f"| {n(x['n_treino'])} | {f(x['teste_acuracia'])} | {f(x['teste_f1_macro'])} |" for x in e6]
    caminho = RESULTADOS / "tabela_resultados.md"
    caminho.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    print(f"tabela: {caminho}")


if __name__ == "__main__":
    curva_sobreajuste()
    comparacao_metodos()
    generalizacao()
    matrizes_confusao()
    curva_aprendizado()
    tabela()
