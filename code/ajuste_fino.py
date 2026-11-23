"""Ajuste fino do BERTimbau para classificação de sentimento (B2W-Reviews01).

Experimentos (ver docs/planejamento.md):
    e1  referência sem ajuste fino: classe majoritária e TF-IDF + regressão logística
    e2  ajuste fino completo com 20 mil avaliações
    e3  sobreajuste proposital: 500 avaliações, 20 épocas, sem regularização
    e4  o mesmo que e3, com weight decay e early stopping
    e5  generalização: modelo do e2 no IMDB-pt (outro domínio) e em frases novas
    e6  curva de aprendizado: desempenho em função do tamanho do treino (opcional)

Uso:
    python code/ajuste_fino.py e2
    python code/ajuste_fino.py e2 --rapido     # teste em CPU com poucos exemplos

As partes pedidas pelo professor estão marcadas nos comentários:
    [PREPARAÇÃO DOS DADOS], [TREINAMENTO], [INFERÊNCIA], [SOBREAJUSTE]
"""

import argparse
import json
import math
import time
from pathlib import Path

import evaluate
import numpy as np
import torch
from datasets import ClassLabel, Dataset, load_dataset
from sklearn.metrics import confusion_matrix
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    pipeline,
    set_seed,
)

MODELO = "neuralmind/bert-base-portuguese-cased"
DATASET = "ruanchaves/b2w-reviews01"
DATASET_FORA_DOMINIO = "maritaca-ai/imdb_pt"
# Os dois datasets usam scripts de carregamento, que o `datasets` >= 4 não
# executa mais. O Hugging Face mantém uma cópia convertida para parquet nesta
# revisão, com o mesmo conteúdo.
REVISAO_PARQUET = "refs/convert/parquet"
ROTULOS = ["negativo", "positivo"]
SEMENTE = 42

# Hiperparâmetros de cada experimento. Os do e2 seguem a faixa sugerida no
# artigo do BERT (taxa de 2e-5 a 5e-5, batch 16 ou 32, 2 a 4 épocas).
EXPERIMENTOS = {
    "e2": dict(n_treino=20_000, epocas=2, taxa=2e-5, weight_decay=0.01, batch=32,
               avaliar_por="steps", avaliar_treino=False, early_stopping=False),
    "e3": dict(n_treino=500, epocas=20, taxa=5e-5, weight_decay=0.0, batch=16,
               avaliar_por="epoch", avaliar_treino=True, early_stopping=False),
    "e4": dict(n_treino=500, epocas=20, taxa=5e-5, weight_decay=0.01, batch=16,
               avaliar_por="epoch", avaliar_treino=True, early_stopping=True),
}
PASTAS = {
    "e1": "e1_referencia",
    "e2": "e2_ajuste_completo",
    "e3": "e3_sobreajuste",
    "e4": "e4_regularizado",
    "e5": "e5_generalizacao",
    "e6": "e6_curva_aprendizado",
}

# Frases escritas à mão para a inferência. O rótulo é o que um leitor esperaria.
# Incluem outro domínio, negação, ironia e um caso neutro.
FRASES_NOVAS = [
    ("Chegou antes do prazo e funciona perfeitamente.", "positivo"),
    ("Não recomendo, quebrou na primeira semana.", "negativo"),
    ("O filme é longo, mas vale cada minuto.", "positivo"),
    ("O atendimento do restaurante foi péssimo.", "negativo"),
    ("Não é ruim.", "positivo"),
    ("Ótimo, mais um carregador que não carrega.", "negativo"),
    ("Esperava mais pelo preço que paguei.", "negativo"),
    ("A entrega foi feita na terça-feira.", "neutro"),
]


def salvar_json(dados, caminho):
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(dados, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# [PREPARAÇÃO DOS DADOS]
# ---------------------------------------------------------------------------

def carregar_b2w(n_treino=None, n_validacao=2_000, n_teste=None, pasta_saida=None):
    """Carrega a B2W, cria o rótulo binário e divide em treino/validação/teste.

    - Nota 1 ou 2 → negativo; 4 ou 5 → positivo; nota 3 (ambígua) sai.
    - Textos vazios e duplicados saem (duplicatas entre treino e teste
      inflariam a métrica).
    - A divisão 80/10/10 é estratificada pelo rótulo e tem semente fixa, então
      todos os experimentos usam o mesmo teste.
    """
    # [PREPARAÇÃO DOS DADOS] carga: o dataset só tem a divisão "train"
    bruto = load_dataset(DATASET, revision=REVISAO_PARQUET, split="train")

    # [PREPARAÇÃO DOS DADOS] limpeza e rótulo binário
    ds = bruto.filter(
        lambda ex: ex["review_text"] is not None
        and ex["review_text"].strip() != ""
        and ex["overall_rating"] != 3
    )
    ds = ds.map(
        lambda ex: {"texto": ex["review_text"].strip(), "label": int(ex["overall_rating"] >= 4)},
        remove_columns=ds.column_names,
    )
    tabela = ds.to_pandas().drop_duplicates(subset="texto")
    ds = Dataset.from_pandas(tabela, preserve_index=False)
    ds = ds.cast_column("label", ClassLabel(names=ROTULOS))

    # [PREPARAÇÃO DOS DADOS] divisão estratificada 80/10/10
    div = ds.train_test_split(test_size=0.2, stratify_by_column="label", seed=SEMENTE)
    resto = div["test"].train_test_split(test_size=0.5, stratify_by_column="label", seed=SEMENTE)
    treino, validacao, teste = div["train"], resto["train"], resto["test"]

    # Subconjuntos: o treino muda por experimento; a validação usada durante o
    # treino é reduzida para economizar tempo. O teste fica inteiro.
    if n_treino:
        treino = treino.shuffle(seed=SEMENTE).select(range(min(n_treino, len(treino))))
    if n_validacao:
        validacao = validacao.shuffle(seed=SEMENTE).select(range(min(n_validacao, len(validacao))))
    if n_teste:
        teste = teste.shuffle(seed=SEMENTE).select(range(min(n_teste, len(teste))))

    if pasta_saida:
        def contagem(d):
            positivos = int(sum(d["label"]))
            return {"total": len(d), "negativo": len(d) - positivos, "positivo": positivos}

        salvar_json(
            {
                "avaliacoes_no_dataset": len(bruto),
                "apos_limpeza_e_sem_duplicatas": len(ds),
                "treino_usado": contagem(treino),
                "validacao_usada": contagem(validacao),
                "teste": contagem(teste),
            },
            Path(pasta_saida) / "dados.json",
        )
    return treino, validacao, teste


def tokenizar(dataset, tokenizador, max_tokens):
    """[PREPARAÇÃO DOS DADOS] texto → ids de tokens.

    O truncamento corta textos longos em max_tokens. O padding não é feito
    aqui: o DataCollatorWithPadding completa cada lote até o maior texto dele,
    o que gasta menos memória que completar tudo até max_tokens.
    """
    return dataset.map(
        lambda lote: tokenizador(lote["texto"], truncation=True, max_length=max_tokens),
        batched=True,
    )


def descrever_tokenizacao(tokenizado, tokenizador, max_tokens, pasta_saida):
    """Salva um exemplo tokenizado (para o slide) e quantos textos foram truncados."""
    ex = tokenizado[0]
    tamanhos = np.array([len(ids) for ids in tokenizado["input_ids"]])
    salvar_json(
        {
            "texto": ex["texto"],
            "rotulo": ROTULOS[ex["label"]],
            "tokens": tokenizador.convert_ids_to_tokens(ex["input_ids"]),
            "input_ids": ex["input_ids"],
            "max_tokens": max_tokens,
            "tokens_por_texto": {
                "mediana": float(np.median(tamanhos)),
                "p95": float(np.percentile(tamanhos, 95)),
                "fracao_truncada": float((tamanhos >= max_tokens).mean()),
            },
        },
        Path(pasta_saida) / "tokenizacao.json",
    )


# ---------------------------------------------------------------------------
# [TREINAMENTO]
# ---------------------------------------------------------------------------

_acuracia = evaluate.load("accuracy")
_f1 = evaluate.load("f1")


def calcular_metricas(predicao):
    """Acurácia e F1 macro. O F1 macro importa porque as classes são desbalanceadas."""
    logits, rotulos = predicao
    previstos = np.argmax(logits, axis=-1)
    return {
        "acuracia": _acuracia.compute(predictions=previstos, references=rotulos)["accuracy"],
        "f1_macro": _f1.compute(predictions=previstos, references=rotulos, average="macro")["f1"],
    }


def contar_parametros(modelo):
    """Total, treináveis e só os da camada nova de classificação."""
    total = sum(p.numel() for p in modelo.parameters())
    treinaveis = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    cabeca = sum(p.numel() for p in modelo.classifier.parameters())
    return {"total": total, "treinaveis": treinaveis, "camada_nova_classificador": cabeca}


class AvaliarNoTreino(TrainerCallback):
    """[SOBREAJUSTE] Ao fim de cada época, mede a perda no próprio treino.

    A perda que o Trainer registra durante o treino é a média dos lotes, com
    dropout ligado e o modelo mudando no meio da época. Para comparar com a
    validação de forma justa, avaliamos o treino com o modelo em modo de
    avaliação. A diferença entre as duas curvas é o sinal do sobreajuste.
    """

    def __init__(self, dataset_treino):
        self.dataset_treino = dataset_treino
        self.trainer = None

    def on_epoch_end(self, args, state, control, **kwargs):
        # evaluate() desliga as flags de avaliar e registrar que o Trainer já
        # tinha marcado para o fim desta época; sem restaurá-las, a validação
        # da época não rodaria.
        avaliar, registrar = control.should_evaluate, control.should_log
        self.trainer.evaluate(self.dataset_treino, metric_key_prefix="treino")
        control.should_evaluate, control.should_log = avaliar, registrar
        return control


def treinar(nome, cfg, args, pasta):
    set_seed(SEMENTE)
    rapido = args.rapido
    max_tokens = 32 if rapido else args.max_tokens

    treino, validacao, teste = carregar_b2w(
        n_treino=min(cfg["n_treino"], 64) if rapido else cfg["n_treino"],
        n_validacao=32 if rapido else 2_000,
        n_teste=32 if rapido else None,
        pasta_saida=pasta,
    )
    tokenizador = AutoTokenizer.from_pretrained(MODELO)
    treino_tok = tokenizar(treino, tokenizador, max_tokens)
    validacao_tok = tokenizar(validacao, tokenizador, max_tokens)
    teste_tok = tokenizar(teste, tokenizador, max_tokens)
    descrever_tokenizacao(treino_tok, tokenizador, max_tokens, pasta)

    # [TREINAMENTO] encoder pré-treinado + camada de classificação nova
    # (pesos aleatórios), com 2 saídas. Todos os pesos serão ajustados.
    modelo = AutoModelForSequenceClassification.from_pretrained(
        MODELO,
        num_labels=len(ROTULOS),
        id2label=dict(enumerate(ROTULOS)),
        label2id={r: i for i, r in enumerate(ROTULOS)},
    )
    parametros = contar_parametros(modelo)
    print(f"[{nome}] parâmetros: {parametros}")

    epocas = 2 if rapido else cfg["epocas"]
    passos_por_epoca = math.ceil(len(treino_tok) / cfg["batch"])
    passos_total = passos_por_epoca * epocas
    # No e2, avalia 10 vezes e registra a perda de treino 20 vezes ao longo do
    # treino; no e3/e4, uma vez por época
    por_passos = {}
    if cfg["avaliar_por"] == "steps":
        por_passos = dict(eval_steps=max(1, passos_total // 10), logging_steps=max(1, passos_total // 20))

    # [TREINAMENTO] hiperparâmetros
    argumentos = TrainingArguments(
        output_dir=str(pasta / "checkpoints"),
        num_train_epochs=epocas,
        learning_rate=cfg["taxa"],
        per_device_train_batch_size=cfg["batch"],
        per_device_eval_batch_size=64,
        weight_decay=cfg["weight_decay"],          # [SOBREAJUSTE] regularização L2 (0 no e3)
        warmup_steps=int(0.1 * passos_total),      # taxa sobe aos poucos nos 10% iniciais
        lr_scheduler_type="linear",
        eval_strategy=cfg["avaliar_por"],
        logging_strategy=cfg["avaliar_por"],
        **por_passos,
        # [SOBREAJUSTE] com early stopping, guarda o melhor checkpoint pela
        # perda de validação e volta para ele no fim
        save_strategy="epoch" if cfg["early_stopping"] else "no",
        save_total_limit=1,
        load_best_model_at_end=cfg["early_stopping"],
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=SEMENTE,
    )

    callbacks = []
    if cfg["early_stopping"]:
        # [SOBREAJUSTE] para se a validação não melhorar por 3 épocas seguidas
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))
    avaliador_treino = AvaliarNoTreino(treino_tok) if cfg["avaliar_treino"] else None
    if avaliador_treino:
        callbacks.append(avaliador_treino)

    # [TREINAMENTO] o Trainer cuida do laço: lotes, perda de entropia cruzada,
    # retropropagação, otimizador AdamW, agenda da taxa e avaliações.
    trainer = Trainer(
        model=modelo,
        args=argumentos,
        train_dataset=treino_tok,
        eval_dataset=validacao_tok,
        processing_class=tokenizador,
        data_collator=DataCollatorWithPadding(tokenizador),
        compute_metrics=calcular_metricas,
        callbacks=callbacks,
    )
    if avaliador_treino:
        avaliador_treino.trainer = trainer

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    inicio = time.perf_counter()
    trainer.train()
    duracao = time.perf_counter() - inicio
    memoria_pico_gb = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else None

    # Avaliação final no teste, que o modelo nunca viu
    pred = trainer.predict(teste_tok, metric_key_prefix="teste")
    previstos = np.argmax(pred.predictions, axis=-1)

    salvar_json(trainer.state.log_history, pasta / "log_history.json")
    salvar_json(parametros, pasta / "parametros.json")
    salvar_json(
        {
            "experimento": nome,
            "hiperparametros": {**cfg, "epocas": epocas, "max_tokens": max_tokens},
            "epocas_executadas": trainer.state.epoch,
            "melhor_checkpoint": trainer.state.best_model_checkpoint,
            "teste": pred.metrics,
            "matriz_confusao": {
                "linhas_real_colunas_previsto": ROTULOS,
                "valores": confusion_matrix(pred.label_ids, previstos).tolist(),
            },
            "tempo_treino_s": round(duracao, 1),
            "memoria_gpu_pico_gb": round(memoria_pico_gb, 2) if memoria_pico_gb else None,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        },
        pasta / "metricas.json",
    )
    print(f"[{nome}] teste: {pred.metrics}")

    if nome == "e2":
        # Guarda o modelo ajustado para o e5 (os pesos ficam fora do git)
        trainer.save_model(str(pasta / "modelo"))
    return pred.metrics


# ---------------------------------------------------------------------------
# e1: referência sem ajuste fino
# ---------------------------------------------------------------------------

def referencia(args, pasta):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score

    treino, _, teste = carregar_b2w(
        n_treino=64 if args.rapido else EXPERIMENTOS["e2"]["n_treino"],
        n_teste=32 if args.rapido else None,
        pasta_saida=pasta,
    )

    def medir(previstos):
        return {
            "acuracia": accuracy_score(teste["label"], previstos),
            "f1_macro": f1_score(teste["label"], previstos, average="macro"),
        }

    # Chuta sempre a classe mais comum no treino
    majoritaria = int(np.bincount(treino["label"]).argmax())
    # Modelo clássico de saco de palavras, treinado com os mesmos exemplos do e2
    vetorizador = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50_000)
    x_treino = vetorizador.fit_transform(treino["texto"])
    classificador = LogisticRegression(max_iter=1_000).fit(x_treino, treino["label"])

    resultado = {
        "classe_majoritaria": {"classe": ROTULOS[majoritaria], **medir([majoritaria] * len(teste))},
        "tfidf_regressao_logistica": medir(classificador.predict(vetorizador.transform(teste["texto"]))),
    }
    salvar_json(resultado, pasta / "metricas.json")
    print(f"[e1] {resultado}")


# ---------------------------------------------------------------------------
# [INFERÊNCIA] e5: modelo do e2 em dados novos
# ---------------------------------------------------------------------------

def generalizacao(args, pasta):
    caminho_modelo = Path(args.saida) / PASTAS["e2"] / "modelo"
    if not caminho_modelo.exists():
        raise SystemExit(f"Rode o e2 antes: {caminho_modelo} não existe.")

    # [INFERÊNCIA] carregar o modelo ajustado e salvo pelo e2
    tokenizador = AutoTokenizer.from_pretrained(caminho_modelo)
    modelo = AutoModelForSequenceClassification.from_pretrained(caminho_modelo)
    dispositivo = 0 if torch.cuda.is_available() else -1

    # [INFERÊNCIA] frases novas com o pipeline: tokeniza, roda o modelo e
    # aplica softmax nos logits, devolvendo o rótulo e a probabilidade
    classificar = pipeline("text-classification", model=modelo, tokenizer=tokenizador, device=dispositivo)
    saidas = classificar([texto for texto, _ in FRASES_NOVAS])
    salvar_json(
        [
            {"texto": texto, "esperado": esperado, "previsto": s["label"], "probabilidade": round(s["score"], 4)}
            for (texto, esperado), s in zip(FRASES_NOVAS, saidas)
        ],
        pasta / "frases_novas.json",
    )

    # Outro domínio: críticas de filmes (IMDB traduzido). O modelo nunca viu
    # filmes no treino. Os textos são bem mais longos que as avaliações da B2W,
    # então o limite de tokens aqui é maior (o BERT aceita até 512).
    imdb = load_dataset(
        DATASET_FORA_DOMINIO,
        revision=REVISAO_PARQUET,
        data_files={"test": "plain_text/test/*.parquet"},
        split="test",
    )
    imdb = imdb.rename_column("text", "texto")
    if args.rapido:
        imdb = imdb.shuffle(seed=SEMENTE).select(range(32))
    max_tokens_imdb = 64 if args.rapido else 512
    imdb_tok = tokenizar(imdb, tokenizador, max_tokens_imdb)

    trainer = Trainer(
        model=modelo,
        args=TrainingArguments(output_dir=str(pasta / "tmp"), per_device_eval_batch_size=32,
                               fp16=torch.cuda.is_available(), report_to="none"),
        processing_class=tokenizador,
        data_collator=DataCollatorWithPadding(tokenizador),
        compute_metrics=calcular_metricas,
    )
    pred = trainer.predict(imdb_tok, metric_key_prefix="imdb")
    previstos = np.argmax(pred.predictions, axis=-1)

    metricas_b2w = json.loads((Path(args.saida) / PASTAS["e2"] / "metricas.json").read_text())["teste"]
    salvar_json(
        {
            "b2w_teste_mesmo_dominio": metricas_b2w,
            "imdb_pt_teste_outro_dominio": pred.metrics,
            "imdb_max_tokens": max_tokens_imdb,
            "imdb_matriz_confusao": {
                "linhas_real_colunas_previsto": ROTULOS,
                "valores": confusion_matrix(pred.label_ids, previstos).tolist(),
            },
        },
        pasta / "metricas.json",
    )
    print(f"[e5] IMDB-pt: {pred.metrics}")


# ---------------------------------------------------------------------------
# e6: curva de aprendizado (opcional)
# ---------------------------------------------------------------------------

def curva_aprendizado(args, pasta):
    tamanhos = [16, 32] if args.rapido else [100, 300, 1_000, 3_000, 10_000]
    resultados = []
    for n in tamanhos:
        cfg = {**EXPERIMENTOS["e2"], "n_treino": n, "epocas": 3, "avaliar_por": "epoch"}
        metricas = treinar(f"e6_n{n}", cfg, args, pasta / f"n{n}")
        resultados.append({"n_treino": n, **metricas})
        salvar_json(resultados, pasta / "metricas.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experimento", choices=list(PASTAS))
    parser.add_argument("--saida", default="results", help="pasta dos resultados")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--rapido", action="store_true",
                        help="poucos exemplos e 2 épocas, para testar o código em CPU")
    args = parser.parse_args()

    if args.rapido and args.saida == "results":
        args.saida = "results_teste_cpu"
    pasta = Path(args.saida) / PASTAS[args.experimento]
    pasta.mkdir(parents=True, exist_ok=True)

    if args.experimento == "e1":
        referencia(args, pasta)
    elif args.experimento == "e5":
        generalizacao(args, pasta)
    elif args.experimento == "e6":
        curva_aprendizado(args, pasta)
    else:
        treinar(args.experimento, EXPERIMENTOS[args.experimento], args, pasta)


if __name__ == "__main__":
    main()
