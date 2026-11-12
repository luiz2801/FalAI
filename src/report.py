"""Tabelas e figuras do artigo a partir de results/raw.csv (guia.md, seção 4.7).

    python src/report.py

Só lê o CSV; não recalcula métrica. Saídas em results/tables/ (.csv e .md) e
results/figures/ (.png 300 dpi e .pdf). Tabelas e figuras sem dados são puladas.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from run_experiments import RAW_CSV, RESULTS
from stt import MODELS

TABLES, FIGURES = RESULTS / "tables", RESULTS / "figures"
COND_ORDER = ["limpo", "snr20", "snr10", "snr0"]
COND_LABEL = {"limpo": "limpo", "snr20": "20 dB", "snr10": "10 dB", "snr0": "0 dB"}
# paleta categórica de referência (ordem fixa; validada para pares adjacentes em linhas)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.figsize": (6.0, 3.6), "figure.dpi": 110, "savefig.dpi": 300,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.frameon": False, "hatch.color": "#ffffff", "hatch.linewidth": 1.2,
})


def _models(values) -> list[str]:
    presentes = set(values)
    return [m for m in MODELS if m in presentes] + sorted(presentes - set(MODELS))


def _save_fig(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("  ", FIGURES / f"{name}.png")


def _to_markdown(df: pd.DataFrame) -> str:
    idx = df.index.names if any(df.index.names) else [""]
    cab = [" / ".join(str(n) for n in idx if n)] + [str(c) for c in df.columns]
    linhas = ["| " + " | ".join(cab) + " |", "|" + "---|" * len(cab)]
    for i, r in df.iterrows():
        rotulo = " / ".join(map(str, i)) if isinstance(i, tuple) else str(i)
        linhas.append("| " + " | ".join([rotulo] + ["" if pd.isna(v) else str(v) for v in r]) + " |")
    return "\n".join(linhas) + "\n"


def _save_table(df: pd.DataFrame, name: str) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / f"{name}.csv")
    (TABLES / f"{name}.md").write_text(_to_markdown(df), encoding="utf-8")
    print("  ", TABLES / f"{name}.md")


def _mean_sd(s: pd.Series, fmt: str = "{:.3f}") -> str:
    return f"{fmt.format(s.mean())} ± {fmt.format(s.std(ddof=1) if len(s) > 1 else 0)} (n={len(s)})"


def load(path=RAW_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    chave = ["video", "tarefa", "modelo", "condicao", "repeticao", "metrica_erro"]
    return df.drop_duplicates(subset=chave, keep="last")  # execuções refeitas valem pela última


# ------------------------------------------------------------------ RQ1
def table1_wer(stt: pd.DataFrame) -> None:
    w = stt[stt.metrica_erro == "wer"]
    if w.empty:
        return
    t = w.groupby(["modelo", "condicao"])["valor_erro"].apply(_mean_sd).unstack()
    t = t.reindex(index=_models(t.index), columns=[c for c in COND_ORDER + ["limpo_vocab"] if c in t.columns])
    _save_table(t, "tabela1_wer")


def table3_time(stt: pd.DataFrame) -> None:
    w = stt[stt.metrica_erro == "wer"].copy()
    if w.empty:
        return
    w["s_por_min"] = w.tempo_s / (w.duracao_s / 60)
    t = w.groupby("modelo").agg(s_por_minuto=("s_por_min", lambda s: _mean_sd(s, "{:.2f}")),
                                rtf=("rtf", _mean_sd))
    _save_table(t.reindex(_models(t.index)), "tabela3_tempo_stt")


def fig1_wer_snr(stt: pd.DataFrame) -> None:
    w = stt[(stt.metrica_erro == "wer") & stt.condicao.isin(COND_ORDER)]
    if w.condicao.nunique() < 2:
        return
    conds = [c for c in COND_ORDER if c in set(w.condicao)]
    fig, ax = plt.subplots()
    for k, m in enumerate(_models(w.modelo)[:len(SERIES)]):
        g = w[w.modelo == m].groupby("condicao")["valor_erro"].mean().reindex(conds) * 100
        ax.plot(range(len(conds)), g.values, color=SERIES[k], lw=2, marker="o", ms=8, mec=SURFACE, mew=2,
                label=m, zorder=3)
        ax.annotate(m, (len(conds) - 1, g.values[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK_2)
    ax.set_xticks(range(len(conds)), [COND_LABEL[c] for c in conds])
    ax.set_xlim(-0.2, len(conds) - 0.4)
    ax.set_ylim(0, None)
    ax.set_xlabel("Condição do áudio (SNR do ruído branco)")
    ax.set_ylabel("WER médio (%)")
    ax.set_title("Robustez ao ruído: WER × SNR")
    ax.legend(loc="upper left", fontsize=8)
    _save_fig(fig, "fig1_wer_snr")


def fig2_rtf_wer(stt: pd.DataFrame) -> None:
    w = stt[(stt.metrica_erro == "wer") & (stt.condicao == "limpo")]
    if w.modelo.nunique() < 1 or w.empty:
        return
    g = w.groupby("modelo").agg(wer=("valor_erro", "mean"), rtf=("rtf", "mean")).reindex(_models(w.modelo))
    fig, ax = plt.subplots()
    ax.plot(g.rtf, g.wer * 100, color=SERIES[0], lw=1, alpha=0.4, zorder=2)
    ax.scatter(g.rtf, g.wer * 100, s=70, color=SERIES[0], edgecolors=SURFACE, linewidths=2, zorder=3)
    for m, r in g.iterrows():  # um ponto por modelo, identificado pelo rótulo (série única)
        ax.annotate(m, (r.rtf, r.wer * 100), xytext=(7, 4), textcoords="offset points", fontsize=8, color=INK_2)
    ax.set_xlabel("RTF (tempo de inferência ÷ duração)")
    ax.set_ylabel("WER médio, áudio limpo (%)")
    ax.set_xlim(0, g.rtf.max() * 1.2)  # folga para os rótulos
    ax.set_ylim(0, g.wer.max() * 100 * 1.15)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_title("Trade-off qualidade × custo do STT")
    _save_fig(fig, "fig2_rtf_wer")


def fig_vocab(stt: pd.DataFrame) -> None:
    a = stt[(stt.metrica_erro == "acerto_termos") & stt.condicao.isin(["limpo", "limpo_vocab"])]
    if a.condicao.nunique() < 2:
        return
    ordem = _models(a.modelo)
    t = a.groupby(["modelo", "condicao"])["valor_erro"].mean().unstack().reindex(ordem) * 100
    fig, ax = plt.subplots()
    w = 0.3
    for k, (cond, rotulo, hatch) in enumerate([("limpo", "sem vocabulário", None),
                                               ("limpo_vocab", "com vocabulário", "//")]):
        xs = [i + (k - 0.5) * (w + 0.03) for i in range(len(ordem))]
        ax.bar(xs, t[cond].values, width=w, color=SERIES[k], hatch=hatch, label=rotulo, zorder=2)
        for xi, v in zip(xs, t[cond].values, strict=True):
            ax.annotate(f"{v:.0f}", (xi, v), xytext=(0, 3), textcoords="offset points", ha="center",
                        va="bottom", fontsize=8)
    ax.set_xticks(range(len(ordem)), ordem)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Termos técnicos reconhecidos (%)")
    ax.set_title("Acerto de termos técnicos com e sem initial_prompt")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncols=2, fontsize=8)
    _save_fig(fig, "fig3_termos_vocabulario")


# ------------------------------------------------------------------ análise de erros (regra 11)
def table5_errors() -> None:
    """Erros classificados à mão no FalAI Studio (results/erros.csv): contagem por modelo × categoria."""
    arq = RESULTS / "erros.csv"
    if not arq.exists():
        return
    e = pd.read_csv(arq)
    if e.empty:
        return
    t = e.pivot_table(index="modelo", columns="categoria", values="idx", aggfunc="count", fill_value=0)
    t["total"] = t.sum(axis=1)
    _save_table(t.reindex(_models(t.index)), "tabela5_erros_classificados")


# ------------------------------------------------------------------ F9
def summaries(df: pd.DataFrame) -> None:
    r = df[df.tarefa == "resumo"]
    if r.empty:
        return
    piv = r.pivot_table(index=["modelo", "condicao"], columns="metrica_erro", values="valor_erro", aggfunc="mean")
    tempo = r[r.metrica_erro == "tempo_llm"].groupby(["modelo", "condicao"])["tempo_s"].mean()
    piv["tempo_llm_s"] = tempo
    piv = piv.drop(columns=[c for c in ["tempo_llm"] if c in piv.columns])
    _save_table(piv.round(3), "tabela4_resumos")

    if "nota_media" not in piv.columns:
        return
    wer = df[(df.tarefa == "stt") & (df.metrica_erro == "wer") & (df.condicao == "limpo")] \
        .groupby("modelo")["valor_erro"].mean()
    notas = r[r.metrica_erro == "nota_media"]
    llms = list(dict.fromkeys(notas.condicao))[:3]  # dispersão: no máximo 3 séries (paleta)
    fig, ax = plt.subplots()
    for k, llm in enumerate(llms):
        g = notas[notas.condicao == llm].groupby("modelo")["valor_erro"].agg(["mean", "std"])
        g = g[g.index.isin(wer.index)].reindex([m for m in _models(g.index) if m in wer.index])
        if g.empty:
            continue
        x = wer[g.index] * 100
        ax.errorbar(x, g["mean"], yerr=g["std"].fillna(0), fmt="o", ms=8, color=SERIES[k], elinewidth=1,
                    capsize=3, mec=SURFACE, mew=2, label=llm, zorder=3)
        for m in g.index:
            ax.annotate(m, (x[m], g.loc[m, "mean"]), xytext=(6, 4), textcoords="offset points",
                        fontsize=8, color=INK_2)
    if not ax.has_data():
        plt.close(fig)
        return
    ax.set_xlabel("WER do modelo de STT, áudio limpo (%)")
    ax.set_ylabel("Nota média do resumo (1–5)")
    ax.set_ylim(1, 5.2)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_title("O erro do STT piora o resumo do LLM?")
    if len(llms) > 1:
        from matplotlib.lines import Line2D
        ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=7, color=SERIES[k], label=l)
                           for k, l in enumerate(llms)], loc="lower left", fontsize=8)
    _save_fig(fig, "fig4_wer_nota_resumo")


def main():
    if not RAW_CSV.exists():
        raise SystemExit("results/raw.csv não existe: rode src/run_experiments.py antes.")
    df = load()
    stt_df = df[df.tarefa == "stt"]
    table1_wer(stt_df)
    table3_time(stt_df)
    fig1_wer_snr(stt_df)
    fig2_rtf_wer(stt_df)
    fig_vocab(stt_df)
    summaries(df)
    table5_errors()


if __name__ == "__main__":
    main()
