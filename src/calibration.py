"""Calibração da confiança do Whisper (F10, subproduto): a probabilidade de uma palavra prevê que ela está certa?

    uv run python src/calibration.py        # -> results/calibracao.csv (lido pelo report.py, figura 6)

Para cada hipótese guardada pela RQ1 e pela RQ6 (repetição 1, em data/derived/<video>/stt/<video>.<modelo>.<cond>.json)
de um vídeo com gabarito (data/gt/<video>.stt.txt), alinha com o gabarito (`metrics.word_flags`) e junta, por faixa de
probabilidade, quantas palavras estão certas. A curva diz se o gate `tau` da correção (L+F) faz sentido e o ECE
resume o desvio. Palavras trocadas pela correção (`orig`) ficam de fora: a probabilidade delas é a da palavra antiga.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import metrics
from run_experiments import DERIVED, GT, RESULTS

N_BINS = 10
CAMPOS = ["modelo", "condicao", "bin_lo", "bin_hi", "n", "conf_media", "acerto"]


def word_probs(segs: list[dict]) -> list[tuple[str, float]]:
    """[(token normalizado, probabilidade da palavra de origem)], na ordem do texto."""
    return [(tok, w["probability"]) for s in segs for w in (s.get("words") or []) if "orig" not in w
            for tok in metrics.norm(w["word"]).split()]


def collect(derived: Path = DERIVED, gt: Path = GT) -> list[dict]:
    """Uma linha por (modelo, condição, faixa de probabilidade)."""
    acc: dict[tuple, list[float]] = {}
    for ref_path in sorted(Path(gt).glob("*.stt.txt")):
        video = ref_path.name.removesuffix(".stt.txt")
        ref = ref_path.read_text(encoding="utf-8")
        for arq in sorted((Path(derived) / video / "stt").glob(f"{video}.*.json")):
            modelo, _, cond = arq.name.removeprefix(f"{video}.").removesuffix(".json").partition(".")
            pares = word_probs(json.loads(arq.read_text(encoding="utf-8"))["segments"]) if cond else []
            if not pares:
                continue
            for (_, p), certo in zip(pares, metrics.word_flags(ref, [t for t, _ in pares]), strict=True):
                a = acc.setdefault((modelo, cond, min(int(p * N_BINS), N_BINS - 1)), [0, 0, 0.0])
                a[0] += 1
                a[1] += certo
                a[2] += p
    return [{"modelo": m, "condicao": c, "bin_lo": b / N_BINS, "bin_hi": (b + 1) / N_BINS, "n": int(n),
             "conf_media": round(sp / n, 4), "acerto": round(ok / n, 4)}
            for (m, c, b), (n, ok, sp) in sorted(acc.items())]


def ece(linhas: list[dict]) -> dict[tuple[str, str], float]:
    """Erro de calibração esperado por (modelo, condição): média ponderada de |acerto − confiança| nas faixas."""
    saida: dict[tuple[str, str], list[float]] = {}
    for r in linhas:
        a = saida.setdefault((r["modelo"], r["condicao"]), [0.0, 0])
        a[0] += r["n"] * abs(r["acerto"] - r["conf_media"])
        a[1] += r["n"]
    return {k: v[0] / v[1] for k, v in saida.items()}


def main() -> None:
    linhas = collect()
    if not linhas:
        raise SystemExit("Nada a calibrar: precisa de hipóteses em data/derived/<video>/stt/ e do gabarito em data/gt/.")
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "calibracao.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS)
        w.writeheader()
        w.writerows(linhas)
    for (m, c), e in sorted(ece(linhas).items()):
        print(f"{m:8} {c:12} ECE={e:.3f}")
    print(f"-> {RESULTS / 'calibracao.csv'}")


if __name__ == "__main__":
    main()
