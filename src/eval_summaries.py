"""F9: avaliação dos resumos às cegas -> linhas tarefa=resumo no results/raw.csv.

  python src/eval_summaries.py prepare      # embaralha results/summaries/ em results/eval/blind/
  python src/eval_summaries.py consolidate  # lê as planilhas preenchidas e grava no raw.csv

Planilhas em results/eval/ (a pasta blind/ é a única que vai para os avaliadores):
  key.csv          id -> vídeo, modelo STT, LLM, repetição (NÃO enviar)
  scores.csv       uma linha por (id, avaliador): clareza, utilidade, correcao (1 a 5)
  fidelity.csv     por id: afirmações no resumo e quantas têm respaldo no gabarito/vídeo
  topics.csv       cobertura e timestamps (±30 s) estimados contra data/gt/<video>.topics.txt;
                   corrija nas colunas *_manual quando a estimativa errar

data/gt/<video>.topics.txt (tópicos reais, anotados à mão): uma linha "[hh:mm:ss] Título".
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import sys
import unicodedata
from pathlib import Path

from run_experiments import GT, RESULTS, append_rows, row
from summarize import extract_topics, to_seconds

EVAL = RESULTS / "eval"
TOLERANCE_S = 30
STOP = set(["a", "o", "e", "de", "da", "do", "das", "dos", "em", "no", "na", "nos", "nas", "um", "uma", "para", "por", "com", "sem", "que", "ao", "aos", "as", "os", "se", "sua", "seu", "entre", "sobre", "como", "mais"])


def _words(t: str) -> set[str]:
    t = "".join(c for c in unicodedata.normalize("NFKD", t.lower()) if not unicodedata.combining(c))
    return {w for w in re.findall(r"\w+", t) if w not in STOP and len(w) > 1}


def similarity(a: str, b: str) -> float:
    """Jaccard das palavras de conteúdo dos títulos."""
    pa, pb = _words(a), _words(b)
    return len(pa & pb) / len(pa | pb) if pa and pb else 0.0


def real_topics(video: str) -> list[tuple[float, str]]:
    arq = GT / f"{video}.topics.txt"
    if not arq.exists():
        return []
    out = []
    for linha in arq.read_text(encoding="utf-8").splitlines():
        if m := re.match(r"\s*[-*]?\s*\[(\d{1,2}:\d{2}(?::\d{2})?)\]\s*(.+)", linha):
            out.append((to_seconds(m.group(1)), m.group(2).strip()))
    return out


def compare_topics(report: str, real: list[tuple[float, str]], threshold: float = 0.4) -> dict:
    """Estimativa automática: casa cada tópico real com o gerado de título mais parecido."""
    gerados = extract_topics(report)
    cobertos = no_tempo = 0
    for t_real, titulo in real:
        melhor = max(gerados, key=lambda g: similarity(titulo, g[1]), default=None)
        if melhor and similarity(titulo, melhor[1]) >= threshold:
            cobertos += 1
            no_tempo += abs(melhor[0] - t_real) <= TOLERANCE_S
    return {"real_topics": len(real), "generated_topics": len(gerados), "covered_auto": cobertos,
            "on_time_auto": no_tempo, "covered_manual": "", "on_time_manual": ""}


def _parse_name(md: Path) -> dict:
    """<video>__<modelo>__<llm>__r<k>.md (run_experiments.py); a chave exata do LLM vem do .json."""
    v, modelo, _, rep = md.stem.split("__")
    llm = json.loads(md.with_suffix(".json").read_text(encoding="utf-8"))["condicao"]
    return {"video": v, "model": modelo, "llm": llm, "rep": int(rep.lstrip("r"))}


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def prepare(a) -> None:
    resumos = sorted((RESULTS / "summaries").glob("*.md"))
    if not resumos:
        sys.exit("Nenhum resumo em results/summaries/ (rode run_experiments.py --summaries).")
    if (EVAL / "key.csv").exists() and not a.redo:
        sys.exit("results/eval/key.csv já existe (use --redo para recriar e perder as planilhas).")
    random.Random(a.seed).shuffle(resumos)
    blind = EVAL / "blind"
    shutil.rmtree(blind, ignore_errors=True)
    blind.mkdir(parents=True)
    key, topics = [], []
    for i, md in enumerate(resumos, 1):
        rid = f"R{i:03}"
        shutil.copy(md, blind / f"{rid}.md")
        meta = _parse_name(md)
        key.append({"id": rid, "file": md.name, **meta})
        topics.append({"id": rid, **compare_topics(md.read_text(encoding="utf-8"), real_topics(meta["video"]))})
    _write_csv(EVAL / "key.csv", key)
    _write_csv(EVAL / "topics.csv", topics)
    _write_csv(EVAL / "scores.csv", [{"id": k["id"], "avaliador": "", "clareza": "", "utilidade": "",
                                      "correcao": ""} for k in key])
    _write_csv(EVAL / "fidelity.csv", [{"id": k["id"], "afirmacoes_total": "", "afirmacoes_com_respaldo": "",
                                        "obs": ""} for k in key])
    shutil.copy(EVAL / "scores.csv", blind / "scores.csv")
    (blind / "INSTRUCOES.txt").write_text(
        "Dê nota de 1 (péssimo) a 5 (excelente) a cada resumo em scores.csv:\n"
        "  clareza: é organizado e fácil de entender?\n"
        "  utilidade: ajudaria a revisar a aula?\n"
        "  correcao: o conteúdo parece tecnicamente correto?\n"
        "Uma linha por resumo, com o seu nome em 'avaliador'.\n", encoding="utf-8")
    sem = sorted({k["video"] for k in key if not real_topics(k["video"])})
    print(f"{len(key)} resumos em {blind}." + (f" Sem data/gt/<video>.topics.txt: {sem}" if sem else ""))


def _num(x: str) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def consolidate(a) -> None:
    key = {k["id"]: k for k in _read_csv(EVAL / "key.csv")}
    linhas = []

    def add(rid: str, metrica: str, valor: float | None):
        if valor is None:
            return
        k = key[rid]
        linhas.append(row(k["video"], "resumo", k["model"], k["llm"], k["rep"], metrica, valor, None, None))

    notas: dict[str, list[float]] = {}
    for s in _read_csv(EVAL / "scores.csv"):
        vals = [_num(s[c]) for c in ("clareza", "utilidade", "correcao")]
        if None not in vals:
            notas.setdefault(s["id"], []).append(sum(vals) / 3)
    for rid, v in notas.items():
        add(rid, "nota_media", sum(v) / len(v))
    for f in _read_csv(EVAL / "fidelity.csv"):
        tot, ok = _num(f["afirmacoes_total"]), _num(f["afirmacoes_com_respaldo"])
        if tot:
            add(f["id"], "fidelidade", ok / tot if ok is not None else None)
    for t in _read_csv(EVAL / "topics.csv"):
        reais = _num(t["real_topics"])
        cob = _num(t["covered_manual"]) if t["covered_manual"] != "" else _num(t["covered_auto"])
        ok = _num(t["on_time_manual"]) if t["on_time_manual"] != "" else _num(t["on_time_auto"])
        if reais:
            add(t["id"], "cobertura", cob / reais)
            if cob:
                add(t["id"], "precisao_timestamps", ok / cob)
    append_rows(linhas)
    print(f"{len(linhas)} linhas tarefa=resumo gravadas em results/raw.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--redo", action="store_true")
    sub.add_parser("consolidate")
    a = ap.parse_args()
    prepare(a) if a.cmd == "prepare" else consolidate(a)


if __name__ == "__main__":
    main()
