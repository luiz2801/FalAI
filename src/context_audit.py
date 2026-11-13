"""Auditoria das correções do context_fix (F10): das marcações do Studio para o results/raw.csv.

    uv run python src/context_audit.py consolidate

Lê results/correcoes_auditoria.csv (tela Correções do Studio: cada troca marcada, ouvindo o áudio, como certa, errada ou
"não sei") e grava, para cada (vídeo, modelo, condição) da RQ6 com marcações, a linha `precisao_correcoes` = certas ÷
(certas + erradas) no raw.csv. "Não sei" fica de fora. As trocas do pipeline não entram: não são execuções da matriz.
Rodar de novo atualiza as linhas (o relatório vale pela última).
"""

from __future__ import annotations

import csv
from collections import Counter

from run_experiments import RESULTS, append_rows, row

AUDIT_CSV = RESULTS / "correcoes_auditoria.csv"


def precisions(rows: list[dict]) -> dict[tuple[str, str, str], tuple[float, int, int]]:
    """{(vídeo, modelo, condição): (precisão, certas, erradas)}; só com ≥ 1 troca julgada e fora do pipeline."""
    cont: dict[tuple[str, str, str], Counter] = {}
    for r in rows:
        if r["auditoria"] in ("certa", "errada") and not r["condicao"].startswith("pipeline@"):
            cont.setdefault((r["video"], r["modelo"], r["condicao"]), Counter())[r["auditoria"]] += 1
    return {k: (c["certa"] / (c["certa"] + c["errada"]), c["certa"], c["errada"]) for k, c in sorted(cont.items())}


def consolidate() -> list[dict]:
    if not AUDIT_CSV.exists():
        raise SystemExit("results/correcoes_auditoria.csv não existe: audite as trocas na tela Correções do Studio.")
    with AUDIT_CSV.open(encoding="utf-8") as f:
        prec = precisions(list(csv.DictReader(f)))
    linhas = [row(v, "stt", m, c, 1, "precisao_correcoes", p, None, None) for (v, m, c), (p, _, _) in prec.items()]
    if linhas:
        append_rows(linhas)
    for (v, m, c), (p, certas, erradas) in prec.items():
        print(f"{v:10} {m:8} {c:12} precisão das correções = {p:.3f} ({certas} certas, {erradas} erradas)")
    print(f"{len(linhas)} linhas gravadas em results/raw.csv")
    return linhas


if __name__ == "__main__":
    import sys

    if sys.argv[1:] != ["consolidate"]:
        raise SystemExit(__doc__)
    consolidate()
