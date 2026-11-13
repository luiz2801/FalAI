"""Avalia o extrator de slides contra um gabarito de tempo anotado à mão (F10, Fase 1; sem modelo).

    uv run python src/eval_slides.py data/derived/aula01/frames/frames.json data/gt/aula01.slides_t.json

O gabarito (`data/gt/<video>.slides_t.json`) é uma lista `[{"t": 25.0, "t_complete": 275.0, "t_end": 311.0, "titulo": "..."}]`
(`t` = quando o slide aparece, `t_complete` = quando fica completo, ambos em segundos, anotados vendo o vídeo).
Cada slide detectado casa com no máximo um do gabarito, pelo início (`t`) dentro de `tol_s`, o mais próximo primeiro.
Métricas: precisão e recall do início; `extras` (detectados sem par: câmera tomada por slide, ou o mesmo slide partido em
dois); `perdidos` (do gabarito sem par); e `completos`, a fração dos casados cujo `t_complete` também bate com o do
gabarito: o quadro guardado é o completo, e não o de um estágio intermediário.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def evaluate(pred: list[dict], gt: list[dict], tol_s: float = 2.0) -> dict:
    """Casa `pred` (frames.json) com `gt` pelo início. Devolve contagens, precisão, recall, completos e os pares."""
    candidatos = sorted(((abs(p["t"] - g["t"]), i, j) for i, p in enumerate(pred) for j, g in enumerate(gt)
                         if abs(p["t"] - g["t"]) <= tol_s))
    usados_p: set[int] = set()
    usados_g: set[int] = set()
    pares = []
    for d, i, j in candidatos:
        if i in usados_p or j in usados_g:
            continue
        usados_p.add(i)
        usados_g.add(j)
        ok = None
        if "t_complete" in gt[j] and "t_complete" in pred[i]:
            ok = abs(pred[i]["t_complete"] - gt[j]["t_complete"]) <= tol_s
        pares.append({"pred": i, "gt": j, "dt_inicio": round(pred[i]["t"] - gt[j]["t"], 3), "completo": ok})
    n = len(pares)
    julgados = [p for p in pares if p["completo"] is not None]
    return {"detectados": len(pred), "gabarito": len(gt), "casados": n,
            "precisao": n / len(pred) if pred else None, "recall": n / len(gt) if gt else None,
            "extras": [i for i in range(len(pred)) if i not in usados_p],
            "perdidos": [j for j in range(len(gt)) if j not in usados_g],
            "completos": sum(p["completo"] for p in julgados) / len(julgados) if julgados else None,
            "pares": sorted(pares, key=lambda p: p["gt"]), "tol_s": tol_s}


def main() -> None:
    if len(sys.argv) not in (3, 4):
        raise SystemExit(__doc__)
    pred = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    gt = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    tol = float(sys.argv[3]) if len(sys.argv) == 4 else 2.0
    r = evaluate(pred, gt, tol)
    f = lambda v: "—" if v is None else f"{v:.3f}"
    print(f"detectados {r['detectados']}, gabarito {r['gabarito']}, casados {r['casados']} (tolerância {tol} s)")
    print(f"precisão {f(r['precisao'])}  recall {f(r['recall'])}  completos {f(r['completos'])}")
    if r["extras"]:
        print("extras (detectados sem par):", ", ".join(f"{pred[i]['file']}@{pred[i]['t']:.1f}s" for i in r["extras"]))
    if r["perdidos"]:
        print("perdidos (do gabarito):", ", ".join(f"{gt[j].get('titulo', j)}@{gt[j]['t']:.1f}s" for j in r["perdidos"]))
    for p in r["pares"]:
        print(f"  gabarito {p['gt']} <- {pred[p['pred']]['file']}  Δinício {p['dt_inicio']:+.1f} s  completo {p['completo']}")


if __name__ == "__main__":
    main()
