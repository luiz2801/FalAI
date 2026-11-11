"""Métricas do projeto. Funções puras: nenhuma lê arquivo (guia.md, seção 4.5)."""

from __future__ import annotations

import re
from collections import Counter

import jiwer


def norm(t: str) -> str:
    """Única normalização do projeto: minúsculas, pontuação trocada por espaço, acentos
    mantidos, espaços colapsados. Mudar esta função invalida o results/raw.csv (major)."""
    t = t.lower()
    t = re.sub(r"[^\w\s]", " ", t)
    t = t.replace("_", " ")
    return re.sub(r"\s+", " ", t).strip()


def _checar_ref(ref: str) -> str:
    r = norm(ref)
    if not r:
        raise ValueError("referência vazia: WER/CER não tem significado (guia.md, armadilha 13)")
    return r


def wer(ref: str, hyp: str) -> float:
    return jiwer.wer(_checar_ref(ref), norm(hyp))


def cer(ref: str, hyp: str) -> float:
    return jiwer.cer(_checar_ref(ref), norm(hyp))


def wer_detail(ref: str, hyp: str) -> dict:
    """WER com substituições, remoções e inserções (para a análise de erros, regra 11)."""
    out = jiwer.process_words(_checar_ref(ref), norm(hyp))
    return {"wer": out.wer, "substitutions": out.substitutions, "deletions": out.deletions,
            "insertions": out.insertions, "ref_words": sum(len(r) for r in out.references)}


def word_pr(ref: str, hyp: str) -> tuple[float, float]:
    """Precisão e recall de palavras por multiconjunto (ignora a ordem; bom para slides)."""
    r, h = Counter(norm(ref).split()), Counter(norm(hyp).split())
    comuns = sum((r & h).values())
    precisao = comuns / sum(h.values()) if h else 0.0
    recall = comuns / sum(r.values()) if r else 0.0
    return precisao, recall


def rtf(tempo_proc_s: float, duracao_s: float) -> float:
    """Real-time factor: tempo de processamento / duração da mídia (< 1 = mais rápido que o real)."""
    if duracao_s <= 0:
        raise ValueError("duração da mídia deve ser positiva")
    return tempo_proc_s / duracao_s


def _contar(termo: str, texto: str) -> int:
    return len(re.findall(rf"(?<!\w){re.escape(termo)}(?!\w)", texto))


def term_accuracy(ref: str, hyp: str, terms: list[str]) -> dict:
    """Acerto de termos técnicos (I5): ocorrências do termo na hipótese, limitadas às da
    referência, sobre o total na referência. Falsos positivos: ocorrências a mais na hipótese."""
    r, h = norm(ref), norm(hyp)
    total = acertos = falsos = 0
    for termo in terms:
        tn = norm(termo)
        if not tn:
            continue
        nr, nh = _contar(tn, r), _contar(tn, h)
        total += nr
        acertos += min(nr, nh)
        falsos += max(0, nh - nr)
    return {"terms_in_ref": total, "terms_hit": acertos,
            "accuracy": acertos / total if total else None, "false_positives": falsos}
