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


def align(ref: str, hyp: str) -> list[dict]:
    """Alinhamento palavra a palavra (o mesmo do WER) em trechos `equal`, `substitute`, `delete` e
    `insert`, cada um com as palavras normalizadas de `ref` e `hyp`. Base da análise de erros (regra 11)."""
    out = jiwer.process_words(_checar_ref(ref), norm(hyp))
    r, h = out.references[0], out.hypotheses[0]
    return [{"op": c.type, "ref": r[c.ref_start_idx:c.ref_end_idx], "hyp": h[c.hyp_start_idx:c.hyp_end_idx]}
            for c in out.alignments[0]]


def word_flags(ref: str, hyp_tokens: list[str]) -> list[bool]:
    """Para cada palavra da hipótese (já normalizada por `norm`), se ela está num trecho `equal` do alinhamento
    do WER com o gabarito (True) ou foi substituída ou inserida (False). Base da calibração da confiança."""
    if not hyp_tokens:
        return []
    out = jiwer.process_words(_checar_ref(ref), " ".join(hyp_tokens))
    flags = [False] * len(hyp_tokens)
    for c in out.alignments[0]:
        if c.type == "equal":
            for i in range(c.hyp_start_idx, c.hyp_end_idx):
                flags[i] = True
    return flags


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


def timed_term_accuracy(occurrences: list[tuple[float, str]], hyp_tokens: list[tuple[float, str]],
                        window_s: float = 3.0) -> dict:
    """Acerto de termos com gabarito **parcial** (F10, camada A): `occurrences` = [(instante, termo)] das
    ocorrências faladas, anotadas por quem ouviu o áudio, sem a transcrição completa. `hyp_tokens` =
    [(instante, palavra normalizada)] da hipótese. Uma ocorrência é acertada se a hipótese tem o termo (como
    sequência de palavras) a até `window_s` do instante anotado; cada ocorrência da hipótese vale uma
    anotação só (a mais próxima primeiro). Falsos positivos: ocorrências na hipótese de um termo do
    vocabulário anotado que não casaram com nenhuma anotação. Mesmo formato de `term_accuracy`."""
    palavras = [w for _, w in hyp_tokens]
    acertos = falsos = 0
    for termo in {norm(x) for _, x in occurrences if norm(x)}:
        alvo = termo.split()
        n = len(alvo)
        na_hip = [hyp_tokens[i][0] for i in range(len(palavras) - n + 1) if palavras[i:i + n] == alvo]
        livres = list(na_hip)
        for t in sorted(t for t, x in occurrences if norm(x) == termo):
            perto = min(livres, key=lambda h: abs(h - t), default=None)
            if perto is not None and abs(perto - t) <= window_s:
                livres.remove(perto)
                acertos += 1
        falsos += len(livres)
    total = len(occurrences)
    return {"terms_in_ref": total, "terms_hit": acertos, "accuracy": acertos / total if total else None,
            "false_positives": falsos}
