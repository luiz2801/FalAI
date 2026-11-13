"""F10 (FalAI Contexto), Fase 3: transcrição com o texto do slide como contexto.

    uv run python src/context_stt.py data/raw/aula01.mp4 --glossary data/derived/aula01/glossario.json --mode L
    uv run python src/context_stt.py data/raw/aula01.mp4 --glossary ... --mode L+F --offset-s 95 --dur-s 180

Modos (docs/plano_contexto_cruzado.md, seção 4): B0 sem contexto; G termos do vídeo todo numa chamada; P placebo
(termos de outro vídeo, nos mesmos intervalos do L); L o glossário do slide visível, por intervalo; L+F o L e a
correção `context_fix`; O oráculo (o `terms.txt` humano). O contexto entra como `hotwords`, que vale em toda
janela do Whisper (o `initial_prompt` só vale na primeira com condition_on_previous_text=False: D15).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

import context
import context_fix
from context import key

SR = 16000
MODES = ("B0", "G", "P", "L", "L+F", "O")


def make_asr(size: str, device: str = "auto", cpu_threads: int = 0, via: str = "hotwords"):
    """O ASR de verdade: `asr(audio, texto) -> (segmentos, tempo_s)`, com os mesmos parâmetros fixos do projeto
    (`stt.transcribe_array`). `texto` (ou None) vai como `hotwords` ou, com via="initial_prompt", como prompt."""
    import stt

    if via not in ("hotwords", "initial_prompt"):
        raise ValueError(f"via desconhecido: {via}")

    def asr(audio, texto):
        kw = {} if not texto else ({"hotwords": texto} if via == "hotwords" else {"initial_prompt": f"Aula: {texto}."})
        segs, _, tempo = stt.transcribe_array(audio, size, device, cpu_threads=cpu_threads, **kw)
        return segs, tempo

    return asr


def _recorta(segs: list[dict], offset: float, t0: float, t1: float, ultimo: bool) -> list[dict]:
    """Segmentos de uma chamada (tempos relativos à fatia que começa em `offset`), ficando só com as PALAVRAS
    cujo ponto médio cai em [t0, t1) e levando os tempos para o relógio do áudio inteiro. Por palavra e não por
    segmento: dois recortes vizinhos segmentam diferente, e a regra por segmento duplicaria ou perderia fala na
    fronteira. O último intervalo fica com tudo o que passa do fim."""
    saida = []
    for s in segs:
        palavras = s.get("words")
        if not palavras:  # sem tempos por palavra: ponto médio do segmento
            meio = offset + (s["start"] + s["end"]) / 2
            if t0 <= meio < t1 or (ultimo and meio >= t1):
                saida.append({**s, "start": round(offset + s["start"], 3), "end": round(offset + s["end"], 3)})
            continue
        mantidas = [w for w in palavras if t0 <= offset + (w["start"] + w["end"]) / 2 < t1
                    or (ultimo and offset + (w["start"] + w["end"]) / 2 >= t1)]
        if not mantidas:
            continue
        deslocadas = [{**w, "start": round(offset + w["start"], 3), "end": round(offset + w["end"], 3)} for w in mantidas]
        saida.append({**s, "start": deslocadas[0]["start"], "end": deslocadas[-1]["end"],
                      "text": s["text"] if len(mantidas) == len(palavras) else " ".join(w["word"] for w in deslocadas),
                      "words": deslocadas})
    return saida


def decode_intervals(audio: np.ndarray, intervals: list[dict], asr, margin_s: float = 2.0):
    """Decodifica cada intervalo do plano com os termos dele. Devolve (segmentos juntos e ordenados, tempo total
    de inferência, nº de chamadas). Cada chamada recebe a fatia [t0 − margem, t1 + margem], para que as palavras
    da borda saiam inteiras (armadilha 21); a palavra fica com o intervalo que contém o seu ponto médio."""
    dur = len(audio) / SR
    todas, proc = [], 0.0
    for k, iv in enumerate(intervals):
        a, b = max(0.0, iv["t0"] - margin_s), min(dur, iv["t1"] + margin_s)
        segs, tempo = asr(audio[int(a * SR):int(b * SR)], context.hotwords(iv["terms"]) or None)
        proc += tempo
        todas.extend(_recorta(segs, a, iv["t0"], iv["t1"], ultimo=k == len(intervals) - 1))
    todas.sort(key=lambda s: (s["start"], s["end"]))
    for i, s in enumerate(todas, 1):
        s["id"] = i
    return todas, proc, len(intervals)


def drop_leaks(segs: list[dict], plano: list[dict]) -> tuple[list[dict], list[dict]]:
    """O Whisper às vezes repete o glossário em silêncio. Sai o segmento com ≥ 3 palavras, todas do vocabulário do
    glossário daquele instante, e com cara de não-fala (no_speech_prob > 0,5 ou avg_logprob < −1). Quem diz de
    fato "Conselho Monetário Nacional" tem fala de verdade e fica. Devolve (mantidos, removidos com reason)."""
    mantidos, removidos = [], []
    for s in segs:
        meio = (s["start"] + s["end"]) / 2
        termos = next((p["terms"] for p in plano if p["t0"] <= meio < p["t1"]), [])
        vocab = {w for t in termos for w in key(t["text"]).split()}
        tokens = key(s["text"]).split()
        if (len(tokens) >= 3 and all(t in vocab for t in tokens)
                and (s.get("no_speech_prob", 0) > 0.5 or s.get("avg_logprob", 0) < -1.0)):
            removidos.append({**s, "reason": "vazamento"})
        else:
            mantidos.append(s)
    for i, s in enumerate(mantidos, 1):
        s["id"] = i
    return mantidos, removidos


def _plano_unico(dur: float, termos: list[dict]) -> list[dict]:
    return [{"t0": 0.0, "t1": round(dur, 3), "slide": None, "terms": termos}]


def transcribe_context(audio: np.ndarray, mode: str, asr, glossary: list[dict] | None = None,
                       oracle_terms: list[str] | None = None, placebo_glossary: list[dict] | None = None,
                       plan_kw: dict | None = None, margin_s: float = 2.0, max_terms: int = 30, tau: float = 0.5,
                       s_min: float = 0.85, seed: int = 0, clean: bool = True) -> dict:
    """Transcreve `audio` (float32, 16 kHz) no `mode`. `glossary` está no relógio do áudio (use
    `context.shift_glossary` para um trecho). Devolve {"segs", "tempo_s", "n_calls", "plan", "removed",
    "corrections", "fix_summary", "mode"}:

      B0   uma chamada, sem termos.            G  uma chamada, com os `max_terms` termos de maior peso do vídeo.
      O    uma chamada, com `oracle_terms`.     L  uma chamada por intervalo do plano, com os termos do slide.
      P    como o L, mas os termos de cada intervalo são sorteados de `placebo_glossary` (outro vídeo), do mesmo
           tipo e fora dos termos do trecho: o controle de "qualquer prompt".
      L+F  o L e, depois, `context_fix.fix_segments`.

    `tempo_s` é só a inferência (soma das chamadas). Com clean=True, aplica `stt.clean_segments` ao final, como
    o `stt.transcribe` (e `drop_leaks`, nos modos com termos)."""
    import stt

    if mode not in MODES:
        raise ValueError(f"modo desconhecido: {mode}")
    dur = len(audio) / SR
    plan_kw = {"max_terms": max_terms, **(plan_kw or {})}
    if mode in ("G", "L", "L+F", "P") and glossary is None:
        raise ValueError(f"o modo {mode} precisa do glossário")
    if mode == "B0":
        plano = _plano_unico(dur, [])
    elif mode == "G":
        plano = _plano_unico(dur, context.global_terms(glossary, max_terms))
    elif mode == "O":
        if not oracle_terms:
            raise ValueError("o modo O precisa dos termos do oráculo (terms.txt)")
        plano = _plano_unico(dur, [{"text": t, "kind": "termo", "weight": round(1 - i / 1000, 4), "pron": []}
                                   for i, t in enumerate(oracle_terms)])
    else:
        plano = context.plan_intervals(glossary, dur, **plan_kw)
        if mode == "P":
            if placebo_glossary is None:
                raise ValueError("o modo P precisa de um glossário de outro vídeo")
            exclude = {key(t["text"]) for g in glossary for t in g["terms"]} | {key(t) for t in oracle_terms or []}
            plano = [{**iv, "terms": context.placebo_like(iv["terms"], placebo_glossary, exclude, seed + i)}
                     for i, iv in enumerate(plano)]
    segs, proc, n_calls = decode_intervals(audio, plano, asr, margin_s)
    removed: list[dict] = []
    if clean:
        segs, removed = stt.clean_segments(segs)
        if mode != "B0":
            segs, vazados = drop_leaks(segs, plano)
            removed += vazados
    correcoes, resumo = [], {}
    if mode == "L+F":
        segs, correcoes, resumo = context_fix.fix_segments(segs, plano, tau, s_min)
    return {"segs": segs, "tempo_s": proc, "n_calls": n_calls, "plan": plano, "removed": removed,
            "corrections": correcoes, "fix_summary": resumo, "mode": mode}


def hyp_text(segs: list[dict]) -> str:
    return " ".join(s["text"] for s in segs)


def hyp_tokens(segs: list[dict]) -> list[tuple[float, str]]:
    """[(instante, palavra normalizada)] da hipótese, para `metrics.timed_term_accuracy`. Usa o início de cada
    palavra; segmento sem tempos por palavra espalha as palavras de modo uniforme entre o início e o fim."""
    from metrics import norm

    saida = []
    for s in segs:
        if s.get("words"):
            saida += [(w["start"], tok) for w in s["words"] for tok in norm(w["word"]).split()]
        else:
            toks = norm(s["text"]).split()
            saida += [(s["start"] + (s["end"] - s["start"]) * i / max(len(toks), 1), tok) for i, tok in enumerate(toks)]
    return saida


def changed_fraction(base: list[dict], novo: list[dict]) -> float:
    """`taxa_alteracao`: palavras do texto novo que não estão (em ordem) no texto base ÷ palavras do base.
    Diagnóstico, não é o WER: não usa gabarito."""
    import jiwer

    from metrics import norm

    b = norm(hyp_text(base))
    if not b:
        return 0.0
    return float(jiwer.wer(b, norm(hyp_text(novo))))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("media")
    ap.add_argument("--mode", default="L", choices=MODES)
    ap.add_argument("--glossary", help="glossario.json (tempo do vídeo)")
    ap.add_argument("--oracle-terms", help="terms.txt (modo O e exclusão do placebo)")
    ap.add_argument("--placebo-glossary", help="glossario.json de OUTRO vídeo (modo P)")
    ap.add_argument("--model", default="small", choices=["tiny", "base", "small", "medium", "large-v3"])
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--via", default="hotwords", choices=["hotwords", "initial_prompt"])
    ap.add_argument("--offset-s", type=float, default=0.0, help="início do trecho no vídeo (s)")
    ap.add_argument("--dur-s", type=float, help="duração do trecho (s); padrão: até o fim")
    ap.add_argument("--margin-s", type=float, default=2.0)
    ap.add_argument("--lead-s", type=float, default=2.0)
    ap.add_argument("--lag-s", type=float, default=1.0)
    ap.add_argument("--no-carry", action="store_true", help="intervalo sem slide fica sem termos")
    ap.add_argument("--max-terms", type=int, default=30)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--s-min", type=float, default=0.85)
    ap.add_argument("--no-clean", action="store_true")
    ap.add_argument("--out", help="pasta de saída (padrão: data/derived/<mídia>/ctx)")
    a = ap.parse_args()

    import stt
    from extract import load_audio

    audio = load_audio(a.media)
    ini = int(a.offset_s * SR)
    audio = audio[ini:ini + int(a.dur_s * SR)] if a.dur_s else audio[ini:]
    glossario = None
    if a.glossary:
        glossario = context.shift_glossary(context.load_glossary(a.glossary), a.offset_s, len(audio) / SR)
    oraculo = stt.read_terms(a.oracle_terms) if a.oracle_terms else None
    placebo = context.load_glossary(a.placebo_glossary) if a.placebo_glossary else None
    r = transcribe_context(
        audio, a.mode, make_asr(a.model, a.device, via=a.via), glossario, oraculo, placebo,
        plan_kw={"lead_s": a.lead_s, "lag_s": a.lag_s, "carry": not a.no_carry}, margin_s=a.margin_s,
        max_terms=a.max_terms, tau=a.tau, s_min=a.s_min, clean=not a.no_clean)
    root = Path(__file__).resolve().parent.parent
    out = Path(a.out) if a.out else root / "data" / "derived" / Path(a.media).stem / "ctx"
    stem = f"{Path(a.media).stem}.{a.model}.{re.sub(r'[^A-Za-z0-9]+', '_', a.mode)}"
    dur = len(audio) / SR
    meta = {"media": str(a.media), "model": a.model, "mode": a.mode, "via": a.via, "offset_s": a.offset_s,
            "duration_s": dur, "time_s": r["tempo_s"], "calls": r["n_calls"], "removed": r["removed"],
            "fix": r["fix_summary"], "plan": r["plan"]}
    stt.save_outputs(r["segs"], out, stem, meta)
    if r["corrections"]:
        (out / f"{stem}.correcoes.jsonl").write_text(
            "\n".join(json.dumps(c, ensure_ascii=False) for c in r["corrections"]) + "\n", encoding="utf-8")
    print(f"{a.mode}: {dur:.0f} s de áudio em {r['tempo_s']:.1f} s (RTF {r['tempo_s'] / dur:.3f}), "
          f"{r['n_calls']} chamadas, {len(r['segs'])} segmentos, {len(r['corrections'])} correções -> {out}")


if __name__ == "__main__":
    main()
