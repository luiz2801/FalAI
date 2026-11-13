"""Transcrição (STT) com faster-whisper e escrita de legendas (F1, guia.md seção 4.2).

Uso direto:
    python src/stt.py data/raw/aula01.mp4 --model small [--vocab data/gt/terms.txt] [--out data/derived/aula01]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import textwrap
import time
from pathlib import Path

from extract import load_audio

MODELS = ["tiny", "base", "small", "medium", "large-v3"]
# Parâmetros fixos da comparação (CLAUDE.md, decisão 5)
LANGUAGE, BEAM_SIZE, VAD_FILTER = "pt", 5, True

# Frases que o Whisper "alucina" em trechos silenciosos (vêm das legendas do treino).
HALLUCINATIONS = [
    r"amara\.?\s*org", r"legendas? (pela|por|da|de) comunidade", r"legenda[sd]?o? por",
    r"tradu[cç][aã]o e legendas", r"legendas?\s*:", r"adriana zanotto", r"inscreva-?se( no canal)?",
    r"obrigad[oa] por assistir", r"deixe (o )?seu like", r"subt[ií]tulos realizados", r"www\.\w+\.(com|org|net)",
]
_RE_HALLU = re.compile("|".join(HALLUCINATIONS), re.IGNORECASE)
# n-grama (>= 3 caracteres) repetido 4 ou mais vezes seguidas -> uma vez só
_RE_LOOP = re.compile(r"\b(.{3,}?)(?:[\s,.]+\1\b){3,}", re.IGNORECASE)
# initial_prompt do Whisper: ~224 tokens; ~600 caracteres em PT-BR é uma margem segura
MAX_PROMPT_CHARS = 600

_models: dict = {}


# ------------------------------------------------------------------ formatos de tempo
def fmt(t: float) -> str:
    """Segundos -> 'HH:MM:SS.mmm' (WebVTT)."""
    ms = round(t * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02}.{ms:03}"


def fmt_srt(t: float) -> str:
    """Segundos -> 'HH:MM:SS,mmm' (SRT)."""
    return fmt(t).replace(".", ",")


# ------------------------------------------------------------------ modelo
_avisou_cuda = False


def _cuda_ok() -> bool:
    """As bibliotecas CUDA que o ctranslate2 carrega na inferência (cuBLAS 12) estão no sistema? O ctranslate2 só
    confere se há GPU; sem o cuBLAS, a inferência falha no meio com `libcublas.so.12 is not found`."""
    import ctypes

    try:
        ctypes.CDLL("libcublas.so.12")
        return True
    except OSError:
        return False


def resolve_device(device: str = "auto") -> tuple[str, str]:
    """(device, compute_type). 'auto' usa CUDA se houver GPU **e** o cuBLAS; senão CPU int8 (decisão 8), com um
    aviso. 'cuda' explícito não é checado: quem pede CUDA recebe o erro do ctranslate2."""
    global _avisou_cuda
    if device == "auto":
        try:
            import ctranslate2

            tem_gpu = ctranslate2.get_cuda_device_count() > 0
        except Exception:  # noqa: BLE001  # ctranslate2 sem suporte a CUDA, ou não instalado
            tem_gpu = False
        device = "cuda" if tem_gpu and _cuda_ok() else "cpu"
        if tem_gpu and device == "cpu" and not _avisou_cuda:
            _avisou_cuda = True
            print("AVISO: GPU CUDA detectada, mas libcublas.so.12 não foi encontrada: usando CPU/int8 "
                  "(instale o CUDA 12 ou use o contêiner; docs/decisoes.md, D17).")
    return device, ("float16" if device == "cuda" else "int8")


def load_model(size: str, device: str = "auto", cpu_threads: int = 0):
    """Carrega (com cache) o WhisperModel. Devolve (modelo, segundos de carga; 0 se em cache)."""
    from faster_whisper import WhisperModel

    dev, compute = resolve_device(device)
    chave = (size, dev, compute, cpu_threads)
    if chave in _models:
        return _models[chave], 0.0
    t0 = time.perf_counter()
    _models[chave] = WhisperModel(size, device=dev, compute_type=compute, cpu_threads=cpu_threads)
    return _models[chave], time.perf_counter() - t0


def unload_models() -> None:
    import gc

    _models.clear()
    gc.collect()


def build_prompt(terms: list[str], header: str = "Aula:") -> str:
    """I5: initial_prompt com o vocabulário da disciplina, dentro do limite de tamanho."""
    prompt = header
    for termo in terms:
        candidato = f"{prompt} {termo},"
        if len(candidato) > MAX_PROMPT_CHARS:
            break
        prompt = candidato
    return prompt.rstrip(",") + "."


def read_terms(path: str | Path) -> list[str]:
    """Um termo por linha; '#' inicia comentário."""
    termos = []
    for linha in Path(path).read_text(encoding="utf-8").splitlines():
        linha = linha.split("#", 1)[0].strip()
        if linha:
            termos.append(linha)
    return termos


# ------------------------------------------------------------------ limpeza
def _is_hallucination(texto: str) -> bool:
    t = texto.strip()
    if not t:
        return True
    m = _RE_HALLU.search(t)
    return bool(m) and len(t) < 2.5 * len(m.group(0)) + 20


def clean_segments(segs: list[dict]) -> tuple[list[dict], list[dict]]:
    """Remove alucinações conhecidas e repetições. Devolve (mantidos, removidos com motivo)."""
    mantidos, removidos, anterior = [], [], None
    for s in segs:
        texto = _RE_LOOP.sub(r"\1", s["text"].strip())
        if _is_hallucination(texto):
            removidos.append({**s, "reason": "hallucination"})
            continue
        if anterior is not None and texto.lower() == anterior.lower():
            removidos.append({**s, "reason": "repetition"})
            continue
        if texto != s["text"].strip():
            s = {**s, "text": texto, "words": None}  # as palavras não batem mais com o texto
        mantidos.append(s)
        anterior = texto
    for i, s in enumerate(mantidos, 1):
        s["id"] = i
    return mantidos, removidos


# ------------------------------------------------------------------ transcrição
def transcribe_array(
    audio,
    size: str = "small",
    device: str = "auto",
    initial_prompt: str | None = None,
    hotwords: str | None = None,
    word_timestamps: bool = True,
    cpu_threads: int = 0,
    progress=None,
) -> tuple[list[dict], float, float]:
    """Transcreve um array float32 (16 kHz, mono) já em memória, **sem** limpar. Devolve (segs, duracao_s,
    tempo_s). É o núcleo de `transcribe` e o que `context_stt` chama a cada trecho.

    - tempo_s cronometra só a inferência: o gerador é consumido dentro do cronômetro; a carga do modelo e a
      decodificação do áudio ficam fora (armadilhas 7.1.1 e 7.1.3).
    - language, beam_size e vad_filter são fixos (decisão 5); condition_on_previous_text=False reduz loops.
    - `hotwords` vale em **toda** janela de ~30 s; o `initial_prompt` só na primeira, porque com
      condition_on_previous_text=False o faster-whisper zera o prompt depois dela (docs/decisoes.md, D15).
    """
    modelo, _ = load_model(size, device, cpu_threads)
    t0 = time.perf_counter()
    gerador, info = modelo.transcribe(
        audio, language=LANGUAGE, beam_size=BEAM_SIZE, vad_filter=VAD_FILTER,
        condition_on_previous_text=False, initial_prompt=initial_prompt, hotwords=hotwords,
        word_timestamps=word_timestamps,
    )
    segs = []
    for s in gerador:
        segs.append({
            "id": s.id, "start": round(s.start, 3), "end": round(s.end, 3), "text": s.text.strip(),
            "avg_logprob": round(s.avg_logprob, 4), "no_speech_prob": round(s.no_speech_prob, 4),
            "words": [{"start": round(w.start, 3), "end": round(w.end, 3), "word": w.word.strip(),
                       "probability": round(w.probability, 4)} for w in (s.words or [])] or None,
        })
        if progress and info.duration:
            progress(min(s.end / info.duration, 1.0))
    return segs, info.duration, time.perf_counter() - t0


def transcribe(
    media: str | Path,
    size: str = "small",
    device: str = "auto",
    initial_prompt: str | None = None,
    clean: bool = True,
    word_timestamps: bool = True,
    cpu_threads: int = 0,
    progress=None,
    hotwords: str | None = None,
) -> tuple[list[dict], float, float]:
    """Transcreve uma mídia. Devolve (segs, duracao_s, tempo_s). Decodifica o áudio (fora do cronômetro) e
    chama `transcribe_array`.

    - Com clean=True, segs não tem as alucinações conhecidas; os removidos ficam em
      transcribe.last_removed.
    """
    audio = load_audio(media)
    segs, duracao, tempo_s = transcribe_array(audio, size, device, initial_prompt, hotwords, word_timestamps,
                                              cpu_threads, progress)
    transcribe.last_removed = []
    if clean:
        segs, transcribe.last_removed = clean_segments(segs)
    return segs, duracao, tempo_s


transcribe.last_removed = []


# ------------------------------------------------------------------ legendas e texto
def _fits(texto: str, max_chars: int, max_lines: int) -> bool:
    return len(textwrap.wrap(texto, max_chars)) <= max_lines


def _split_balanced(words: list[dict], n: int) -> list[list[dict]]:
    total = sum(len(w["word"]) + 1 for w in words)
    partes, atual, acc = [], [], 0
    for w in words:
        atual.append(w)
        acc += len(w["word"]) + 1
        if len(partes) < n - 1 and acc >= total * (len(partes) + 1) / n:
            partes.append(atual)
            atual = []
    if atual:
        partes.append(atual)
    return partes


def subtitle_blocks(segs: list[dict], max_chars: int | None = 42, max_lines: int = 2, max_dur: float = 7.0):
    """Blocos (início, fim, texto). max_chars=None -> um bloco por segmento.

    Com max_chars, cada segmento é repartido em blocos equilibrados de <= max_lines linhas de
    <= max_chars caracteres e <= max_dur s (sem sobras de uma palavra só)."""
    for s in segs:
        if max_chars is None:
            yield s["start"], s["end"], s["text"]
            continue
        words = s.get("words")
        if words:
            texto = " ".join(w["word"] for w in words)
            dur = words[-1]["end"] - words[0]["start"]
            n = max(math.ceil(len(texto) / (max_chars * max_lines * 0.9)), math.ceil(dur / max_dur), 1)
            while True:
                partes = _split_balanced(words, n)
                if n >= len(words) or all(
                        _fits(" ".join(w["word"] for w in p), max_chars, max_lines) for p in partes):
                    break
                n += 1
            for p in partes:
                yield p[0]["start"], p[-1]["end"], " ".join(w["word"] for w in p)
        else:  # sem tempo por palavra: divide texto e tempo proporcionalmente
            pedacos = textwrap.wrap(s["text"], max_chars * max_lines) or [s["text"]]
            total, inicio = sum(len(p) for p in pedacos), s["start"]
            for p in pedacos:
                fim = inicio + (s["end"] - s["start"]) * len(p) / total
                yield inicio, fim, p
                inicio = fim


def _subtitle_text(segs, header: str, tfmt, numbered: bool, max_chars: int | None) -> str:
    blocos = [header] if header else []
    for i, (ini, fim, texto) in enumerate(subtitle_blocks(segs, max_chars), 1):
        corpo = "\n".join(textwrap.wrap(texto, max_chars)) if max_chars else texto
        linhas = ([str(i)] if numbered else []) + [f"{tfmt(ini)} --> {tfmt(fim)}", corpo]
        blocos.append("\n".join(linhas))
    return "\n\n".join(blocos) + "\n"


def write_vtt(segs: list[dict], path: str | Path, max_chars: int | None = 42) -> Path:
    """WEBVTT, linha em branco e blocos 'início --> fim' + texto (contrato 5.4)."""
    Path(path).write_text(_subtitle_text(segs, "WEBVTT", fmt, False, max_chars), encoding="utf-8")
    return Path(path)


def write_srt(segs: list[dict], path: str | Path, max_chars: int | None = 42) -> Path:
    """SRT numerado. Com max_chars=None (um bloco por segmento) é o formato enviado ao LLM (F9)."""
    Path(path).write_text(_subtitle_text(segs, "", fmt_srt, True, max_chars), encoding="utf-8")
    return Path(path)


def plain_text(segs: list[dict], pause_s: float = 1.5, max_par_chars: int = 500) -> str:
    """Transcrição completa em parágrafos (nova quebra após pausa longa ou, passado
    max_par_chars, no fim de uma frase). Sem cabeçalho nem tempos: pode ser hipótese de WER."""
    pars, atual, anterior = [], [], None
    for s in segs:
        tam = sum(len(t) + 1 for t in atual)
        pausa = anterior is not None and s["start"] - anterior["end"] > pause_s
        fim_frase = atual and atual[-1].rstrip().endswith((".", "?", "!"))
        if atual and (pausa or (tam >= max_par_chars and fim_frase)):
            pars.append(" ".join(atual))
            atual = []
        atual.append(s["text"])
        anterior = s
    if atual:
        pars.append(" ".join(atual))
    return "\n\n".join(pars).strip() + "\n"


def save_outputs(segs: list[dict], out_dir: str | Path, stem: str, meta: dict | None = None) -> dict[str, Path]:
    """Grava <stem>.vtt, .srt, .txt, .segments.srt (para o LLM) e .json (segmentos + meta)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    caminhos = {
        "vtt": write_vtt(segs, out_dir / f"{stem}.vtt"),
        "srt": write_srt(segs, out_dir / f"{stem}.srt"),
        "srt_segments": write_srt(segs, out_dir / f"{stem}.segments.srt", max_chars=None),
        "txt": out_dir / f"{stem}.txt",
        "json": out_dir / f"{stem}.json",
    }
    caminhos["txt"].write_text(plain_text(segs), encoding="utf-8")
    caminhos["json"].write_text(json.dumps({**(meta or {}), "segments": segs}, ensure_ascii=False, indent=2),
                                encoding="utf-8")
    return caminhos


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("media")
    ap.add_argument("--model", default="small", choices=MODELS)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--vocab", help="lista de termos (um por linha) para o initial_prompt (I5)")
    ap.add_argument("--out", help="pasta de saída (padrão: data/derived/<nome>)")
    a = ap.parse_args()
    prompt = build_prompt(read_terms(a.vocab)) if a.vocab else None
    segs, dur, tempo = transcribe(a.media, a.model, a.device, initial_prompt=prompt)
    stem = f"{Path(a.media).stem}.{a.model}{'.vocab' if prompt else ''}"
    out = Path(a.out) if a.out else Path(__file__).resolve().parent.parent / "data" / "derived" / Path(a.media).stem
    meta = {"media": str(a.media), "model": a.model, "device": resolve_device(a.device)[0],
            "initial_prompt": prompt, "duration_s": dur, "time_s": tempo, "removed": transcribe.last_removed}
    caminhos = save_outputs(segs, out, stem, meta)
    print(f"{dur:.0f} s de áudio em {tempo:.1f} s (RTF {tempo / dur:.3f}); {len(segs)} segmentos, "
          f"{len(transcribe.last_removed)} removidos")
    for k, c in caminhos.items():
        print(f"  {k:13} {c}")


if __name__ == "__main__":
    main()
