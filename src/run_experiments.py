"""Matriz de experimentos -> results/raw.csv (guia.md, seções 4.6 e 5.5).

    python src/run_experiments.py                       # RQ1: STT × ruído (tiny, base, small)
    python src/run_experiments.py --models tiny small medium --reps 3
    python src/run_experiments.py --vocab               # + condição limpo_vocab (I5)
    python src/run_experiments.py --summaries --backend ollama    # F9: resumo de cada transcrição
    python src/run_experiments.py --context G P L L+F O --models small   # F10: o slide como contexto (RQ6 a RQ8)

Entradas: data/raw/<video>.mp4, data/gt/clips.csv (trecho com gabarito de cada vídeo),
data/gt/<video>.stt.txt (gabarito humano) e, com --vocab, data/gt/terms.txt.
Execuções já registradas no raw.csv são puladas (use --redo para refazer).
Com --context, exige também data/derived/<video>/glossario.json (python src/context.py glossary <vídeo>).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path

import metrics
import stt
from degrade import add_noise
from extract import clip_audio, load_audio, media_duration

ROOT = Path(__file__).resolve().parent.parent
RAW, GT, DERIVED, RESULTS = ROOT / "data/raw", ROOT / "data/gt", ROOT / "data/derived", ROOT / "results"
RAW_CSV = RESULTS / "raw.csv"
FIELDS = ["video", "tarefa", "modelo", "condicao", "repeticao", "metrica_erro", "valor_erro",
          "tempo_s", "duracao_s", "rtf"]
CONDITIONS = {"limpo": None, "snr20": 20, "snr10": 10, "snr0": 0}
VIDEO_EXT = {".mp4", ".mkv", ".webm", ".mov"}


# ------------------------------------------------------------------ utilidades
def hardware() -> dict:
    cpu = platform.processor()
    try:
        cpu = next(l.split(":", 1)[1].strip() for l in Path("/proc/cpuinfo").read_text().splitlines()
                   if l.startswith("model name"))
    except (OSError, StopIteration):
        pass
    try:
        ram_kb = int(next(l.split()[1] for l in Path("/proc/meminfo").read_text().splitlines()
                          if l.startswith("MemTotal")))
    except (OSError, StopIteration):
        ram_kb = 0
    try:
        import ctranslate2
        gpus = ctranslate2.get_cuda_device_count()
    except Exception:
        gpus = 0
    # gpus_cuda conta as GPUs visíveis; device_auto é o que o `--device auto` de fato usa (sem cuBLAS, cai para CPU)
    return {"cpu": cpu, "threads": os.cpu_count(), "ram_gb": round(ram_kb / 2**20, 1), "gpus_cuda": gpus,
            "device_auto": stt.resolve_device("auto")[0],
            "os": platform.platform(), "python": platform.python_version()}


def read_clips() -> dict[str, tuple[float, float]]:
    """data/gt/clips.csv: video,start_s,dur_s."""
    arq = GT / "clips.csv"
    if not arq.exists():
        return {}
    with arq.open(encoding="utf-8") as f:
        return {r["video"]: (float(r["start_s"]), float(r["dur_s"])) for r in csv.DictReader(f)}


def done_keys() -> set[tuple]:
    if not RAW_CSV.exists():
        return set()
    with RAW_CSV.open(encoding="utf-8") as f:
        return {(r["video"], r["tarefa"], r["modelo"], r["condicao"], r["repeticao"], r["metrica_erro"])
                for r in csv.DictReader(f)}


def append_rows(rows: list[dict]) -> None:
    novo = not RAW_CSV.exists()
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    with RAW_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if novo:
            w.writeheader()
        w.writerows(rows)


def row(video, tarefa, modelo, condicao, rep, metrica, valor, tempo, dur) -> dict:
    return {"video": video, "tarefa": tarefa, "modelo": modelo, "condicao": condicao, "repeticao": rep,
            "metrica_erro": metrica, "valor_erro": "" if valor is None else round(valor, 6),
            "tempo_s": "" if tempo is None else round(tempo, 4),
            "duracao_s": "" if dur is None else round(dur, 3),
            "rtf": "" if tempo is None or not dur else round(metrics.rtf(tempo, dur), 6)}


def prepare_clip(video: str, cond: str, clips) -> Path:
    """data/derived/<video>/<video>.clip.<cond>.wav (gerado se não existir; seed 0)."""
    pasta = DERIVED / video
    limpo = pasta / f"{video}.clip.limpo.wav"
    if not limpo.exists():
        midia = next(p for p in RAW.iterdir() if p.stem == video and p.suffix.lower() in VIDEO_EXT)
        clip_audio(midia, limpo, *clips[video])
    snr = CONDITIONS[cond.removesuffix("_vocab")]
    if snr is None:
        return limpo
    destino = pasta / f"{video}.clip.{cond}.wav"
    if not destino.exists():
        medido = add_noise(limpo, destino, snr, seed=0)
        if abs(medido - snr) > 0.5:
            print(f"  AVISO {destino.name}: SNR medido {medido:.2f} dB (alvo {snr})")
    return destino


# ------------------------------------------------------------------ RQ1: STT × ruído
def run_stt(a) -> None:
    clips = read_clips()
    videos = sorted(v for v in clips if (GT / f"{v}.stt.txt").exists())
    sem_gt = sorted(set(clips) - set(videos))
    if sem_gt:
        print("Sem gabarito (pulados):", ", ".join(sem_gt))
    if not videos:
        sys.exit("Nada a rodar: preencha data/gt/clips.csv e data/gt/<video>.stt.txt (ver README).")
    termos = stt.read_terms(GT / "terms.txt") if (GT / "terms.txt").exists() else []
    prompt = stt.build_prompt(termos) if a.vocab and termos else None
    if a.vocab and not termos:
        sys.exit("--vocab exige data/gt/terms.txt")
    conds = list(a.conditions) + (["limpo_vocab"] if prompt else [])
    feitos = set() if a.redo else done_keys()

    for modelo in a.models:
        # sem o filtro de alucinações, o modelo ganha outro rótulo para não misturar linhas
        rotulo = f"{modelo}-semfiltro" if a.no_clean else modelo
        pendentes = [(v, c) for v in videos for c in conds
                     if (v, "stt", rotulo, c, str(a.reps), "wer") not in feitos]
        if not pendentes:
            print(f"{rotulo}: tudo feito")
            continue
        _, t_carga = stt.load_model(modelo, a.device)
        append_rows([row("_", "stt", rotulo, "carga", 1, "carga_modelo", None, t_carga, None)])
        # aquecimento: primeira execução descartada (armadilha 7.1.2)
        stt.transcribe(prepare_clip(pendentes[0][0], "limpo", clips), modelo, a.device)
        for video, cond in pendentes:
            audio = prepare_clip(video, cond, clips)
            ref = (GT / f"{video}.stt.txt").read_text(encoding="utf-8")
            for rep in range(1, a.reps + 1):
                segs, dur, tempo = stt.transcribe(audio, modelo, a.device, clean=not a.no_clean,
                                                  initial_prompt=prompt if cond.endswith("_vocab") else None)
                hyp = " ".join(s["text"] for s in segs)
                linhas = [row(video, "stt", rotulo, cond, rep, "wer", metrics.wer(ref, hyp), tempo, dur),
                          row(video, "stt", rotulo, cond, rep, "cer", metrics.cer(ref, hyp), tempo, dur)]
                if termos:
                    acc = metrics.term_accuracy(ref, hyp, termos)["accuracy"]
                    if acc is not None:
                        linhas.append(row(video, "stt", rotulo, cond, rep, "acerto_termos", acc, tempo, dur))
                append_rows(linhas)
                if rep == 1:  # hipótese guardada para a análise de erros (regra 11)
                    stt.save_outputs(segs, DERIVED / video / "stt", f"{video}.{rotulo}.{cond}",
                                     {"time_s": tempo, "duration_s": dur, "removed": stt.transcribe.last_removed})
            print(f"[{time.strftime('%H:%M:%S')}] {rotulo:8} {video} {cond:12} "
                  f"WER={metrics.wer(ref, hyp):.3f} RTF={tempo / dur:.3f}")
        stt.unload_models()


# ------------------------------------------------------------------ RQ6 a RQ8: o slide como contexto (F10)
CONTEXTS = ["B0", "G", "P", "L", "L+F", "O"]  # B0 repete a RQ1 pelo caminho do contexto: a linha de base com as
# métricas por slide (acerto_termos_slide e _fora), que a RQ1 não grava


def glossary_path(video: str) -> Path:
    return DERIVED / video / "glossario.json"


def context_metrics(ref: str, hyp: str, termos: list[str], no_slide: list[str], fora: list[str],
                    base_segs: list[dict], segs: list[dict]) -> list[tuple[str, float]]:
    """(métrica, valor) de uma execução em modo de contexto (contrato 5.5). `acerto_termos_slide` e `_fora`
    dividem os termos humanos entre os que estavam em algum slide do trecho e os que não estavam."""
    import context_stt

    saida = [("wer", metrics.wer(ref, hyp)), ("cer", metrics.cer(ref, hyp))]
    if termos:
        todos = metrics.term_accuracy(ref, hyp, termos)
        if todos["accuracy"] is not None:
            saida.append(("acerto_termos", todos["accuracy"]))
        saida.append(("falsos_positivos", todos["false_positives"]))
        for nome, lista in (("acerto_termos_slide", no_slide), ("acerto_termos_fora", fora)):
            acc = metrics.term_accuracy(ref, hyp, lista)["accuracy"] if lista else None
            if acc is not None:
                saida.append((nome, acc))
    saida.append(("taxa_alteracao", context_stt.changed_fraction(base_segs, segs)))
    return saida


def run_context(a) -> None:
    """RQ6 a RQ8: cada vídeo com trecho, gabarito e glossário × ruído × modo de contexto. Grava as condições como
    `<ruído>@<modo>` (limpo@L, snr10@L+F...). O B0 só é gravado se pedido (`--context B0 ...`); de qualquer jeito
    ele roda aqui uma vez por vídeo e ruído, para a `taxa_alteracao`."""
    import context
    import context_stt

    clips = read_clips()
    videos = sorted(v for v in clips if (GT / f"{v}.stt.txt").exists())
    com_glossario = [v for v in videos if glossary_path(v).exists()]
    if sem := sorted(set(videos) - set(com_glossario)):
        print("Sem glossário (pulados; rode python src/context.py glossary <vídeo>):", ", ".join(sem))
    if not com_glossario:
        sys.exit("Nada a rodar: preencha data/gt/ (trecho e gabarito) e gere data/derived/<video>/glossario.json.")
    termos = stt.read_terms(GT / "terms.txt") if (GT / "terms.txt").exists() else []
    if "O" in a.context and not termos:
        sys.exit("O modo O (oráculo) exige data/gt/terms.txt")
    if "P" in a.context and len(com_glossario) < 2:
        sys.exit("O modo P (placebo) exige o glossário de pelo menos 2 vídeos: os termos vêm de outro vídeo.")
    feitos = set() if a.redo else done_keys()
    plan_kw = {"lead_s": a.lead_s, "lag_s": a.lag_s, "carry": not a.no_carry}

    for video in com_glossario:  # custo de OCR e glossário, uma vez por vídeo (o ΔRTF do relatório usa)
        info = DERIVED / video / "ocr.json"
        midia = next((p for p in RAW.iterdir() if p.stem == video and p.suffix.lower() in VIDEO_EXT), None) \
            if RAW.exists() else None
        if info.exists() and midia and (video, "stt", "_", "ctx", "1", "tempo_ctx") not in feitos:
            meta = json.loads(info.read_text(encoding="utf-8"))
            append_rows([row(video, "stt", "_", "ctx", 1, "tempo_ctx", None,
                             meta.get("tempo_s", 0.0) + meta.get("glossario_s", 0.0), media_duration(midia))])

    for modelo in a.models:
        rotulo = f"{modelo}-semfiltro" if a.no_clean else modelo
        pendentes = [(v, c, m) for v in com_glossario for c in a.conditions for m in a.context
                     if (v, "stt", rotulo, f"{c}@{m}", str(a.reps), "wer") not in feitos]
        if not pendentes:
            print(f"{rotulo}: tudo feito")
            continue
        asr = context_stt.make_asr(modelo, a.device, via=a.via)
        _, t_carga = stt.load_model(modelo, a.device)
        if ("_", "stt", rotulo, "carga", "1", "carga_modelo") not in feitos:
            append_rows([row("_", "stt", rotulo, "carga", 1, "carga_modelo", None, t_carga, None)])
        # aquecimento: primeira execução descartada (armadilha 7.1.2)
        context_stt.transcribe_context(load_audio(prepare_clip(pendentes[0][0], "limpo", clips)), "B0", asr)
        base: dict[tuple[str, str], list[dict]] = {}
        for video, cond, modo in pendentes:
            audio = load_audio(prepare_clip(video, cond, clips))
            dur = len(audio) / context_stt.SR
            ref = (GT / f"{video}.stt.txt").read_text(encoding="utf-8")
            gl = context.shift_glossary(context.load_glossary(glossary_path(video)), clips[video][0], dur)
            outro = com_glossario[(com_glossario.index(video) + 1) % len(com_glossario)]
            placebo = context.load_glossary(glossary_path(outro)) if "P" in a.context and outro != video else None
            if (video, cond) not in base:
                base[(video, cond)] = context_stt.transcribe_context(audio, "B0", asr, clean=not a.no_clean)["segs"]
            no_slide, fora = context.split_terms(termos, gl)
            kw = {"glossary": gl, "oracle_terms": termos or None, "placebo_glossary": placebo, "plan_kw": plan_kw,
                  "margin_s": a.margin_s, "max_terms": a.max_terms, "tau": a.tau, "s_min": a.s_min,
                  "clean": not a.no_clean}
            for rep in range(1, a.reps + 1):
                r = context_stt.transcribe_context(audio, modo, asr, **kw)
                hyp = context_stt.hyp_text(r["segs"])
                linhas = [row(video, "stt", rotulo, f"{cond}@{modo}", rep, nome, valor, r["tempo_s"], dur)
                          for nome, valor in context_metrics(ref, hyp, termos, no_slide, fora, base[(video, cond)],
                                                             r["segs"])]
                append_rows(linhas)
                if rep == 1:  # hipótese e correções guardadas para a análise de erros e a auditoria (regra 11)
                    stem = f"{video}.{rotulo}.{cond}@{modo}"
                    stt.save_outputs(r["segs"], DERIVED / video / "stt", stem,
                                     {"time_s": r["tempo_s"], "duration_s": dur, "calls": r["n_calls"],
                                      "removed": r["removed"], "fix": r["fix_summary"], "plan": r["plan"]})
                    if r["corrections"]:
                        (DERIVED / video / "stt" / f"{stem}.correcoes.jsonl").write_text(
                            "\n".join(json.dumps(c, ensure_ascii=False) for c in r["corrections"]) + "\n",
                            encoding="utf-8")
            print(f"[{time.strftime('%H:%M:%S')}] {rotulo:8} {video} {cond}@{modo:4} "
                  f"WER={metrics.wer(ref, hyp):.3f} RTF={r['tempo_s'] / dur:.3f} chamadas={r['n_calls']}")
        stt.unload_models()


# ------------------------------------------------------------------ F9: resumos
def run_summaries(a) -> None:
    from summarize import backend_status, summarize

    if motivo := backend_status(a.backend, a.llm_model):
        sys.exit(f"Backend {a.backend} indisponível: {motivo}.")
    if a.backend == "claude" and not a.confirm_paid:
        sys.exit("O backend claude é pago: rode de novo com --confirm-paid (CLAUDE.md, regra 12).")
    videos = sorted(p for p in RAW.iterdir() if p.suffix.lower() in VIDEO_EXT) if RAW.exists() else []
    feitos = set() if a.redo else done_keys()
    llm = f"{a.backend}:{a.llm_model or 'padrao'}"
    for modelo in a.models:
        for midia in videos:
            v = midia.stem
            pasta = DERIVED / v / "summary"
            arq = pasta / f"{v}.{modelo}.json"
            if arq.exists():
                segs = json.loads(arq.read_text(encoding="utf-8"))["segments"]
                dur = media_duration(midia)
            else:
                segs, dur, tempo = stt.transcribe(midia, modelo, a.device)
                stt.save_outputs(segs, pasta, f"{v}.{modelo}", {"time_s": tempo, "duration_s": dur})
            srt = (pasta / f"{v}.{modelo}.segments.srt").read_text(encoding="utf-8")
            for rep in range(1, a.reps + 1):
                if (v, "resumo", modelo, llm, str(rep), "tempo_llm") in feitos:
                    continue
                texto, meta = summarize(srt, a.backend, a.llm_model)
                md = RESULTS / "summaries" / f"{v}__{modelo}__{llm.replace(':', '-').replace('/', '-')}__r{rep}.md"
                md.parent.mkdir(parents=True, exist_ok=True)
                md.write_text(texto, encoding="utf-8")
                meta["condicao"] = llm  # chave exata usada no raw.csv (lida por eval_summaries.py)
                md.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
                append_rows([row(v, "resumo", modelo, llm, rep, "tempo_llm", None, meta["time_s"], dur),
                             row(v, "resumo", modelo, llm, rep, "custo_usd", meta["cost_usd"], None, dur)])
                print(f"[{time.strftime('%H:%M:%S')}] resumo {md.name} ({meta['time_s']} s)")
        stt.unload_models()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["tiny", "base", "small"], choices=stt.MODELS)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--vocab", action="store_true", help="acrescenta a condição limpo_vocab (I5)")
    ap.add_argument("--summaries", action="store_true", help="F9 em vez da RQ1")
    ap.add_argument("--context", nargs="+", choices=CONTEXTS,
                    help="F10 em vez da RQ1: transcreve com o texto do slide como contexto (B0 sem contexto, G global, "
                         "P placebo, L local, L+F local com correção, O oráculo). Exige glossario.json de cada vídeo")
    ap.add_argument("--via", default="hotwords", choices=["hotwords", "initial_prompt"],
                    help="como o contexto entra no Whisper (initial_prompt só vale na 1ª janela: D15)")
    ap.add_argument("--margin-s", type=float, default=2.0, help="margem de áudio de cada trecho do modo L (s)")
    ap.add_argument("--lead-s", type=float, default=2.0, help="o slide vale t − lead_s antes de aparecer")
    ap.add_argument("--lag-s", type=float, default=1.0, help="e t_end + lag_s depois de sair")
    ap.add_argument("--no-carry", action="store_true", help="intervalo sem slide fica sem termos")
    ap.add_argument("--max-terms", type=int, default=30, help="termos por intervalo (e no modo G)")
    ap.add_argument("--tau", type=float, default=0.5, help="gate de confiança da correção (L+F)")
    ap.add_argument("--s-min", type=float, default=0.85, help="semelhança mínima da correção (L+F)")
    ap.add_argument("--backend", default="ollama", choices=["ollama", "claude"])
    ap.add_argument("--llm-model")
    ap.add_argument("--confirm-paid", action="store_true")
    ap.add_argument("--no-clean", action="store_true",
                    help="não aplica o filtro de alucinações do stt.clean_segments (para comparar)")
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args()
    if a.summaries and a.reps == 3:
        a.reps = 2
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "hardware.json").write_text(json.dumps(hardware(), ensure_ascii=False, indent=2), encoding="utf-8")
    run_summaries(a) if a.summaries else run_context(a) if a.context else run_stt(a)


if __name__ == "__main__":
    main()
