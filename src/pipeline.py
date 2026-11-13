"""Processa um vídeo de ponta a ponta e monta a demo da página (F1, F2, F5 e, opcional, F9).

    python src/pipeline.py data/raw/aula01.mp4 --model small
    python src/pipeline.py data/raw/aula01.mp4 --model small --summary ollama
    python src/pipeline.py data/raw/aula01.mp4 --model small --context L+F    # F10: o slide guia o STT (exige Tesseract)
    cd web && python -m http.server        # abrir http://localhost:8000

Saídas: data/derived/<video>/ (transcrição, keyframes, resumo) e web/data/<video>/ (cópias para a
página). Sem backend de LLM disponível, o resumo é pulado e o resto segue.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import stt
from extract import keyframes, slides as extract_slides

ROOT = Path(__file__).resolve().parent.parent
DERIVED, WEB = ROOT / "data" / "derived", ROOT / "web"


def process(video: str | Path, model: str = "small", device: str = "auto", vocab: str | None = None,
            summary: str | None = None, llm_model: str | None = None, step_s: float | None = None,
            thr: float = 0.04, web: bool = True, progress=print, frames_mode: str = "slides",
            context: str | None = None, ocr: bool = False) -> dict:
    """context: None | "G" | "L" | "L+F" (F10): o texto dos slides guia o STT (context_stt). Exige o modo de slides
    e o OCR (Tesseract); sem eles o contexto é pulado, com o motivo, e o STT segue sem ele. ocr=True só preenche o
    `texto` dos slides (a busca da página), sem contexto. Sem context nem ocr, o comportamento é o de sempre."""
    video = Path(video)
    v = video.stem
    pasta = DERIVED / v
    out: dict = {"video": video}
    frames_dir = pasta / "frames"

    glossario = None
    if (context or ocr) and frames_mode != "slides":
        progress("OCR e contexto pulados: exigem o modo de slides (--frames-mode slides).")
        context = None
        ocr = False
    if context or ocr:  # F10: os slides vêm antes do STT, porque o glossário alimenta a decodificação
        import context as ctx
        import ocr as ocr_mod

        progress("Slides completos...")
        out["frames"] = extract_slides(video, frames_dir, step_s or 0.5)
        progress(f"  {len(out['frames'])} slides")
        if motivo := ocr_mod.status("tesseract"):
            progress(f"OCR pulado ({motivo}); {'o contexto também' if context else 'seguindo sem texto nos slides'}.")
            context = None
        else:
            progress(f"OCR (tesseract) em {len(out['frames'])} slides...")
            panel = json.loads((frames_dir / "panel.json").read_text(encoding="utf-8"))
            box, title_box = ocr_mod.title_region(panel)
            out["frames"] = ocr_mod.ocr_slides(frames_dir, out["frames"], box or panel.get("ocr_box"),
                                               title_box=title_box)
            glossario = ctx.build_glossary(out["frames"])
            ctx.save_glossary(pasta / "glossario.json", glossario)
            (pasta / "ocr.json").write_text(json.dumps({"motor": "tesseract", "psm": 6, "tempo_s": round(
                sum(f["ocr_tempo_s"] for f in out["frames"]), 3), "slides": out["frames"]}, ensure_ascii=False, indent=2),
                encoding="utf-8")

    progress(f"STT ({model}{', contexto ' + context if context else ''})...")
    prompt = stt.build_prompt(stt.read_terms(vocab)) if vocab else None
    meta_ctx: dict = {}
    if context:
        import context_stt
        from extract import load_audio

        audio = load_audio(video)
        r = context_stt.transcribe_context(audio, context, context_stt.make_asr(model, device), glossario)
        segs, dur, tempo = r["segs"], len(audio) / context_stt.SR, r["tempo_s"]
        removed = r["removed"]
        meta_ctx = {"context": context, "calls": r["n_calls"], "fix": r["fix_summary"], "plan": r["plan"]}
        out["corrections"] = r["corrections"]
    else:
        segs, dur, tempo = stt.transcribe(video, model, device, initial_prompt=prompt)
        removed = stt.transcribe.last_removed
    stem = f"{v}.{model}{'.vocab' if prompt else ''}{'.ctx-' + context.replace('+', '') if context else ''}"
    out.update(stt.save_outputs(segs, pasta, stem, {
        "media": str(video), "model": model, "device": stt.resolve_device(device)[0], "initial_prompt": prompt,
        "duration_s": dur, "time_s": tempo, "removed": removed, **meta_ctx}))
    if out.get("corrections"):
        (pasta / f"{stem}.correcoes.jsonl").write_text(
            "\n".join(json.dumps(c, ensure_ascii=False) for c in out["corrections"]) + "\n", encoding="utf-8")
    out.update(segments=segs, duration_s=dur, time_s=tempo)
    progress(f"  {dur:.0f} s de áudio em {tempo:.1f} s (RTF {tempo / dur:.3f}), {len(segs)} segmentos"
             + (f", {len(out['corrections'])} correções" if context == "L+F" else ""))

    if "frames" in out:  # já calculados (caminho do F10)
        pass
    elif frames_mode == "slides":  # padrão: um quadro completo por slide (extract.slides)
        progress("Slides completos...")
        out["frames"] = extract_slides(video, frames_dir, step_s or 0.5)
        progress(f"  {len(out['frames'])} slides")
    else:  # baseline: primeiro quadro após cada mudança (extract.keyframes)
        progress("Keyframes (baseline)...")
        out["frames"] = keyframes(video, frames_dir, step_s or 1.0, thr)
        progress(f"  {len(out['frames'])} keyframes")

    if summary:
        from summarize import backend_status, extract_topics, summarize, write_chapters_vtt

        motivo = backend_status(summary, llm_model)
        if motivo:  # a F9 é opcional: sem LLM, o resto do pipeline segue
            progress(f"Resumo pulado: {motivo}.")
        else:
            progress(f"Resumo ({summary})...")
            try:
                texto, meta = summarize(out["srt_segments"].read_text(encoding="utf-8"), summary, llm_model)
            except RuntimeError as e:
                progress(f"Resumo falhou, seguindo sem ele: {e}")
            else:
                md = pasta / f"{stem}.summary.{summary}.md"
                md.write_text(texto, encoding="utf-8")
                md.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
                out.update(summary_md=md, summary_meta=meta, chapters=write_chapters_vtt(
                    extract_topics(texto), md.with_suffix(".chapters.vtt"), dur))

    if web:
        out["web"] = build_web(v, video, out, frames_dir)
        progress(f"Demo em {out['web']}")
    return out


def build_web(v: str, video: Path, out: dict, frames_dir: Path) -> Path:
    """Copia para web/data/<video>/ o que a página lê e atualiza web/videos.json."""
    dest = WEB / "data" / v
    shutil.rmtree(dest, ignore_errors=True)
    (dest / "frames").mkdir(parents=True)
    shutil.copy(video, dest / f"video{video.suffix.lower()}")
    shutil.copy(out["vtt"], dest / "subs.vtt")
    for f in out["frames"]:
        shutil.copy(frames_dir / f["file"], dest / "frames" / f["file"])
    # contrato 5.3; "texto" fica vazio até o OCR (etapa 3) preencher
    slides = [{"t": f["t"], "file": f"frames/{f['file']}", "texto": f.get("texto", ""),
               **{k: f[k] for k in ("t_complete", "t_end", "n_stages") if k in f}} for f in out["frames"]]
    (dest / "slides.json").write_text(json.dumps(slides, ensure_ascii=False, indent=2), encoding="utf-8")
    (dest / "transcript.json").write_text(json.dumps(
        [{"t": s["start"], "end": s["end"], "text": s["text"]} for s in out["segments"]],
        ensure_ascii=False), encoding="utf-8")
    manifest = {"id": v, "title": v, "video": f"video{video.suffix.lower()}", "subs": "subs.vtt",
                "slides": "slides.json", "transcript": "transcript.json", "chapters": None, "summary": None}
    if out.get("chapters"):
        shutil.copy(out["chapters"], dest / "chapters.vtt")
        shutil.copy(out["summary_md"], dest / "summary.md")
        manifest.update(chapters="chapters.vtt", summary="summary.md")
    (dest / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    ids = sorted(p.name for p in (WEB / "data").iterdir() if (p / "manifest.json").exists())
    (WEB / "data" / "videos.json").write_text(json.dumps(ids), encoding="utf-8")
    return dest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--model", default="small", choices=stt.MODELS)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--vocab", help="lista de termos para o initial_prompt (ex.: data/gt/terms.txt)")
    ap.add_argument("--summary", choices=["ollama", "claude"], help="F9: gera o resumo com esse backend")
    ap.add_argument("--llm-model")
    ap.add_argument("--frames-mode", choices=["slides", "diff"], default="slides",
                    help="slides: quadro completo de cada slide (padrão); diff: baseline por diferença de quadros")
    ap.add_argument("--step-s", type=float, help="intervalo de amostragem (padrão: 0,5 s em slides, 1,0 s em diff)")
    ap.add_argument("--thr", type=float, default=0.04, help="só no modo diff")
    ap.add_argument("--context", choices=["G", "L", "L+F"],
                    help="F10: o texto dos slides (OCR) guia o STT. G: vocabulário do vídeo; L: o do slide visível; "
                         "L+F: o L e a correção de siglas e nomes. Exige o Tesseract")
    ap.add_argument("--ocr", action="store_true", help="só preenche o texto dos slides (a busca da página); sem contexto")
    ap.add_argument("--no-web", action="store_true")
    a = ap.parse_args()
    process(a.video, a.model, a.device, a.vocab, a.summary, a.llm_model, a.step_s, a.thr, not a.no_web,
            frames_mode=a.frames_mode, context=a.context, ocr=a.ocr)


if __name__ == "__main__":
    main()
