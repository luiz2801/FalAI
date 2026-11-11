"""Processa um vídeo de ponta a ponta e monta a demo da página (F1, F2, F5 e, opcional, F9).

    python src/pipeline.py data/raw/aula01.mp4 --model small
    python src/pipeline.py data/raw/aula01.mp4 --model small --summary ollama
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
from extract import keyframes

ROOT = Path(__file__).resolve().parent.parent
DERIVED, WEB = ROOT / "data" / "derived", ROOT / "web"


def process(video: str | Path, model: str = "small", device: str = "auto", vocab: str | None = None,
            summary: str | None = None, llm_model: str | None = None, step_s: float = 1.0,
            thr: float = 0.04, web: bool = True, progress=print) -> dict:
    video = Path(video)
    v = video.stem
    pasta = DERIVED / v
    out: dict = {"video": video}

    progress(f"STT ({model})...")
    prompt = stt.build_prompt(stt.read_terms(vocab)) if vocab else None
    segs, dur, tempo = stt.transcribe(video, model, device, initial_prompt=prompt)
    stem = f"{v}.{model}{'.vocab' if prompt else ''}"
    out.update(stt.save_outputs(segs, pasta, stem, {
        "media": str(video), "model": model, "device": stt.resolve_device(device)[0], "initial_prompt": prompt,
        "duration_s": dur, "time_s": tempo, "removed": stt.transcribe.last_removed}))
    out.update(segments=segs, duration_s=dur, time_s=tempo)
    progress(f"  {dur:.0f} s de áudio em {tempo:.1f} s (RTF {tempo / dur:.3f}), {len(segs)} segmentos")

    progress("Keyframes...")
    frames_dir = pasta / "frames"
    out["frames"] = keyframes(video, frames_dir, step_s, thr)
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
    slides = [{"t": f["t"], "file": f"frames/{f['file']}", "texto": f.get("texto", "")} for f in out["frames"]]
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
    ap.add_argument("--step-s", type=float, default=1.0)
    ap.add_argument("--thr", type=float, default=0.04)
    ap.add_argument("--no-web", action="store_true")
    a = ap.parse_args()
    process(a.video, a.model, a.device, a.vocab, a.summary, a.llm_model, a.step_s, a.thr, not a.no_web)


if __name__ == "__main__":
    main()
