"""FalAI Studio: GUI local do projeto (guia.md, seção 4.10).

    uv run --extra gui python src/studio.py            # http://127.0.0.1:8000
    uv run --extra gui python src/studio.py --port 9000

Servidor FastAPI que serve `web/` (`studio.html` é a GUI; `index.html`, o visualizador) e expõe a API
das telas: biblioteca e envio de vídeos, processamento (`pipeline.process`), experimentos
(`run_experiments.py`) e relatório (`report.py`) como subprocessos, resultados lidos do `raw.csv`,
edição do gabarito, classificação de erros e comparação A/B. Um trabalhador só, porque a GPU é única.
Escuta só em 127.0.0.1: não tem autenticação e não deve ser exposto.

A GUI não calcula nem grava métrica do artigo: os números vêm do `results/raw.csv` (CLAUDE.md, regra 2)
e o gabarito é digitado por humano, sem saída de modelo à vista (regra 3).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import csv
import json
import math
import os
import queue
import re
import subprocess
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import stt

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
RAW, DERIVED, GT = ROOT / "data" / "raw", ROOT / "data" / "derived", ROOT / "data" / "gt"
WEB, RESULTS = ROOT / "web", ROOT / "results"
RAW_CSV = RESULTS / "raw.csv"
ERRORS_CSV = RESULTS / "erros.csv"
LICENSES = RAW / "licenses.json"  # regra 4: licença anotada por vídeo (data/raw fica fora do git)
EXTS = {".mp4", ".mkv", ".webm", ".mov"}  # as mesmas de run_experiments.VIDEO_EXT
MAX_UPLOAD = 4 * 1024**3
CONDITIONS = ["limpo", "snr20", "snr10", "snr0"]  # run_experiments.CONDITIONS (teste confere)
ERROR_FIELDS = ["video", "modelo", "condicao", "idx", "tipo", "ref", "hyp", "categoria", "nota"]
# categorias da análise de erros (docs/decisoes.md, D12)
CATEGORIES = {"nome_proprio": "Nome próprio", "sigla": "Sigla", "numero": "Número",
              "termo_tecnico": "Termo técnico", "sotaque": "Sotaque/pronúncia", "ruido": "Ruído",
              "hesitacao": "Hesitação/fala espontânea", "normalizacao": "Normalização/grafia",
              "alucinacao": "Alucinação", "outro": "Outro"}

# (prefixo da mensagem de progresso do pipeline, id da etapa)
STAGES = [("STT", "stt"), ("Slides", "frames"), ("Keyframes", "frames"), ("Resumo", "summary"), ("Demo", "web")]
# linha que run_experiments.py imprime ao terminar cada (modelo, vídeo, condição) ou cada resumo
_DONE_LINE = re.compile(r"^\[\d\d:\d\d:\d\d\] ")
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

app = FastAPI(title="FalAI Studio")
_lock = threading.Lock()
jobs: dict[str, Job] = {}
_fila: queue.Queue[str] = queue.Queue()
_workers: list[threading.Thread] = []


# ================================================================== trabalhos (fila única)
class Job:
    def __init__(self, kind: str, title: str, params: dict, cmd: str, argv: list[str] | None = None,
                 total: int | None = None, video: str | None = None):
        self.id = uuid.uuid4().hex[:8]
        self.kind, self.title, self.params, self.cmd, self.argv = kind, title, params, cmd, argv
        self.video = video
        self.status = "queued"  # queued | running | done | error | cancelled
        self.stage: str | None = None
        self.done, self.total = 0, total
        self.log: list[str] = []
        self.error: str | None = None
        self.created = time.time()
        self.started: float | None = None
        self.finished: float | None = None
        self.summary: dict | None = None
        self.proc: subprocess.Popen | None = None

    def say(self, msg: str) -> None:
        if self.kind == "pipeline" and not msg.startswith(" "):  # só inícios de etapa (sem indentação)
            for prefix, st in STAGES:
                if msg.startswith(prefix):
                    self.stage = st
        elif self.kind == "experiment" and _DONE_LINE.match(msg):
            self.done += 1
        self.log.append(msg)

    def public(self, with_log: bool = False) -> dict:
        d = {"id": self.id, "kind": self.kind, "title": self.title, "video": self.video, "status": self.status,
             "stage": self.stage, "done": self.done, "total": self.total, "cmd": self.cmd, "error": self.error,
             "params": self.params, "created": self.created, "started": self.started, "finished": self.finished,
             "summary": self.summary, "log_lines": len(self.log), "cancellable": self.kind != "pipeline"}
        if with_log:
            d["log"] = self.log
        return d


def _enqueue(job: Job) -> Job:
    with _lock:
        jobs[job.id] = job
        _fila.put(job.id)
        if not _workers:  # trabalhador único, iniciado no primeiro trabalho
            _workers.append(threading.Thread(target=_worker, daemon=True))
            _workers[0].start()
    return job


def _run_pipeline(job: Job) -> None:
    import pipeline  # importado aqui: carrega o faster-whisper só quando houver trabalho

    p = job.params
    out = pipeline.process(
        RAW / job.video, p["model"], p["device"], p.get("vocab"), p.get("summary"), p.get("llm_model"),
        p.get("step_s"), 0.04, True, job.say, p["frames_mode"])
    job.summary = {"duration_s": out["duration_s"], "time_s": out["time_s"],
                   "rtf": out["time_s"] / out["duration_s"] if out["duration_s"] else None,
                   "n_segments": len(out["segments"]), "n_slides": len(out["frames"]),
                   "has_summary": "summary_md" in out}


def _run_subprocess(job: Job) -> None:
    """Roda o mesmo comando do CLI (reprodutível) e repassa a saída linha a linha."""
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    job.proc = subprocess.Popen([sys.executable, *job.argv], cwd=ROOT, env=env, text=True, bufsize=1,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in job.proc.stdout:
        job.say(line.rstrip("\n"))
    code = job.proc.wait()
    if job.status == "cancelled":
        return
    if code != 0:
        raise RuntimeError(f"o processo terminou com código {code}" + (f": {job.log[-1]}" if job.log else ""))


def _worker() -> None:
    while True:
        job = jobs[_fila.get()]
        if job.status == "cancelled":
            continue
        job.status, job.started = "running", time.time()
        try:
            _run_pipeline(job) if job.kind == "pipeline" else _run_subprocess(job)
            if job.status != "cancelled":
                job.status = "done"
        except Exception as e:  # noqa: BLE001  (o erro vai para a GUI, o trabalhador segue vivo)
            job.status, job.error = "error", f"{type(e).__name__}: {e}"
            job.log.append(traceback.format_exc())
        finally:
            job.finished = time.time()
            if job.kind == "pipeline":
                with contextlib.suppress(Exception):  # liberar a GPU nunca deve derrubar o trabalhador
                    stt.unload_models()


# ================================================================== utilidades de dados
def _check_name(name: str) -> str:
    if not _NAME.fullmatch(name or "") or ".." in name:
        raise HTTPException(400, f"nome inválido: {name!r}")
    return name


def _read_licenses() -> dict:
    try:
        return json.loads(LICENSES.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _safe_name(name: str) -> str:
    p = Path(name)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", p.stem).strip("._") or "video"
    return stem + p.suffix.lower()


def _raw_videos() -> list[Path]:
    return sorted(f for f in RAW.glob("*") if f.suffix.lower() in EXTS) if RAW.exists() else []


def _read_clips() -> dict[str, tuple[float, float]]:
    """data/gt/clips.csv (guia.md, 5.7), na mesma leitura de run_experiments.read_clips."""
    arq = GT / "clips.csv"
    if not arq.exists():
        return {}
    with arq.open(encoding="utf-8") as f:
        return {r["video"]: (float(r["start_s"]), float(r["dur_s"])) for r in csv.DictReader(f)}


def _write_clips(clips: dict[str, tuple[float, float]]) -> None:
    GT.mkdir(parents=True, exist_ok=True)
    with (GT / "clips.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video", "start_s", "dur_s"])
        for v in sorted(clips):
            w.writerow([v, f"{clips[v][0]:g}", f"{clips[v][1]:g}"])


def _raw_rows() -> list[dict]:
    if not RAW_CSV.exists():
        return []
    with RAW_CSV.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _done_keys() -> set[tuple]:
    """As mesmas chaves de run_experiments.done_keys."""
    return {(r["video"], r["tarefa"], r["modelo"], r["condicao"], r["repeticao"], r["metrica_erro"])
            for r in _raw_rows()}


def _clean(x):
    """NaN/inf -> None, para o JSON."""
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def _hyp_dir(video: str) -> Path:
    return DERIVED / video / "stt"


def _hyp_sources() -> list[dict]:
    """Hipóteses guardadas por run_experiments (rep 1): data/derived/<v>/stt/<v>.<modelo>.<cond>.json."""
    out = []
    for v in sorted(p.name for p in DERIVED.iterdir()) if DERIVED.exists() else []:
        if not (GT / f"{v}.stt.txt").exists():
            continue
        for j in sorted(_hyp_dir(v).glob(f"{v}.*.json")):
            partes = j.stem[len(v) + 1:].split(".")
            if len(partes) == 2:
                out.append({"video": v, "modelo": partes[0], "condicao": partes[1]})
    ordem = CONDITIONS + ["limpo_vocab"]
    modelos = [m.removesuffix("-semfiltro") for m in stt.MODELS]
    return sorted(out, key=lambda s: (
        s["video"], modelos.index(b) if (b := s["modelo"].removesuffix("-semfiltro")) in modelos else 99,
        s["modelo"], ordem.index(s["condicao"]) if s["condicao"] in ordem else 99))


def _load_hyp(video: str, modelo: str, cond: str) -> dict:
    for x in (video, modelo, cond):
        _check_name(x)
    arq = _hyp_dir(video) / f"{video}.{modelo}.{cond}.json"
    if not arq.exists():
        raise HTTPException(404, "hipótese não encontrada (rode os experimentos)")
    return json.loads(arq.read_text(encoding="utf-8"))


def _read_errors() -> list[dict]:
    if not ERRORS_CSV.exists():
        return []
    with ERRORS_CSV.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_errors(rows: list[dict]) -> None:
    ERRORS_CSV.parent.mkdir(parents=True, exist_ok=True)
    tmp = ERRORS_CSV.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ERROR_FIELDS)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(ERRORS_CSV)


def _raw_mean(video: str, modelo: str, cond: str, metrica: str) -> dict | None:
    """Média do raw.csv para um vídeo/modelo/condição (o número que a GUI mostra; regra 2)."""
    vals, rtfs = [], []
    ultimo: dict[str, dict] = {}  # repetição refeita vale pela última linha, como em report.load
    for r in _raw_rows():
        if (r["video"], r["tarefa"], r["modelo"], r["condicao"], r["metrica_erro"]) == (video, "stt", modelo, cond, metrica):
            ultimo[r["repeticao"]] = r
    for r in ultimo.values():
        if r["valor_erro"] != "":
            vals.append(float(r["valor_erro"]))
        if r["rtf"] != "":
            rtfs.append(float(r["rtf"]))
    if not vals:
        return None
    return {"mean": sum(vals) / len(vals), "n": len(vals), "rtf": sum(rtfs) / len(rtfs) if rtfs else None}


# ================================================================== configuração e biblioteca
@app.get("/api/config")
def config() -> dict:
    from summarize import DEFAULT_MODELS, backend_status

    return {
        "models": stt.MODELS, "devices": ["auto", "cpu", "cuda"], "conditions": CONDITIONS,
        "resolved_device": stt.resolve_device("auto")[0],
        "vocab_files": sorted(str(p.relative_to(ROOT)) for p in GT.glob("terms*.txt")),
        "has_terms": (GT / "terms.txt").exists(),
        "categories": CATEGORIES,
        "llm": {"ollama": {"model": DEFAULT_MODELS["ollama"], "problem": backend_status("ollama")},
                "claude": {"model": DEFAULT_MODELS["claude"], "problem": backend_status("claude"), "paid": True}},
    }


def _library() -> list[dict]:
    lic, items = _read_licenses(), []
    clips = _read_clips()
    for f in _raw_videos():
        v, wd = f.stem, WEB / "data" / f.stem
        manifest = None
        if (wd / "manifest.json").exists():
            manifest = json.loads((wd / "manifest.json").read_text(encoding="utf-8"))
        thumb = None
        if manifest:
            try:
                sl = json.loads((wd / manifest["slides"]).read_text(encoding="utf-8"))
                thumb = f"data/{v}/{sl[0]['file']}" if sl else None
            except (OSError, ValueError, KeyError):
                pass
        items.append({
            "id": v, "file": f.name, "size_mb": round(f.stat().st_size / 1e6, 1),
            "license": lic.get(f.name), "gt": (GT / f"{v}.stt.txt").exists(), "clip": clips.get(v),
            "status": {"stt": bool(manifest), "slides": bool(manifest), "ocr": False,
                       "summary": bool(manifest and manifest.get("summary"))},
            "thumb": thumb, "viewer": f"index.html?v={v}" if manifest else None,
        })
    return items


@app.get("/api/videos")
def videos() -> list[dict]:
    return _library()


@app.post("/api/videos")
async def upload(file: UploadFile = File(...), license: str = Form(...)) -> dict:  # noqa: B008
    name = _safe_name(file.filename or "video.mp4")
    if Path(name).suffix not in EXTS:
        raise HTTPException(400, f"formato não suportado: {Path(name).suffix}")
    if not license.strip():
        raise HTTPException(400, "informe a licença/origem do vídeo (regra de dados do projeto)")
    RAW.mkdir(parents=True, exist_ok=True)
    dest = RAW / name
    if dest.exists():
        raise HTTPException(409, f"{name} já existe em data/raw/")
    size = 0
    try:
        with dest.open("wb") as fh:
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > MAX_UPLOAD:
                    raise HTTPException(413, "arquivo maior que 4 GB")
                fh.write(chunk)
    except BaseException:
        dest.unlink(missing_ok=True)
        raise
    lic = _read_licenses()
    lic[name] = license.strip()
    LICENSES.write_text(json.dumps(lic, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"file": name}


@app.get("/api/raw/{name}")
def raw_file(name: str) -> FileResponse:
    """O vídeo original, para escolher o trecho do gabarito."""
    f = RAW / _check_name(name)
    if f.suffix.lower() not in EXTS or not f.is_file():
        raise HTTPException(404)
    return FileResponse(f)


# ================================================================== processar (pipeline)
class JobIn(BaseModel):
    video: str
    model: str = "small"
    device: str = "auto"
    vocab: str | None = None
    summary: str | None = None  # None | ollama | claude
    llm_model: str | None = None
    frames_mode: str = "slides"
    step_s: float | None = None
    confirm_paid: bool = False


def _pipeline_cmd(j: JobIn) -> str:
    c = f"uv run python src/pipeline.py data/raw/{j.video} --model {j.model} --device {j.device}"
    if j.vocab:
        c += f" --vocab {j.vocab}"
    if j.summary:
        c += f" --summary {j.summary}"
    if j.llm_model:
        c += f" --llm-model {j.llm_model}"
    if j.frames_mode != "slides":
        c += f" --frames-mode {j.frames_mode}"
    if j.step_s:
        c += f" --step-s {j.step_s}"
    return c


def _check_llm_model(m: str | None) -> None:
    if m and not re.fullmatch(r"[A-Za-z0-9][\w.:/-]*", m):
        raise HTTPException(400, "nome de modelo do LLM inválido")


@app.post("/api/jobs")
def create_job(j: JobIn) -> dict:
    if not _NAME.fullmatch(j.video) or ".." in j.video or not (RAW / j.video).is_file():
        raise HTTPException(404, "vídeo não encontrado em data/raw/")
    if j.model not in stt.MODELS or j.device not in ("auto", "cpu", "cuda"):
        raise HTTPException(400, "modelo ou dispositivo inválido")
    if j.frames_mode not in ("slides", "diff") or j.summary not in (None, "ollama", "claude"):
        raise HTTPException(400, "parâmetro inválido")
    if j.vocab and not (re.fullmatch(r"data/gt/[\w.-]+\.txt", j.vocab) and (ROOT / j.vocab).is_file()):
        raise HTTPException(400, "vocabulário deve ser um .txt em data/gt/")
    _check_llm_model(j.llm_model)
    if j.summary == "claude" and not j.confirm_paid:  # CLAUDE.md, decisão 11: API paga só com confirmação
        raise HTTPException(402, "o backend Claude é pago: marque a confirmação de custo")
    params = j.model_dump(exclude={"video", "confirm_paid"})
    title = f"Processar {j.video} ({j.model}{' + ' + j.summary if j.summary else ''})"
    return _enqueue(Job("pipeline", title, params, _pipeline_cmd(j), video=j.video)).public()


@app.get("/api/jobs")
def list_jobs() -> list[dict]:
    return [j.public() for j in sorted(jobs.values(), key=lambda j: -j.created)]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    if job_id not in jobs:
        raise HTTPException(404)
    return jobs[job_id].public(with_log=True)


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404)
    if job.status == "queued":
        job.status, job.finished = "cancelled", time.time()
    elif job.status == "running":
        if job.kind == "pipeline":
            raise HTTPException(409, "o processamento de um vídeo não pode ser interrompido no meio")
        job.status = "cancelled"
        if job.proc:
            job.proc.terminate()
        job.log.append("Interrompido pelo usuário. Rode de novo para retomar: o que já está no raw.csv é pulado.")
    return job.public()


@app.get("/api/jobs/{job_id}/events")
async def events(job_id: str, since: int = 0) -> StreamingResponse:
    """SSE: eventos `log` (uma linha) e `state` (status/etapa). Fecha quando o trabalho termina."""
    if job_id not in jobs:
        raise HTTPException(404)
    job = jobs[job_id]

    async def gen():
        i, last = since, None
        while True:
            while i < len(job.log):
                yield f"event: log\ndata: {json.dumps(job.log[i], ensure_ascii=False)}\n\n"
                i += 1
            state = json.dumps(job.public())
            if state != last:
                yield f"event: state\ndata: {state}\n\n"
                last = state
            if job.status in ("done", "error", "cancelled") and i >= len(job.log):
                return
            await asyncio.sleep(0.5)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


# ================================================================== experimentos (run_experiments.py)
class ExpIn(BaseModel):
    mode: str = "stt"  # stt (RQ1) | summaries (F9)
    models: list[str] = Field(default_factory=lambda: ["tiny", "base", "small"])
    conditions: list[str] = Field(default_factory=lambda: list(CONDITIONS))
    reps: int | None = None  # padrão do script: 3 no STT, 2 nos resumos
    device: str = "auto"
    vocab: bool = False
    no_clean: bool = False
    redo: bool = False
    backend: str = "ollama"
    llm_model: str | None = None
    confirm_paid: bool = False


def _exp_validate(e: ExpIn) -> int:
    if e.mode not in ("stt", "summaries"):
        raise HTTPException(400, "modo inválido")
    if not e.models or any(m not in stt.MODELS for m in e.models):
        raise HTTPException(400, "modelo inválido")
    if e.mode == "stt" and (not e.conditions or any(c not in CONDITIONS for c in e.conditions)):
        raise HTTPException(400, "condição inválida")
    if e.device not in ("auto", "cpu", "cuda") or e.backend not in ("ollama", "claude"):
        raise HTTPException(400, "dispositivo ou backend inválido")
    _check_llm_model(e.llm_model)
    reps = e.reps or (3 if e.mode == "stt" else 2)
    if not 1 <= reps <= 10:
        raise HTTPException(400, "repetições entre 1 e 10")
    return reps


def _exp_argv(e: ExpIn, reps: int) -> list[str]:
    a = ["src/run_experiments.py", "--models", *e.models, "--reps", str(reps), "--device", e.device]
    if e.mode == "stt":
        a += ["--conditions", *e.conditions]
        a += ["--vocab"] if e.vocab else []
        a += ["--no-clean"] if e.no_clean else []
    else:
        a += ["--summaries", "--backend", e.backend]
        a += ["--llm-model", e.llm_model] if e.llm_model else []
        a += ["--confirm-paid"] if e.confirm_paid else []
    a += ["--redo"] if e.redo else []
    return a


def _exp_plan(e: ExpIn, reps: int) -> dict:
    """O que run_experiments vai rodar, com a mesma regra de pulo (chaves do raw.csv)."""
    feitos = set() if e.redo else _done_keys()
    problems: list[str] = []
    per_model: dict[str, int] = {}
    if e.mode == "stt":
        clips = _read_clips()
        com_gt = sorted(v for v in clips if (GT / f"{v}.stt.txt").exists())
        sem_gt = sorted(set(clips) - set(com_gt))
        sem_video = [v for v in com_gt if not any(p.stem == v for p in _raw_videos())]
        conds = list(e.conditions) + (["limpo_vocab"] if e.vocab else [])
        if e.vocab and not (GT / "terms.txt").exists():
            problems.append("--vocab exige data/gt/terms.txt")
        if not clips:
            problems.append("data/gt/clips.csv não existe ou está vazio (tela Gabarito)")
        elif not com_gt:
            problems.append("nenhum trecho tem gabarito data/gt/<video>.stt.txt (tela Gabarito)")
        if sem_video:
            problems.append("sem o vídeo em data/raw/ (o trecho de áudio ainda não gerado falhará): " + ", ".join(sem_video))
        for m in e.models:
            rot = f"{m}-semfiltro" if e.no_clean else m
            per_model[rot] = sum((v, "stt", rot, c, str(reps), "wer") not in feitos for v in com_gt for c in conds)
        pend = sum(per_model.values())
        return {"mode": "stt", "videos": com_gt, "without_gt": sem_gt, "conditions": conds, "reps": reps,
                "per_model": per_model, "pending": pend, "total": len(e.models) * len(com_gt) * len(conds),
                "transcriptions": pend * reps + sum(1 for n in per_model.values() if n),
                "problems": problems, "cmd": "uv run python " + " ".join(_exp_argv(e, reps))}
    from summarize import backend_status

    vids = [p.stem for p in _raw_videos()]
    llm = f"{e.backend}:{e.llm_model or 'padrao'}"
    if motivo := backend_status(e.backend, e.llm_model):
        problems.append(f"backend {e.backend} indisponível: {motivo}")
    if e.backend == "claude" and not e.confirm_paid:
        problems.append("o backend Claude é pago: marque a confirmação de custo")
    if not vids:
        problems.append("nenhum vídeo em data/raw/")
    for m in e.models:
        per_model[m] = sum((v, "resumo", m, llm, str(r), "tempo_llm") not in feitos
                           for v in vids for r in range(1, reps + 1))
    pend = sum(per_model.values())
    return {"mode": "summaries", "videos": vids, "without_gt": [], "conditions": [llm], "reps": reps,
            "per_model": per_model, "pending": pend, "total": len(e.models) * len(vids) * reps,
            "transcriptions": pend, "problems": problems, "cmd": "uv run python " + " ".join(_exp_argv(e, reps))}


@app.post("/api/experiments/plan")
def exp_plan(e: ExpIn) -> dict:
    return _exp_plan(e, _exp_validate(e))


@app.post("/api/experiments")
def exp_start(e: ExpIn) -> dict:
    reps = _exp_validate(e)
    plan = _exp_plan(e, reps)
    if plan["problems"] and (e.mode == "summaries" or not plan["videos"]):
        raise HTTPException(400, "; ".join(plan["problems"]))
    if e.mode == "summaries" and e.backend == "claude" and not e.confirm_paid:
        raise HTTPException(402, "o backend Claude é pago: marque a confirmação de custo")
    if not plan["pending"]:
        raise HTTPException(409, "nada pendente: tudo já está no raw.csv (marque “refazer” para repetir)")
    total = plan["pending"]  # combinações (STT) ou resumos (F9): o script imprime uma linha por item
    title = ("RQ1 STT × ruído" if e.mode == "stt" else f"F9 resumos ({e.backend})") + f": {', '.join(e.models)}"
    argv = _exp_argv(e, reps)
    return _enqueue(Job("experiment", title, e.model_dump(), "uv run python " + " ".join(argv), argv, total)).public()


@app.post("/api/report")
def run_report() -> dict:
    if not RAW_CSV.exists():
        raise HTTPException(409, "results/raw.csv não existe: rode os experimentos antes")
    argv = ["src/report.py"]
    return _enqueue(Job("report", "Tabelas e figuras (report.py)", {}, "uv run python src/report.py", argv)).public()


# ================================================================== resultados (só lê o raw.csv)
@app.get("/api/results")
def results(video: str | None = None) -> dict:
    import report  # pandas e matplotlib só quando a tela abrir

    out: dict = {"exists": RAW_CSV.exists(), "videos": [], "stt": [], "load": [], "summary": [],
                 "models_order": stt.MODELS, "conditions_order": CONDITIONS + ["limpo_vocab"]}
    hw = RESULTS / "hardware.json"
    out["hardware"] = json.loads(hw.read_text(encoding="utf-8")) if hw.exists() else None
    fig = RESULTS / "figures"
    out["figures"] = sorted(p.name for p in fig.glob("*.png")) if fig.exists() else []
    tab = RESULTS / "tables"
    out["tables"] = sorted(p.name for p in tab.glob("*.md")) if tab.exists() else []
    if not RAW_CSV.exists():
        return out
    df = report.load(RAW_CSV)
    out["videos"] = sorted(set(df.video.astype(str)) - {"_"})
    s = df[df.tarefa == "stt"]
    if video:
        s = s[s.video.astype(str) == video]

    def agg(serie):
        n = int(serie.count())
        return {"mean": _clean(float(serie.mean())) if n else None,
                "sd": _clean(float(serie.std(ddof=1))) if n > 1 else (0.0 if n else None), "n": n}

    w = s[s.metrica_erro == "wer"]
    for (m, c), g in w.groupby(["modelo", "condicao"]):
        linha = {"modelo": m, "condicao": c, "n_videos": int(g.video.nunique()), "wer": agg(g.valor_erro),
                 "rtf": agg(g.rtf), "tempo_s": agg(g.tempo_s)}
        for met in ("cer", "acerto_termos"):
            x = s[(s.metrica_erro == met) & (s.modelo == m) & (s.condicao == c)]
            linha[met] = agg(x.valor_erro) if not x.empty else None
        out["stt"].append(linha)
    carga = df[(df.tarefa == "stt") & (df.metrica_erro == "carga_modelo")]
    out["load"] = [{"modelo": m, "tempo_s": agg(g.tempo_s)} for m, g in carga.groupby("modelo")]
    r = df[df.tarefa == "resumo"]
    if video:
        r = r[r.video.astype(str) == video]
    for (m, c, met), g in r.groupby(["modelo", "condicao", "metrica_erro"]):
        col = g.tempo_s if met == "tempo_llm" else g.valor_erro
        out["summary"].append({"modelo": m, "condicao": c, "metrica": met, **agg(col)})
    return out


@app.get("/api/results/table/{name}")
def result_table(name: str) -> dict:
    f = RESULTS / "tables" / _check_name(name)
    if f.suffix != ".md" or not f.is_file():
        raise HTTPException(404)
    return {"name": name, "markdown": f.read_text(encoding="utf-8")}


@app.get("/api/results/figure/{name}")
def result_figure(name: str) -> FileResponse:
    f = RESULTS / "figures" / _check_name(name)
    if f.suffix != ".png" or not f.is_file():
        raise HTTPException(404)
    return FileResponse(f)


# ================================================================== gabarito (humano; regra 3)
@app.get("/api/gt")
def gt_list() -> list[dict]:
    clips, lic = _read_clips(), _read_licenses()
    raw = {p.stem: p.name for p in _raw_videos()}
    out = []
    for v in sorted(set(raw) | set(clips)):
        f = GT / f"{v}.stt.txt"
        texto = f.read_text(encoding="utf-8") if f.exists() else ""
        out.append({"video": v, "file": raw.get(v), "license": lic.get(raw.get(v, "")), "clip": clips.get(v),
                    "gt_words": len(texto.split()), "has_gt": f.exists(),
                    "clip_audio": (DERIVED / v / f"{v}.clip.limpo.wav").exists()})
    return out


class ClipIn(BaseModel):
    start_s: float
    dur_s: float


@app.put("/api/gt/{video}/clip")
def gt_set_clip(video: str, c: ClipIn) -> dict:
    _check_name(video)
    if c.start_s < 0 or not 10 <= c.dur_s <= 900:
        raise HTTPException(400, "início ≥ 0 e duração entre 10 e 900 s")
    clips = _read_clips()
    mudou = video in clips and clips[video] != (c.start_s, c.dur_s)
    clips[video] = (c.start_s, c.dur_s)
    _write_clips(clips)
    # o áudio do trecho fica em cache em run_experiments.prepare_clip: apagar para regenerar
    for f in (DERIVED / video).glob(f"{video}.clip.*.wav") if (DERIVED / video).exists() else []:
        f.unlink()
    avisos = []
    if mudou and any(r["video"] == video and r["tarefa"] == "stt" for r in _raw_rows()):
        avisos.append("o raw.csv tem resultados do trecho anterior deste vídeo: refaça a RQ1 com “refazer”")
    if mudou and (GT / f"{video}.stt.txt").exists():
        avisos.append("o gabarito existente foi transcrito para o trecho anterior: confira se ainda corresponde")
    return {"clip": clips[video], "warnings": avisos}


@app.post("/api/gt/{video}/clip-audio")
def gt_clip_audio(video: str) -> dict:
    """Gera o áudio limpo do trecho, o mesmo arquivo que run_experiments.prepare_clip usa."""
    from extract import clip_audio

    _check_name(video)
    clips = _read_clips()
    if video not in clips:
        raise HTTPException(409, "defina o trecho antes")
    midia = next((p for p in _raw_videos() if p.stem == video), None)
    if not midia:
        raise HTTPException(404, "vídeo não está em data/raw/")
    destino = DERIVED / video / f"{video}.clip.limpo.wav"
    clip_audio(midia, destino, *clips[video])
    return {"ok": True}


@app.get("/api/gt/{video}/text")
def gt_get_text(video: str) -> dict:
    f = GT / f"{_check_name(video)}.stt.txt"
    return {"text": f.read_text(encoding="utf-8") if f.exists() else "", "exists": f.exists()}


class TextIn(BaseModel):
    text: str


@app.put("/api/gt/{video}/text")
def gt_put_text(video: str, t: TextIn) -> dict:
    f = GT / f"{_check_name(video)}.stt.txt"
    texto = t.text.replace("\r\n", "\n").strip() + "\n"
    GT.mkdir(parents=True, exist_ok=True)
    antigo = f.read_text(encoding="utf-8") if f.exists() else None
    f.write_text(texto, encoding="utf-8")
    avisos = []
    if antigo is not None and antigo != texto and any(r["video"] == video and r["tarefa"] == "stt" for r in _raw_rows()):
        avisos.append("o WER já registrado no raw.csv usou o gabarito anterior: refaça a RQ1 com “refazer”")
    return {"words": len(texto.split()), "warnings": avisos}


@app.get("/api/media/{video}/{cond}")
def media(video: str, cond: str) -> FileResponse:
    """Áudio do trecho: data/derived/<v>/<v>.clip.<cond>.wav (limpo_vocab usa o áudio limpo)."""
    _check_name(video)
    cond = cond.removesuffix(".wav").removesuffix("_vocab")
    if cond not in CONDITIONS:
        raise HTTPException(400, "condição inválida")
    f = DERIVED / video / f"{video}.clip.{cond}.wav"
    if not f.is_file():
        raise HTTPException(404, "áudio do trecho não gerado (rode os experimentos ou gere na tela Gabarito)")
    return FileResponse(f, media_type="audio/wav")


# ================================================================== análise de erros (regra 11)
@app.get("/api/errors/sources")
def err_sources() -> list[dict]:
    return _hyp_sources()


@app.get("/api/errors/{video}/{modelo}/{cond}")
def err_align(video: str, modelo: str, cond: str) -> dict:
    import metrics

    hyp = _load_hyp(video, modelo, cond)
    ref = (GT / f"{video}.stt.txt").read_text(encoding="utf-8")
    texto = " ".join(s["text"] for s in hyp["segments"])  # a mesma hipótese de run_experiments
    chunks = metrics.align(ref, texto)
    labels = {int(r["idx"]): r for r in _read_errors()
              if (r["video"], r["modelo"], r["condicao"]) == (video, modelo, cond)}
    orfaos = []
    for i, c in enumerate(chunks):
        lab = labels.pop(i, None)
        if lab and lab["tipo"] == c["op"] and lab["ref"] == " ".join(c["ref"]) and lab["hyp"] == " ".join(c["hyp"]):
            c["label"] = {"categoria": lab["categoria"], "nota": lab["nota"]}
        elif lab:
            orfaos.append(lab)
    orfaos += list(labels.values())
    contagem = {op: sum(1 for c in chunks if c["op"] == op) for op in ("substitute", "delete", "insert")}
    return {"chunks": chunks, "counts": contagem, "orphans": orfaos, "raw": _raw_mean(video, modelo, cond, "wer"),
            "ref_words": sum(len(c["ref"]) for c in chunks)}


class LabelIn(BaseModel):
    categoria: str | None = None  # None/"" remove a marcação
    nota: str = ""


@app.put("/api/errors/{video}/{modelo}/{cond}/{idx}")
def err_label(video: str, modelo: str, cond: str, idx: int, lab: LabelIn) -> dict:
    import metrics

    hyp = _load_hyp(video, modelo, cond)
    ref = (GT / f"{video}.stt.txt").read_text(encoding="utf-8")
    chunks = metrics.align(ref, " ".join(s["text"] for s in hyp["segments"]))
    if not 0 <= idx < len(chunks) or chunks[idx]["op"] == "equal":
        raise HTTPException(400, "índice não é um erro")
    if lab.categoria and lab.categoria not in CATEGORIES:
        raise HTTPException(400, "categoria inválida")
    chave = (video, modelo, cond, str(idx))
    rows = [r for r in _read_errors() if (r["video"], r["modelo"], r["condicao"], r["idx"]) != chave]
    if lab.categoria:
        c = chunks[idx]
        rows.append({"video": video, "modelo": modelo, "condicao": cond, "idx": idx, "tipo": c["op"],
                     "ref": " ".join(c["ref"]), "hyp": " ".join(c["hyp"]), "categoria": lab.categoria,
                     "nota": lab.nota.replace("\n", " ").strip()[:300]})
    rows.sort(key=lambda r: (r["video"], r["modelo"], r["condicao"], int(r["idx"])))
    _write_errors(rows)
    return {"ok": True}


@app.get("/api/errors/summary")
def err_summary() -> dict:
    rows = _read_errors()
    tab: dict[str, dict[str, int]] = {}
    for r in rows:
        tab.setdefault(r["modelo"], {}).setdefault(r["categoria"], 0)
        tab[r["modelo"]][r["categoria"]] += 1
    return {"by_model": tab, "total": len(rows)}


# ================================================================== A/B de robustez
@app.get("/api/ab/{video}/{modelo}/{cond}")
def ab_side(video: str, modelo: str, cond: str) -> dict:
    hyp = _load_hyp(video, modelo, cond)
    base = cond.removesuffix("_vocab")
    audio = (DERIVED / video / f"{video}.clip.{base}.wav").exists()
    return {"segments": [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in hyp["segments"]],
            "audio": f"/api/media/{video}/{base}" if audio else None,
            "wer": _raw_mean(video, modelo, cond, "wer"), "cer": _raw_mean(video, modelo, cond, "cer")}


# ================================================================== páginas
@app.get("/")
def home() -> RedirectResponse:
    return RedirectResponse("/studio.html")


app.mount("/", StaticFiles(directory=WEB), name="web")


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=a.port)


if __name__ == "__main__":
    main()
