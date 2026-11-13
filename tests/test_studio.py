"""Testes da API do Studio (sem carregar modelos nem iniciar o trabalhador real)."""

import csv
import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("multipart")
from fastapi.testclient import TestClient

import run_experiments
import studio

FIELDS = run_experiments.FIELDS


@pytest.fixture
def env(tmp_path, monkeypatch):
    raw, gt, der, res, web = (tmp_path / n for n in ("raw", "gt", "derived", "results", "web"))
    for d in (raw, gt, der, res, web):
        d.mkdir()
    (raw / "aula01.mp4").write_bytes(b"x")
    for nome, valor in {"RAW": raw, "GT": gt, "DERIVED": der, "RESULTS": res, "WEB": web,
                        "RAW_CSV": res / "raw.csv", "ERRORS_CSV": res / "erros.csv", "AUDIT_CSV": res / "correcoes_auditoria.csv",
                        "LICENSES": raw / "licenses.json"}.items():
        monkeypatch.setattr(studio, nome, valor)
    monkeypatch.setattr(studio, "_workers", [object()])  # impede iniciar o trabalhador real
    studio.jobs.clear()
    return tmp_path


@pytest.fixture
def client(env):
    return TestClient(studio.app)


def _raw_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(dict(zip(FIELDS, r, strict=True)))


def _hyp(env, video, modelo, cond, texto):
    d = env / "derived" / video / "stt"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{video}.{modelo}.{cond}.json").write_text(json.dumps(
        {"segments": [{"start": 0.0, "end": 2.0, "text": texto}]}), encoding="utf-8")


def test_constants_match_run_experiments():
    assert studio.CONDITIONS == list(run_experiments.CONDITIONS)
    assert studio.CONTEXTS == run_experiments.CONTEXTS
    assert studio.EXTS == run_experiments.VIDEO_EXT


def test_safe_name():
    assert studio._safe_name("../../etc/Aula 01!.MP4") == "Aula_01.mp4"


def test_library_lists_videos(client):
    v = client.get("/api/videos").json()
    assert [x["file"] for x in v] == ["aula01.mp4"] and v[0]["license"] is None


def test_upload_requires_license_and_known_format(client):
    f = {"file": ("nova.mp4", b"abc")}
    assert client.post("/api/videos", files=f, data={"license": "  "}).status_code == 400
    assert client.post("/api/videos", files={"file": ("a.exe", b"abc")}, data={"license": "x"}).status_code == 400
    assert client.post("/api/videos", files=f, data={"license": "CC BY 4.0"}).status_code == 200
    assert client.post("/api/videos", files=f, data={"license": "CC BY 4.0"}).status_code == 409
    assert {x["file"]: x["license"] for x in client.get("/api/videos").json()}["nova.mp4"] == "CC BY 4.0"


def test_job_validation(client):
    ok = {"video": "aula01.mp4"}
    assert client.post("/api/jobs", json={"video": "../x.mp4"}).status_code == 404
    assert client.post("/api/jobs", json={**ok, "model": "gigante"}).status_code == 400
    assert client.post("/api/jobs", json={**ok, "vocab": "../../etc/passwd"}).status_code == 400
    assert client.post("/api/jobs", json={**ok, "llm_model": "--rm"}).status_code == 400
    assert client.post("/api/jobs", json={**ok, "summary": "claude"}).status_code == 402  # API paga
    r = client.post("/api/jobs", json={**ok, "summary": "claude", "confirm_paid": True})
    assert r.status_code == 200 and "--summary claude" in r.json()["cmd"]


def test_cancel_queued_job(client):
    j = client.post("/api/jobs", json={"video": "aula01.mp4"}).json()
    assert client.post(f"/api/jobs/{j['id']}/cancel").status_code == 200
    assert client.get(f"/api/jobs/{j['id']}").json()["status"] == "cancelled"


def test_stage_and_progress_tracking():
    j = studio.Job("pipeline", "t", {}, "")
    j.say("STT (small)...")
    j.say("  12 s de áudio")
    assert j.stage == "stt"
    j.say("Slides completos...")
    assert j.stage == "frames"
    e = studio.Job("experiment", "t", {}, "", total=2)
    e.say("[10:00:00] tiny     aula01 limpo        WER=0.100 RTF=0.010")
    e.say("tiny: tudo feito")
    assert e.done == 1


def test_experiment_plan_skips_done(env, client):
    (env / "gt" / "clips.csv").write_text("video,start_s,dur_s\naula01,0,120\naula02,0,120\n", encoding="utf-8")
    (env / "gt" / "aula01.stt.txt").write_text("olá mundo", encoding="utf-8")
    _raw_csv(env / "results" / "raw.csv", [("aula01", "stt", "tiny", "limpo", 3, "wer", 0.1, 1, 10, 0.1)])
    p = client.post("/api/experiments/plan", json={"models": ["tiny"], "conditions": ["limpo", "snr0"], "reps": 3}).json()
    assert p["videos"] == ["aula01"] and p["without_gt"] == ["aula02"]
    assert p["per_model"] == {"tiny": 1} and p["total"] == 2
    assert p["transcriptions"] == 3 + 1  # 3 repetições + aquecimento
    assert "--conditions limpo snr0" in p["cmd"]
    p = client.post("/api/experiments/plan", json={"models": ["tiny"], "conditions": ["limpo"], "reps": 3,
                                                   "no_clean": True}).json()
    assert p["per_model"] == {"tiny-semfiltro": 1}
    assert client.post("/api/experiments/plan", json={"models": ["x"]}).status_code == 400
    j = client.post("/api/experiments", json={"models": ["tiny"], "conditions": ["snr0"], "reps": 3}).json()
    assert j["total"] == 1 and j["kind"] == "experiment"
    assert client.post("/api/experiments", json={"models": ["tiny"], "conditions": ["limpo"], "reps": 3}).status_code == 409


def test_results_aggregate_from_raw_csv(env, client):
    assert client.get("/api/results").json()["exists"] is False
    _raw_csv(env / "results" / "raw.csv", [
        ("aula01", "stt", "tiny", "limpo", 1, "wer", 0.2, 1.0, 10, 0.1),
        ("aula01", "stt", "tiny", "limpo", 2, "wer", 0.4, 1.0, 10, 0.1),
        ("aula01", "stt", "tiny", "limpo", 2, "wer", 0.3, 1.0, 10, 0.1),  # refeita: vale a última
        ("aula01", "stt", "tiny", "limpo", 1, "cer", 0.1, 1.0, 10, 0.1),
        ("_", "stt", "tiny", "carga", 1, "carga_modelo", "", 2.5, "", ""),
    ])
    r = client.get("/api/results").json()
    (linha,) = r["stt"]
    assert linha["wer"]["n"] == 2 and linha["wer"]["mean"] == pytest.approx(0.25)
    assert linha["cer"]["n"] == 1 and linha["cer"]["sd"] == 0.0
    assert r["load"][0]["tempo_s"]["mean"] == 2.5 and r["videos"] == ["aula01"]
    assert client.get("/api/results?video=outro").json()["stt"] == []


def test_gt_clip_and_text(env, client):
    der = env / "derived" / "aula01"
    der.mkdir()
    (der / "aula01.clip.limpo.wav").write_bytes(b"x")
    r = client.put("/api/gt/aula01/clip", json={"start_s": 30, "dur_s": 150}).json()
    assert r["clip"] == [30, 150] and not (der / "aula01.clip.limpo.wav").exists()  # cache invalidado
    assert "aula01,30,150" in (env / "gt" / "clips.csv").read_text(encoding="utf-8")
    assert client.put("/api/gt/aula01/clip", json={"start_s": 0, "dur_s": 1}).status_code == 400
    assert client.put("/api/gt/.x/clip", json={"start_s": 0, "dur_s": 60}).status_code == 400
    assert client.put("/api/gt/aula01/text", json={"text": "olá  mundo\r\n"}).json()["words"] == 2
    assert client.get("/api/gt/aula01/text").json()["text"] == "olá  mundo\n"
    _raw_csv(env / "results" / "raw.csv", [("aula01", "stt", "tiny", "limpo", 1, "wer", 0.1, 1, 10, 0.1)])
    assert client.put("/api/gt/aula01/text", json={"text": "outro texto"}).json()["warnings"]  # WER antigo
    g = client.get("/api/gt").json()
    assert g[0]["has_gt"] and g[0]["clip"] == [30, 150]


def test_error_alignment_and_labels(env, client):
    (env / "gt" / "aula01.stt.txt").write_text("a casa azul é bonita demais", encoding="utf-8")
    _hyp(env, "aula01", "tiny", "snr0", "a cada azul bonita demais sim")
    assert client.get("/api/errors/sources").json() == [{"video": "aula01", "modelo": "tiny", "condicao": "snr0"}]
    d = client.get("/api/errors/aula01/tiny/snr0").json()
    ops = [c["op"] for c in d["chunks"]]
    assert ops == ["equal", "substitute", "equal", "delete", "equal", "insert"]
    assert d["counts"] == {"substitute": 1, "delete": 1, "insert": 1}
    assert client.put("/api/errors/aula01/tiny/snr0/0", json={"categoria": "ruido"}).status_code == 400  # equal
    assert client.put("/api/errors/aula01/tiny/snr0/1", json={"categoria": "xyz"}).status_code == 400
    assert client.put("/api/errors/aula01/tiny/snr0/1", json={"categoria": "ruido", "nota": "chiado"}).status_code == 200
    d = client.get("/api/errors/aula01/tiny/snr0").json()
    assert d["chunks"][1]["label"] == {"categoria": "ruido", "nota": "chiado"}
    assert client.get("/api/errors/summary").json() == {"by_model": {"tiny": {"ruido": 1}}, "total": 1}
    # o gabarito muda: a marcação vira desatualizada em vez de cair num erro diferente
    (env / "gt" / "aula01.stt.txt").write_text("a coisa azul é bonita demais", encoding="utf-8")
    d = client.get("/api/errors/aula01/tiny/snr0").json()
    assert "label" not in d["chunks"][1] and d["orphans"][0]["ref"] == "casa"
    assert client.get("/api/errors/summary").json()["total"] == 1  # a órfã continua no CSV
    client.put("/api/errors/aula01/tiny/snr0/1", json={"categoria": "nome_proprio"})  # refeita: substitui
    assert client.get("/api/errors/summary").json() == {"by_model": {"tiny": {"nome_proprio": 1}}, "total": 1}
    assert client.get("/api/errors/aula01/tiny/snr0").json()["orphans"] == []


def test_ab_and_media(env, client):
    (env / "gt" / "aula01.stt.txt").write_text("olá", encoding="utf-8")
    _hyp(env, "aula01", "tiny", "limpo_vocab", "olá")
    d = client.get("/api/ab/aula01/tiny/limpo_vocab").json()
    assert d["audio"] is None and d["segments"][0]["text"] == "olá"
    (env / "derived" / "aula01" / "aula01.clip.limpo.wav").write_bytes(b"RIFF")
    assert client.get("/api/ab/aula01/tiny/limpo_vocab").json()["audio"] == "/api/media/aula01/limpo"
    assert client.get("/api/media/aula01/limpo").status_code == 200
    assert client.get("/api/media/aula01/snr7").status_code == 400
    assert client.get("/api/ab/aula01/tiny/naoexiste").status_code == 404


# --- F10: o slide como contexto -----------------------------------------------------------------------------
def test_job_validation_context(client):
    ok = {"video": "aula01.mp4"}
    assert client.post("/api/jobs", json={**ok, "context": "X"}).status_code == 400
    assert client.post("/api/jobs", json={**ok, "context": "L", "frames_mode": "diff"}).status_code == 400
    r = client.post("/api/jobs", json={**ok, "context": "L+F"}).json()
    assert "--context L+F" in r["cmd"] and "contexto L+F" in r["title"] and r["params"]["context"] == "L+F"
    assert "--ocr" in client.post("/api/jobs", json={**ok, "ocr": True}).json()["cmd"]


def test_stage_ocr_e_biblioteca(env, client):
    j = studio.Job("pipeline", "t", {}, "")
    j.say("Slides completos...")
    j.say("OCR (tesseract) em 4 slides...")
    assert j.stage == "ocr"
    j.say("STT (small, contexto L)...")
    assert j.stage == "stt"
    st = client.get("/api/videos").json()[0]["status"]
    assert st["ocr"] is False and st["glossary"] is False
    (env / "derived" / "aula01").mkdir(parents=True)
    (env / "derived" / "aula01" / "ocr.json").write_text("{}", encoding="utf-8")
    (env / "derived" / "aula01" / "glossario.json").write_text("[]", encoding="utf-8")
    st = client.get("/api/videos").json()[0]["status"]
    assert st["ocr"] is True and st["glossary"] is True
    assert client.get("/api/config").json()["contexts"] == studio.CONTEXTS


def _preparar_contexto(env, videos=("aula01", "aula02")):
    (env / "gt" / "clips.csv").write_text("".join(["video,start_s,dur_s\n"] + [f"{v},0,120\n" for v in videos]), encoding="utf-8")
    for v in videos:
        (env / "gt" / f"{v}.stt.txt").write_text("olá mundo", encoding="utf-8")
        (env / "derived" / v).mkdir(parents=True, exist_ok=True)
        (env / "derived" / v / "glossario.json").write_text("[]", encoding="utf-8")


def test_experiment_plan_context(env, client):
    _preparar_contexto(env)
    _raw_csv(env / "results" / "raw.csv", [("aula01", "stt", "tiny", "limpo@L", 3, "wer", 0.1, 1, 10, 0.1)])
    corpo = {"mode": "context", "models": ["tiny"], "conditions": ["limpo", "snr10"], "contexts": ["B0", "L"], "reps": 3}
    p = client.post("/api/experiments/plan", json=corpo).json()
    assert p["mode"] == "context" and p["videos"] == ["aula01", "aula02"] and p["total"] == 2 * 2 * 2
    assert p["pending"] == 7 and p["per_model"] == {"tiny": 7}  # aula01 limpo@L já está no raw.csv
    assert p["conditions"] == ["limpo@B0", "limpo@L", "snr10@B0", "snr10@L"] and p["problems"] == []
    assert "--context B0 L" in p["cmd"] and "--conditions limpo snr10" in p["cmd"]
    assert p["transcriptions"] == 7 * 3 + 1
    j = client.post("/api/experiments", json=corpo).json()
    assert j["total"] == 7 and j["title"].startswith("F10 contexto do slide")
    assert client.post("/api/experiments/plan", json={**corpo, "contexts": ["Z"]}).status_code == 400
    assert client.post("/api/experiments/plan", json={**corpo, "contexts": []}).status_code == 400


def test_experiment_plan_context_avisa_o_que_falta(env, client):
    _preparar_contexto(env)
    (env / "derived" / "aula02" / "glossario.json").unlink()
    p = client.post("/api/experiments/plan", json={"mode": "context", "models": ["tiny"], "conditions": ["limpo"],
                                                   "contexts": ["P", "O"]}).json()
    assert p["videos"] == ["aula01"] and p["without_glossary"] == ["aula02"]
    msgs = " | ".join(p["problems"])
    assert "src/context.py glossary" in msgs and "terms.txt" in msgs and "pelo menos 2 vídeos" in msgs
    r = client.post("/api/experiments", json={"mode": "context", "models": ["tiny"], "conditions": ["limpo"], "contexts": ["P"]})
    assert r.status_code == 200  # há vídeo com glossário: o script reclama do que faltar, o plano já avisou


def test_results_separam_a_rq1_do_contexto(env, client):
    _raw_csv(env / "results" / "raw.csv", [
        ("aula01", "stt", "tiny", "limpo", 1, "wer", 0.20, 1, 10, 0.5),
        ("aula01", "stt", "tiny", "limpo@L", 1, "wer", 0.15, 2, 10, 0.9),
        ("aula01", "stt", "tiny", "limpo@L", 1, "acerto_termos_slide", 0.7, 2, 10, 0.9),
        ("aula01", "stt", "tiny", "limpo@L", 2, "acerto_termos_slide", 0.5, 2, 10, 0.9),
        ("_", "stt", "tiny", "carga", 1, "carga_modelo", None, 3.0, None, None)])
    d = client.get("/api/results").json()
    assert [(r["modelo"], r["condicao"]) for r in d["stt"]] == [("tiny", "limpo")]  # a tabela de tempo da RQ1 não se mistura
    assert d["stt"][0]["rtf"]["mean"] == 0.5
    ctx = {(r["condicao"], r["metrica"]): r for r in d["ctx"]}
    assert ctx[("limpo@L", "wer")]["mean"] == 0.15 and ctx[("limpo@L", "wer")]["rtf"]["mean"] == 0.9
    assert ctx[("limpo@L", "acerto_termos_slide")]["mean"] == 0.6 and ctx[("limpo@L", "acerto_termos_slide")]["n"] == 2
    assert d["contexts_order"] == studio.CONTEXTS


def test_erros_e_ab_aceitam_condicoes_de_contexto(env, client):
    (env / "gt" / "aula01.stt.txt").write_text("o presidente do BACEN define", encoding="utf-8")
    _hyp(env, "aula01", "tiny", "limpo", "o presidente do Bassim define")
    _hyp(env, "aula01", "tiny", "snr10@L+F", "o presidente do BACEN define")
    _hyp(env, "aula01", "tiny", "limpo@L", "o presidente do BACEN define")
    srcs = client.get("/api/errors/sources").json()
    assert [s["condicao"] for s in srcs] == ["limpo", "limpo@L", "snr10@L+F"]  # sem contexto, depois pelo ruído
    d = client.get("/api/errors/aula01/tiny/snr10@L+F").json()
    assert d["counts"] == {"substitute": 0, "delete": 0, "insert": 0}
    assert client.get("/api/errors/aula01/tiny/limpo@L").status_code == 200
    (env / "derived" / "aula01" / "aula01.clip.snr10.wav").write_bytes(b"RIFF")
    ab = client.get("/api/ab/aula01/tiny/snr10@L+F").json()
    assert ab["audio"] == "/api/media/aula01/snr10"  # o áudio de uma condição de contexto é o do seu ruído
    assert client.get("/api/media/aula01/snr10@L+F").status_code == 200
    assert client.get("/api/media/aula01/snr7@L").status_code == 400
    assert client.get("/api/errors/aula01/tiny/..%2Fx@L").status_code in (400, 404)
    assert client.get("/api/errors/aula01/tiny/limpo@L;rm").status_code == 400


def _correcoes(env, video="aula01", modelo="tiny", cond="limpo@L+F", itens=None):
    itens = itens if itens is not None else [
        {"t": 21.3, "antes": "Bassé", "depois": "BACEN", "p": 0.41, "sim": 1.0, "regra": "R1", "slide": "0000.png",
         "termo": "BACEN", "seg_antes": "o presidente do Bassé", "seg_depois": "o presidente do BACEN"},
        {"t": 40.0, "antes": "cml", "depois": "CMN", "p": 0.15, "sim": 0.67, "regra": "R2", "slide": "0000.png",
         "termo": "CMN", "seg_antes": "papel de cml", "seg_depois": "papel de CMN"}]
    d = env / "derived" / video / "stt"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{video}.{modelo}.{cond}.correcoes.jsonl").write_text("\n".join(json.dumps(i) for i in itens) + "\n", encoding="utf-8")


def test_auditoria_das_correcoes_de_ponta_a_ponta(env, client):
    assert client.get("/api/ctx/corrections").json()["total"] == 0
    _correcoes(env)
    (env / "derived" / "aula01" / "aula01.clip.limpo.wav").write_bytes(b"RIFF")
    d = client.get("/api/ctx/corrections").json()
    assert d["total"] == 2 and d["precision"] is None and d["items"][0]["auditoria"] is None
    assert d["items"][0]["audio"] == "/api/media/aula01/limpo" and d["items"][0]["seg_depois"] == "o presidente do BACEN"
    url = "/api/ctx/corrections/aula01/tiny/limpo@L+F"
    assert client.put(f"{url}/0", json={"auditoria": "certa"}).status_code == 200
    assert client.put(f"{url}/1", json={"auditoria": "errada", "nota": "era \"cml\" mesmo\nem duas linhas"}).status_code == 200
    d = client.get("/api/ctx/corrections").json()
    assert d["counts"] == {"certa": 1, "errada": 1, "duvida": 0} and d["precision"] == 0.5
    assert d["items"][1]["nota"] == 'era "cml" mesmo em duas linhas'
    assert client.put(f"{url}/1", json={"auditoria": "duvida"}).status_code == 200  # não sei: fora da precisão
    d = client.get("/api/ctx/corrections").json()
    assert d["precision"] == 1.0 and d["counts"]["duvida"] == 1
    assert client.put(f"{url}/1", json={"auditoria": None}).status_code == 200  # limpar
    assert client.get("/api/ctx/corrections").json()["items"][1]["auditoria"] is None
    assert client.put(f"{url}/9", json={"auditoria": "certa"}).status_code == 400
    assert client.put(f"{url}/0", json={"auditoria": "talvez"}).status_code == 400
    assert client.put("/api/ctx/corrections/aula01/tiny/limpo@G/0", json={"auditoria": "certa"}).status_code == 404
    with (env / "results" / "correcoes_auditoria.csv").open(encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))
    assert [(r["video"], r["condicao"], r["idx"], r["auditoria"]) for r in linhas] == [("aula01", "limpo@L+F", "0", "certa")]


def test_auditoria_marca_como_desatualizada_a_correcao_que_mudou(env, client):
    _correcoes(env)
    url = "/api/ctx/corrections/aula01/tiny/limpo@L+F"
    client.put(f"{url}/0", json={"auditoria": "certa"})
    _correcoes(env, itens=[{"t": 21.3, "antes": "Bassim", "depois": "BACEN", "p": 0.3, "sim": 1.0, "regra": "R1",
                            "slide": "0000.png", "termo": "BACEN"}])  # a RQ6 foi refeita: a troca é outra
    it = client.get("/api/ctx/corrections").json()["items"][0]
    assert it["auditoria"] is None and it["desatualizada"] is True  # a marcação antiga não vale para a troca nova


def test_auditoria_correcoes_do_pipeline_usam_o_audio_do_video(env, client):
    d = env / "derived" / "aula01"
    d.mkdir(parents=True)
    (d / "aula01.small.ctx-LF.correcoes.jsonl").write_text(json.dumps(
        {"t": 5.0, "antes": "Bassim", "depois": "BACEN", "p": 0.3, "sim": 1.0, "regra": "R1", "slide": "0000.png",
         "termo": "BACEN"}) + "\n", encoding="utf-8")
    it = client.get("/api/ctx/corrections").json()["items"][0]
    assert (it["condicao"], it["origem"], it["audio"]) == ("pipeline@L+F", "pipeline", "/api/raw/aula01.mp4")
    assert client.put("/api/ctx/corrections/aula01/small/pipeline@L+F/0", json={"auditoria": "errada"}).status_code == 200
