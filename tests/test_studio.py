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
                        "RAW_CSV": res / "raw.csv", "ERRORS_CSV": res / "erros.csv",
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
