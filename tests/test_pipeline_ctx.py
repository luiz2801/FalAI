import json
from pathlib import Path

import numpy as np
import pytest
from test_extract import _video_moldura

import context_stt
import extract
import ocr
import pipeline
import stt

SR = 16000


class AsrFalso:
    """Fala 'o presidente do Bassim' com baixa confiança, uma vez a cada 4 s; o hotwords com BACEN corrige."""

    def __init__(self, vies=True):
        self.vies, self.chamadas = vies, []

    def __call__(self, audio, texto):
        a = round(float(audio[0]), 3)
        b = a + len(audio) / SR
        self.chamadas.append(texto)
        ws = []
        for k in range(int(a // 4) + 1, int(b // 4) + 1):
            t = 4.0 * k
            palavra = "BACEN" if (self.vies and texto and "BACEN" in texto) else "Bassim"
            if a <= t and t + 0.4 <= b:
                ws.append({"start": round(t - a, 3), "end": round(t + 0.4 - a, 3), "word": palavra, "probability": 0.3})
        return [{"id": i + 1, "start": w["start"], "end": w["end"], "text": w["word"], "avg_logprob": -0.3,
                 "no_speech_prob": 0.0, "words": [w]} for i, w in enumerate(ws)], 0.1


@pytest.fixture
def ambiente(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "DERIVED", tmp_path / "derived")
    video = _video_moldura(tmp_path)  # 16 s: slide com 4 estágios e um segundo slide
    monkeypatch.setattr(extract, "load_audio", lambda _m: np.arange(int(16.0 * SR), dtype=np.float32) / SR)

    def ocr_falso(_pasta, frames, _box, **_kw):
        return [{**f, "texto": "BACEN – Banco Central do Brasil\nCMN e SFN", "words": [], "ocr_conf": 90.0, "ocr_tempo_s": 0.2}
                for f in frames]

    monkeypatch.setattr(ocr, "status", lambda *_a, **_k: None)
    monkeypatch.setattr(ocr, "ocr_slides", ocr_falso)
    return video


def test_context_roda_slides_ocr_glossario_e_stt_com_contexto(ambiente, tmp_path, monkeypatch):
    asr = AsrFalso()
    monkeypatch.setattr(context_stt, "make_asr", lambda *a, **k: asr)
    msgs: list[str] = []
    out = pipeline.process(ambiente, "tiny", "cpu", context="L", web=False, progress=msgs.append)
    pasta = tmp_path / "derived" / "moldura"
    assert msgs[0].startswith("Slides")  # no F10 os slides vêm antes do STT: o glossário alimenta a decodificação
    assert any(m.startswith("OCR (tesseract)") for m in msgs) and any("contexto L" in m for m in msgs)
    assert all("texto" in f and "BACEN" in f["texto"] for f in out["frames"])  # o texto do OCR vai para os slides da página
    gl = json.loads((pasta / "glossario.json").read_text(encoding="utf-8"))
    assert gl and {t["text"] for t in gl[0]["terms"]} >= {"BACEN", "CMN", "SFN"}
    assert json.loads((pasta / "ocr.json").read_text(encoding="utf-8"))["motor"] == "tesseract"
    assert {w["word"] for s in out["segments"] for w in s["words"]} == {"BACEN"}  # o hotwords do slide chegou ao ASR
    assert any(t and "BACEN" in t for t in asr.chamadas)
    assert (pasta / "moldura.tiny.ctx-L.vtt").exists()  # o contexto vai no nome: não sobrescreve a transcrição sem contexto
    meta = json.loads((pasta / "moldura.tiny.ctx-L.json").read_text(encoding="utf-8"))
    assert meta["context"] == "L" and meta["calls"] >= 1 and out["duration_s"] == 16.0


def test_context_l_mais_f_corrige_e_grava_o_registro(ambiente, tmp_path, monkeypatch):
    monkeypatch.setattr(context_stt, "make_asr", lambda *a, **k: AsrFalso(vies=False))  # o ASR erra; a correção conserta
    out = pipeline.process(ambiente, "tiny", "cpu", context="L+F", web=False, progress=lambda _m: None)
    assert out["corrections"] and {c["depois"] for c in out["corrections"]} == {"BACEN"}
    linhas = (tmp_path / "derived" / "moldura" / "moldura.tiny.ctx-LF.correcoes.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(linhas) == len(out["corrections"]) and json.loads(linhas[0])["antes"] == "Bassim"


def test_sem_tesseract_o_contexto_e_pulado_com_o_motivo_e_o_stt_segue(ambiente, monkeypatch):
    monkeypatch.setattr(ocr, "status", lambda *_a, **_k: "o Tesseract não está no PATH (Arch: sudo pacman -S tesseract)")
    chamado = {}

    def stt_falso(media, size, device, initial_prompt=None, **_kw):
        chamado["ok"] = True
        return [{"id": 1, "start": 0, "end": 1, "text": "oi", "words": None}], 16.0, 1.0

    stt_falso.last_removed = []
    monkeypatch.setattr(stt, "transcribe", stt_falso)
    msgs: list[str] = []
    out = pipeline.process(ambiente, "tiny", "cpu", context="L+F", web=False, progress=msgs.append)
    assert chamado["ok"] and any(m.startswith("OCR pulado") and "o contexto também" in m for m in msgs)
    assert "corrections" not in out and not any("texto" in f for f in out["frames"])


def test_padrao_nao_muda_sem_context_nem_ocr(ambiente, tmp_path, monkeypatch):
    monkeypatch.setattr(ocr, "ocr_slides", lambda *a, **k: pytest.fail("o OCR não devia rodar"))
    monkeypatch.setattr(context_stt, "make_asr", lambda *a, **k: pytest.fail("o contexto não devia rodar"))
    def stt_falso(media, size, device, initial_prompt=None, **_kw):
        return [{"id": 1, "start": 0, "end": 1, "text": "oi", "words": None}], 16.0, 1.0

    stt_falso.last_removed = []
    monkeypatch.setattr(stt, "transcribe", stt_falso)
    out = pipeline.process(ambiente, "tiny", "cpu", web=False, progress=lambda _m: None)
    pasta = tmp_path / "derived" / "moldura"
    assert (pasta / "moldura.tiny.vtt").exists() and not (pasta / "glossario.json").exists()
    assert len(out["frames"]) == 2 and "texto" not in out["frames"][0]


def test_ocr_sozinho_preenche_o_texto_dos_slides_sem_contexto(ambiente, tmp_path, monkeypatch):
    monkeypatch.setattr(context_stt, "make_asr", lambda *a, **k: pytest.fail("sem contexto"))
    def stt_falso(media, size, device, initial_prompt=None, **_kw):
        return [{"id": 1, "start": 0, "end": 1, "text": "oi", "words": None}], 16.0, 1.0

    stt_falso.last_removed = []
    monkeypatch.setattr(stt, "transcribe", stt_falso)
    out = pipeline.process(ambiente, "tiny", "cpu", ocr=True, web=False, progress=lambda _m: None)
    assert all("BACEN" in f["texto"] for f in out["frames"]) and Path(tmp_path / "derived" / "moldura" / "glossario.json").exists()


def test_modo_diff_pula_ocr_e_contexto(ambiente, monkeypatch):
    def stt_falso(media, size, device, initial_prompt=None, **_kw):
        return [{"id": 1, "start": 0, "end": 1, "text": "oi", "words": None}], 16.0, 1.0

    stt_falso.last_removed = []
    monkeypatch.setattr(stt, "transcribe", stt_falso)
    msgs: list[str] = []
    out = pipeline.process(ambiente, "tiny", "cpu", context="L", web=False, progress=msgs.append, frames_mode="diff")
    assert any("exigem o modo de slides" in m for m in msgs) and "corrections" not in out
