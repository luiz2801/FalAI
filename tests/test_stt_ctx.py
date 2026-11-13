import sys
import types

import numpy as np

import stt


def _fake_ctranslate2(monkeypatch, gpus):
    monkeypatch.setitem(sys.modules, "ctranslate2", types.SimpleNamespace(get_cuda_device_count=lambda: gpus))


def test_auto_usa_cuda_so_com_gpu_e_cublas(monkeypatch, capsys):
    monkeypatch.setattr(stt, "_avisou_cuda", False)
    _fake_ctranslate2(monkeypatch, gpus=1)
    monkeypatch.setattr(stt, "_cuda_ok", lambda: True)
    assert stt.resolve_device("auto") == ("cuda", "float16")
    # C9: a GPU existe, mas falta o cuBLAS: antes a inferência falhava no meio; agora cai para CPU e avisa uma vez
    monkeypatch.setattr(stt, "_cuda_ok", lambda: False)
    assert stt.resolve_device("auto") == ("cpu", "int8")
    assert "libcublas.so.12" in capsys.readouterr().out
    assert stt.resolve_device("auto") == ("cpu", "int8") and capsys.readouterr().out == ""  # sem repetir o aviso
    _fake_ctranslate2(monkeypatch, gpus=0)
    assert stt.resolve_device("auto") == ("cpu", "int8")
    assert stt.resolve_device("cuda") == ("cuda", "float16") and stt.resolve_device("cpu") == ("cpu", "int8")


def test_cuda_ok_devolve_falso_quando_a_biblioteca_nao_carrega(monkeypatch):
    import ctypes

    def falha(_nome):
        raise OSError("libcublas.so.12: cannot open shared object file")

    monkeypatch.setattr(ctypes, "CDLL", falha)
    assert stt._cuda_ok() is False
    monkeypatch.setattr(ctypes, "CDLL", lambda _nome: object())
    assert stt._cuda_ok() is True


class _ModeloFalso:
    def __init__(self):
        self.chamadas = []

    def transcribe(self, audio, **kw):
        self.chamadas.append({"audio": audio, **kw})
        palavra = types.SimpleNamespace(start=0.0, end=0.5, word=" Bassim", probability=0.33)
        seg = types.SimpleNamespace(id=1, start=0.0, end=0.5, text=" Bassim ", avg_logprob=-0.3,
                                    no_speech_prob=0.01, words=[palavra])
        lixo = types.SimpleNamespace(id=2, start=1.0, end=2.0, text=" Legendas pela comunidade Amara.org",
                                     avg_logprob=-0.9, no_speech_prob=0.8, words=None)
        return iter([seg, lixo]), types.SimpleNamespace(duration=2.0)


def test_transcribe_array_passa_hotwords_e_os_parametros_fixos(monkeypatch):
    modelo = _ModeloFalso()
    monkeypatch.setattr(stt, "load_model", lambda *a, **k: (modelo, 0.0))
    audio = np.zeros(32000, np.float32)
    segs, dur, tempo = stt.transcribe_array(audio, "tiny", "cpu", hotwords="CMN, BACEN")
    kw = modelo.chamadas[0]
    assert kw["audio"] is audio and kw["hotwords"] == "CMN, BACEN" and kw["initial_prompt"] is None
    assert (kw["language"], kw["beam_size"], kw["vad_filter"], kw["condition_on_previous_text"],
            kw["word_timestamps"]) == ("pt", 5, True, False, True)
    assert len(segs) == 2 and dur == 2.0 and tempo >= 0  # transcribe_array NÃO limpa
    assert segs[0]["text"] == "Bassim" and segs[0]["words"] == [
        {"start": 0.0, "end": 0.5, "word": "Bassim", "probability": 0.33}] and segs[1]["words"] is None


def test_transcribe_continua_limpando_e_sem_hotwords_por_padrao(monkeypatch):
    modelo = _ModeloFalso()
    monkeypatch.setattr(stt, "load_model", lambda *a, **k: (modelo, 0.0))
    monkeypatch.setattr(stt, "load_audio", lambda _m: np.zeros(32000, np.float32))
    segs, dur, _ = stt.transcribe("qualquer.mp4", "tiny", "cpu")
    assert [s["text"] for s in segs] == ["Bassim"] and dur == 2.0  # a alucinação conhecida saiu
    assert [r["reason"] for r in stt.transcribe.last_removed] == ["hallucination"]
    assert modelo.chamadas[0]["hotwords"] is None  # o comportamento de antes não muda
    segs2, _, _ = stt.transcribe("qualquer.mp4", "tiny", "cpu", clean=False, hotwords="CMN")
    assert len(segs2) == 2 and modelo.chamadas[1]["hotwords"] == "CMN" and stt.transcribe.last_removed == []
