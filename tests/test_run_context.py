import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pytest

import context_stt
import run_experiments as rx
import stt

SR, DUR, INI = 16000, 140.0, 10.0  # o trecho começa aos 10 s do vídeo: o glossário tem de ser deslocado


def _roteiro():
    r = [(2.0 * i, 2.0 * i + 0.4, f"w{i}", 0.9) for i in range(70)]
    r[10] = (20.0, 20.4, "Bassim", 0.3)
    r[55] = (110.0, 110.4, "Bassim", 0.3)
    r[15] = (30.0, 30.4, "Selic", 0.9)  # termo que NÃO está em nenhum slide
    return r


class AsrFalso:
    def __call__(self, audio, texto):
        a = round(float(audio[0]), 3)
        b = a + len(audio) / SR
        ws = []
        for s, e, w, p in _roteiro():
            if s >= a - 1e-4 and e <= b + 1e-4:
                if texto and w == "Bassim" and "BACEN" in texto:
                    w, p = "BACEN", 0.95
                ws.append({"start": round(s - a, 3), "end": round(e - a, 3), "word": w, "probability": p})
        segs = [{"id": i + 1, "start": g[0]["start"], "end": g[-1]["end"], "text": " ".join(x["word"] for x in g),
                 "avg_logprob": -0.2, "no_speech_prob": 0.0, "words": g}
                for i, g in enumerate(ws[k:k + 3] for k in range(0, len(ws), 3))]
        return segs, 0.5


def _g(slide, t, t_end, termo):
    return {"slide": slide, "t": t + INI, "t_end": t_end + INI, "visits": [[t + INI, t_end + INI]],
            "terms": [{"text": termo, "kind": "sigla", "weight": 1.0, "pron": []}]}


@pytest.fixture
def ambiente(tmp_path, monkeypatch):
    for nome in ("RAW", "GT", "DERIVED", "RESULTS"):
        monkeypatch.setattr(rx, nome, tmp_path / nome.lower())
    monkeypatch.setattr(rx, "RAW_CSV", tmp_path / "results" / "raw.csv")
    (tmp_path / "gt").mkdir()
    (tmp_path / "gt" / "clips.csv").write_text(f"video,start_s,dur_s\naula01,{INI},{DUR}\naula02,{INI},{DUR}\n")
    ref = " ".join("BACEN" if w == "Bassim" else w for _, _, w, _ in _roteiro())
    for v in ("aula01", "aula02"):
        (tmp_path / "gt" / f"{v}.stt.txt").write_text(ref, encoding="utf-8")
        gl = [_g("A", 10, 50, "BACEN"), _g("B", 50, 90, "CMN"), _g("C", 90, 130, "CVM")] if v == "aula01" else \
            [_g("X", 10, 130, "ZZZ")]
        (tmp_path / "derived" / v).mkdir(parents=True)
        (tmp_path / "derived" / v / "glossario.json").write_text(json.dumps(gl), encoding="utf-8")
    (tmp_path / "gt" / "terms.txt").write_text("BACEN\nSelic\n", encoding="utf-8")
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "aula01.mp4").write_bytes(b"")
    (tmp_path / "derived" / "aula01" / "ocr.json").write_text(json.dumps({"tempo_s": 3.0, "glossario_s": 0.05}))
    rampa = np.arange(int(DUR * SR), dtype=np.float32) / SR
    monkeypatch.setattr(rx, "prepare_clip", lambda video, cond, clips: Path(f"{video}.{cond}.wav"))
    monkeypatch.setattr(rx, "load_audio", lambda _p: rampa)
    monkeypatch.setattr(rx, "media_duration", lambda _p: 300.0)
    monkeypatch.setattr(context_stt, "make_asr", lambda *a, **k: AsrFalso())
    monkeypatch.setattr(stt, "load_model", lambda *a, **k: (None, 0.25))
    return tmp_path


def _args(**kw):
    base = {"models": ["tiny"], "conditions": ["limpo"], "reps": 2, "device": "cpu", "context": ["G", "P", "L", "L+F", "O"],
            "via": "hotwords", "margin_s": 2.0, "lead_s": 2.0, "lag_s": 1.0, "no_carry": False, "max_terms": 30,
            "tau": 0.5, "s_min": 0.85, "redo": False, "no_clean": False}
    return argparse.Namespace(**{**base, **kw})


def _linhas(amb):
    with (amb / "results" / "raw.csv").open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _valor(linhas, video, cond, metrica, rep="1"):
    achados = [r for r in linhas if (r["video"], r["condicao"], r["metrica_erro"], r["repeticao"]) == (video, cond, metrica, rep)]
    assert len(achados) == 1, (video, cond, metrica, achados)
    return float(achados[0]["valor_erro"])


def test_run_context_grava_as_condicoes_e_as_metricas_do_contrato(ambiente):
    rx.run_context(_args())
    linhas = _linhas(ambiente)
    conds = {r["condicao"] for r in linhas if r["video"] == "aula01" and r["metrica_erro"] == "wer"}
    assert conds == {"limpo@G", "limpo@P", "limpo@L", "limpo@L+F", "limpo@O"}  # B0 é a RQ1: não é regravado
    # nos 2 vídeos × 5 modos × 2 repetições há uma linha de WER por execução
    assert sum(r["metrica_erro"] == "wer" for r in linhas) == 2 * 5 * 2
    # contexto local acerta o BACEN do slide A (1 de 2); o placebo não muda nada (= B0: 0 de 2); G e O acertam os 2
    assert _valor(linhas, "aula01", "limpo@L", "wer") < _valor(linhas, "aula01", "limpo@P", "wer")
    assert _valor(linhas, "aula01", "limpo@G", "wer") < _valor(linhas, "aula01", "limpo@L", "wer")
    assert _valor(linhas, "aula01", "limpo@L", "acerto_termos_slide") == 0.5  # BACEN: 1 de 2
    assert _valor(linhas, "aula01", "limpo@G", "acerto_termos_slide") == 1.0
    assert _valor(linhas, "aula01", "limpo@L", "acerto_termos_fora") == 1.0  # Selic, que o slide não tem: intacta
    assert _valor(linhas, "aula01", "limpo@L", "acerto_termos") == pytest.approx(2 / 3)
    assert _valor(linhas, "aula01", "limpo@P", "acerto_termos_slide") == 0.0
    assert _valor(linhas, "aula01", "limpo@L", "falsos_positivos") == 0.0
    assert 0 < _valor(linhas, "aula01", "limpo@L", "taxa_alteracao") < 0.05  # 1 palavra em ~70
    assert _valor(linhas, "aula01", "limpo@P", "taxa_alteracao") == 0.0  # o placebo não alterou o texto base
    # L+F: o ASR de teste já acerta o BACEN do slide A com o hotwords, então não sobra o que corrigir ali
    l = [r for r in linhas if r["condicao"] == "limpo@L" and r["metrica_erro"] == "wer" and r["video"] == "aula01"]
    assert [r["repeticao"] for r in l] == ["1", "2"] and float(l[0]["tempo_s"]) == 2.5 and float(l[0]["duracao_s"]) == DUR
    assert float(l[0]["rtf"]) == pytest.approx(2.5 / DUR, abs=1e-5)  # 5 chamadas de 0,5 s: só a inferência entra


def test_run_context_registra_carga_custo_de_ocr_e_guarda_as_hipoteses(ambiente):
    rx.run_context(_args(context=["L+F"], reps=1))
    linhas = _linhas(ambiente)
    carga = [r for r in linhas if r["metrica_erro"] == "carga_modelo"]
    assert len(carga) == 1 and carga[0]["tempo_s"] == "0.25" and carga[0]["video"] == "_"
    ctx = [r for r in linhas if r["metrica_erro"] == "tempo_ctx"]
    assert [(r["video"], float(r["tempo_s"]), float(r["duracao_s"])) for r in ctx] == [("aula01", 3.05, 300.0)]
    ja = ambiente / "derived" / "aula01" / "stt" / "aula01.tiny.limpo@L+F"
    assert Path(f"{ja}.json").exists() and Path(f"{ja}.vtt").exists()
    meta = json.loads(Path(f"{ja}.json").read_text(encoding="utf-8"))
    assert meta["calls"] == 5 and meta["duration_s"] == DUR and meta["plan"][0]["t0"] == 0.0


def test_run_context_retoma_do_raw_csv_e_redo_refaz(ambiente, capsys):
    rx.run_context(_args(context=["L"], reps=1))
    n = len(_linhas(ambiente))
    rx.run_context(_args(context=["L"], reps=1))
    assert len(_linhas(ambiente)) == n and "tudo feito" in capsys.readouterr().out  # nada se repete
    rx.run_context(_args(context=["L"], reps=1, redo=True))
    assert len(_linhas(ambiente)) > n  # --redo grava de novo (a última linha vale, como na RQ1)


def test_run_context_corrige_o_que_o_asr_deixou_com_l_mais_f(ambiente, monkeypatch):
    class SemViés(AsrFalso):
        def __call__(self, audio, texto):
            return super().__call__(audio, None)  # ignora o hotwords: o erro chega ao pós-processamento

    monkeypatch.setattr(context_stt, "make_asr", lambda *a, **k: SemViés())
    rx.run_context(_args(context=["L", "L+F"], reps=1))
    linhas = _linhas(ambiente)
    assert _valor(linhas, "aula01", "limpo@L+F", "wer") < _valor(linhas, "aula01", "limpo@L", "wer")  # a correção ajudou
    correcoes = (ambiente / "derived" / "aula01" / "stt" / "aula01.tiny.limpo@L+F.correcoes.jsonl").read_text(encoding="utf-8")
    c = [json.loads(x) for x in correcoes.splitlines()]
    assert [(x["antes"], x["depois"], x["slide"]) for x in c] == [("Bassim", "BACEN", "A")]  # só no slide que tem o termo


def test_run_context_exige_glossario_oraculo_e_dois_videos_para_o_placebo(ambiente):
    (ambiente / "gt" / "terms.txt").unlink()
    with pytest.raises(SystemExit, match="terms.txt"):
        rx.run_context(_args(context=["O"]))
    (ambiente / "gt" / "terms.txt").write_text("BACEN\n", encoding="utf-8")
    (ambiente / "derived" / "aula02" / "glossario.json").unlink()
    with pytest.raises(SystemExit, match="pelo menos 2 vídeos"):
        rx.run_context(_args(context=["P"]))
    (ambiente / "derived" / "aula01" / "glossario.json").unlink()
    with pytest.raises(SystemExit, match="Nada a rodar"):
        rx.run_context(_args(context=["L"]))
