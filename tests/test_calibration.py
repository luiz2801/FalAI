import json

import pytest

import calibration


def _hip(pasta, nome, palavras):
    (pasta / "derived" / "aula01" / "stt").mkdir(parents=True, exist_ok=True)
    seg = {"id": 1, "start": 0, "end": 5, "text": " ".join(w for w, _ in palavras),
           "words": [{"start": i, "end": i + 0.5, "word": w, "probability": p} for i, (w, p) in enumerate(palavras)]}
    (pasta / "derived" / "aula01" / "stt" / nome).write_text(json.dumps({"segments": [seg]}), encoding="utf-8")


@pytest.fixture
def dados(tmp_path):
    (tmp_path / "gt").mkdir()
    (tmp_path / "gt" / "aula01.stt.txt").write_text("o presidente do bacen define as metas", encoding="utf-8")
    _hip(tmp_path, "aula01.tiny.limpo.json", [("o", 0.95), ("presidente", 0.92), ("do", 0.91), ("Bassim.", 0.33),
                                              ("define", 0.99), ("as", 0.45), ("metas", 0.97)])
    return tmp_path


def test_collect_junta_por_faixa_de_probabilidade(dados):
    linhas = calibration.collect(dados / "derived", dados / "gt")
    por_faixa = {(r["bin_lo"]): r for r in linhas}
    assert all((r["modelo"], r["condicao"]) == ("tiny", "limpo") for r in linhas)
    assert por_faixa[0.3]["n"] == 1 and por_faixa[0.3]["acerto"] == 0.0 and por_faixa[0.3]["conf_media"] == 0.33  # Bassim: errada
    assert por_faixa[0.4]["n"] == 1 and por_faixa[0.4]["acerto"] == 1.0  # "as": certa, com p = 0,45
    assert por_faixa[0.9]["n"] == 5 and por_faixa[0.9]["acerto"] == 1.0
    assert sum(r["n"] for r in linhas) == 7


def test_ece_e_a_media_ponderada_do_desvio(dados):
    linhas = calibration.collect(dados / "derived", dados / "gt")
    e = calibration.ece(linhas)[("tiny", "limpo")]
    esperado = (abs(0 - 0.33) + abs(1 - 0.45) + (0.05 + 0.08 + 0.09 + 0.01 + 0.03)) / 7  # |acerto − confiança| por palavra
    assert e == pytest.approx(esperado, abs=2e-3)


def test_collect_ignora_palavras_corrigidas_e_videos_sem_gabarito(dados):
    _hip(dados, "aula01.tiny.limpo@L+F.json", [("o", 0.9), ("BACEN", 0.2)])
    seg = json.loads((dados / "derived" / "aula01" / "stt" / "aula01.tiny.limpo@L+F.json").read_text())
    seg["segments"][0]["words"][1]["orig"] = "Bassim"  # trocada pela correção: a probabilidade é a da palavra antiga
    (dados / "derived" / "aula01" / "stt" / "aula01.tiny.limpo@L+F.json").write_text(json.dumps(seg))
    (dados / "derived" / "aula02" / "stt").mkdir(parents=True)  # sem gabarito: nem é lido
    linhas = [r for r in calibration.collect(dados / "derived", dados / "gt") if r["condicao"] == "limpo@L+F"]
    assert sum(r["n"] for r in linhas) == 1  # só o "o"; o BACEN corrigido não entra
