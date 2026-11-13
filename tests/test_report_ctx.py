import pandas as pd
import pytest

import report

# valor médio do acerto de termos do slide por (ruído, modo): o ganho do local sobre o B0 é de 50 pp no limpo e 30 pp a 10 dB
ACERTO = {("limpo", "B0"): 0.2, ("limpo", "P"): 0.2, ("limpo", "G"): 0.6, ("limpo", "L"): 0.7, ("limpo", "L+F"): 0.8,
          ("limpo", "O"): 0.9, ("snr10", "B0"): 0.1, ("snr10", "L"): 0.4}
RTF = {"B0": 0.10, "G": 0.10, "P": 0.12, "L": 0.14, "L+F": 0.14, "O": 0.10}


def _linha(video, modelo, cond, metrica, valor, rtf=0.1):
    return {"video": video, "tarefa": "stt", "modelo": modelo, "condicao": cond, "repeticao": 1, "metrica_erro": metrica,
            "valor_erro": valor, "tempo_s": rtf * 100, "duracao_s": 100.0, "rtf": rtf}


@pytest.fixture
def stt():
    linhas = []
    for modelo in ("tiny", "base"):
        for video in ("v1", "v2"):
            for (ruido, modo), v in ACERTO.items():
                cond = f"{ruido}@{modo}"
                linhas += [_linha(video, modelo, cond, "acerto_termos_slide", v, RTF[modo]),
                           _linha(video, modelo, cond, "wer", 0.3 - v / 10, RTF[modo]),
                           _linha(video, modelo, cond, "taxa_alteracao", 0.0 if modo == "B0" else 0.02, RTF[modo])]
            linhas.append(_linha(video, modelo, "limpo", "wer", 0.31, 0.5))  # a RQ1, que não pode entrar nas tabelas de contexto
    linhas += [_linha(v, "_", "ctx", "tempo_ctx", None, 0.01) for v in ("v1", "v2")]
    return pd.DataFrame(linhas)


@pytest.fixture
def saida(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "RESULTS", tmp_path)
    monkeypatch.setattr(report, "TABLES", tmp_path / "tables")
    monkeypatch.setattr(report, "FIGURES", tmp_path / "figures")
    return tmp_path


def test_tabela6_media_desvio_n_por_modelo_ruido_metrica_e_modo(stt, saida):
    report.table6_context(stt)
    t = pd.read_csv(saida / "tables" / "tabela6_contexto.csv", index_col=[0, 1, 2])
    assert list(t.columns) == ["B0", "G", "P", "L", "L+F", "O"]  # ordem fixa dos modos
    assert t.loc[("tiny", "limpo", "acerto_termos_slide"), "L"] == "0.700 ± 0.000 (n=2)"
    assert t.loc[("base", "snr10", "acerto_termos_slide"), "B0"] == "0.100 ± 0.000 (n=2)"
    assert "taxa_alteracao" in t.index.get_level_values(2) and "|" in (saida / "tables" / "tabela6_contexto.md").read_text()


def test_tabela7_custo_soma_o_ocr_so_nos_modos_que_usam_slide(stt, saida):
    report.table7_context_cost(stt)
    t = pd.read_csv(saida / "tables" / "tabela7_custo_contexto.csv", index_col=[0, 1, 2])
    l = t.loc[("tiny", "limpo", "L")]
    assert l.rtf_inferencia == pytest.approx(0.14) and l.delta_rtf_inferencia == pytest.approx(0.04)
    assert l.rtf_ocr_glossario == pytest.approx(0.01) and l.delta_rtf_total == pytest.approx(0.05)
    o = t.loc[("tiny", "limpo", "O")]  # o oráculo usa o terms.txt humano: não paga OCR
    assert pd.isna(o.rtf_ocr_glossario) and o.delta_rtf_total == pytest.approx(0.0)
    assert t.loc[("tiny", "limpo", "B0")].delta_rtf_total == 0.0


def test_tabela3_tempo_so_da_rq1_nao_mistura_as_linhas_de_contexto(stt, saida):
    report.table3_time(stt)
    t = pd.read_csv(saida / "tables" / "tabela3_tempo_stt.csv", index_col=0)
    assert t.loc["tiny", "rtf"] == "0.500 ± 0.000 (n=2)"  # só as 2 linhas da RQ1 (rtf 0,5); as de contexto não entram


def test_figuras_5_e_7_sao_gravadas(stt, saida):
    report.fig5_context_gain_snr(stt)
    report.fig7_placebo(stt)
    for nome in ("fig5_ganho_termos_snr", "fig7_placebo"):
        assert (saida / "figures" / f"{nome}.png").stat().st_size > 1000 and (saida / "figures" / f"{nome}.pdf").exists()


def test_figuras_sem_dados_de_contexto_sao_puladas(saida):
    vazio = pd.DataFrame([_linha("v1", "tiny", "limpo", "wer", 0.3)])
    report.table6_context(vazio)
    report.table7_context_cost(vazio)
    report.fig5_context_gain_snr(vazio)
    report.fig7_placebo(vazio)
    report.fig6_calibration()  # sem results/calibracao.csv
    assert not (saida / "tables").exists() and not (saida / "figures").exists()


def test_figura_6_le_o_csv_de_calibracao(saida):
    pd.DataFrame([{"modelo": "tiny", "condicao": "limpo", "bin_lo": b / 10, "bin_hi": (b + 1) / 10, "n": 10,
                   "conf_media": b / 10 + 0.05, "acerto": min(1.0, b / 10 + 0.1)} for b in range(10)]) \
        .to_csv(saida / "calibracao.csv", index=False)
    report.fig6_calibration()
    assert (saida / "figures" / "fig6_calibracao_confianca.png").exists()
