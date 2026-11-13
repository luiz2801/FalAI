import csv

import pytest

import context_audit
import run_experiments as rx


def _marca(video, modelo, cond, idx, aud):
    return {"video": video, "modelo": modelo, "condicao": cond, "idx": str(idx), "t": "1.0", "antes": "a", "depois": "b",
            "auditoria": aud, "nota": ""}


def test_precisions_ignora_nao_sei_e_o_pipeline():
    rows = ([_marca("aula01", "tiny", "limpo@L+F", i, "certa") for i in range(3)]
            + [_marca("aula01", "tiny", "limpo@L+F", 3, "errada")]
            + [_marca("aula01", "tiny", "limpo@L+F", i, "duvida") for i in (4, 5)]
            + [_marca("aula02", "tiny", "limpo@L+F", 0, "errada")]
            + [_marca("aula01", "small", "pipeline@L+F", 0, "certa")]  # o pipeline não é execução da matriz
            + [_marca("aula03", "tiny", "limpo@L+F", 0, "duvida")])  # só "não sei": sem precisão
    p = context_audit.precisions(rows)
    assert p == {("aula01", "tiny", "limpo@L+F"): (0.75, 3, 1), ("aula02", "tiny", "limpo@L+F"): (0.0, 0, 1)}


@pytest.fixture
def ambiente(tmp_path, monkeypatch):
    monkeypatch.setattr(rx, "RAW_CSV", tmp_path / "raw.csv")
    monkeypatch.setattr(context_audit, "AUDIT_CSV", tmp_path / "correcoes_auditoria.csv")
    return tmp_path


def _grava(amb, rows):
    with (amb / "correcoes_auditoria.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["video", "modelo", "condicao", "idx", "t", "antes", "depois", "auditoria", "nota"])
        w.writeheader()
        w.writerows(rows)


def _linhas(amb):
    with (amb / "raw.csv").open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_consolidate_grava_precisao_correcoes_no_raw_csv_e_atualiza(ambiente, capsys):
    _grava(ambiente, [_marca("aula01", "tiny", "limpo@L+F", 0, "certa"), _marca("aula01", "tiny", "limpo@L+F", 1, "errada")])
    context_audit.consolidate()
    r = _linhas(ambiente)
    assert [(x["video"], x["tarefa"], x["modelo"], x["condicao"], x["repeticao"], x["metrica_erro"], x["valor_erro"], x["tempo_s"])
            for x in r] == [("aula01", "stt", "tiny", "limpo@L+F", "1", "precisao_correcoes", "0.5", "")]
    assert "precisão das correções = 0.500 (1 certas, 1 erradas)" in capsys.readouterr().out
    _grava(ambiente, [_marca("aula01", "tiny", "limpo@L+F", 0, "certa"), _marca("aula01", "tiny", "limpo@L+F", 1, "certa")])
    context_audit.consolidate()  # reauditou: acrescenta; o relatório vale pela última linha
    import report

    df = report.load(ambiente / "raw.csv")
    assert len(_linhas(ambiente)) == 2 and list(df.valor_erro) == [1.0]


def test_consolidate_sem_marcacoes_nao_grava_nada(ambiente):
    with pytest.raises(SystemExit, match="não existe"):
        context_audit.consolidate()
    _grava(ambiente, [_marca("aula01", "tiny", "limpo@L+F", 0, "duvida")])
    assert context_audit.consolidate() == [] and not (ambiente / "raw.csv").exists()
