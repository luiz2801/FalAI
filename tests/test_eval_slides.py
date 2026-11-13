import json

import eval_slides as ev


def _p(t, t_complete=None, file="0000.png"):
    return {"file": file, "t": t, **({"t_complete": t_complete} if t_complete is not None else {})}


def test_casa_pelo_inicio_dentro_da_tolerancia_e_conta_extras_e_perdidos():
    gt = [{"t": 25.0, "t_complete": 275.0}, {"t": 311.0, "t_complete": 330.0}, {"t": 500.0, "t_complete": 520.0}]
    pred = [_p(26.0, 275.5), _p(0.0, 0.0, "camera.png"), _p(312.5, 400.0), _p(313.0, 330.0, "dup.png")]
    r = ev.evaluate(pred, gt, tol_s=2.0)
    assert (r["detectados"], r["gabarito"], r["casados"]) == (4, 3, 2)
    assert r["precisao"] == 0.5 and abs(r["recall"] - 2 / 3) < 1e-9
    assert r["extras"] == [1, 3] and r["perdidos"] == [2]  # a câmera e a duplicata sobram; o slide de 500 s se perdeu
    # o 2º slide: detectado a 312,5 s (casa com 311), mas o quadro foi guardado como completo aos 400 s: estágio errado
    assert [p["completo"] for p in r["pares"]] == [True, False] and r["completos"] == 0.5
    assert r["pares"][0]["dt_inicio"] == 1.0


def test_cada_slide_do_gabarito_casa_uma_vez_e_o_mais_proximo_vence():
    gt = [{"t": 10.0}]
    r = ev.evaluate([_p(11.5, file="a.png"), _p(10.2, file="b.png")], gt)
    assert r["casados"] == 1 and r["pares"][0]["pred"] == 1 and r["extras"] == [0]  # a duplicata é o extra


def test_sem_t_complete_nao_avalia_completos_e_listas_vazias_nao_dividem_por_zero():
    r = ev.evaluate([_p(10.0)], [{"t": 10.0}])
    assert r["completos"] is None and r["precisao"] == 1.0 and r["recall"] == 1.0
    vazio = ev.evaluate([], [])
    assert vazio["precisao"] is None and vazio["recall"] is None and vazio["casados"] == 0
    assert ev.evaluate([], [{"t": 1.0}])["recall"] == 0.0 and ev.evaluate([_p(1.0)], [])["precisao"] == 0.0


def test_cli_imprime_o_resumo(tmp_path, monkeypatch, capsys):
    (tmp_path / "frames.json").write_text(json.dumps([_p(26.0, 275.5)]), encoding="utf-8")
    (tmp_path / "gt.json").write_text(json.dumps([{"t": 25.0, "t_complete": 275.0, "titulo": "CMN"},
                                                  {"t": 400.0, "titulo": "Perdido"}]), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["eval_slides.py", str(tmp_path / "frames.json"), str(tmp_path / "gt.json")])
    ev.main()
    out = capsys.readouterr().out
    assert "casados 1" in out and "precisão 1.000  recall 0.500  completos 1.000" in out and "Perdido@400.0s" in out
