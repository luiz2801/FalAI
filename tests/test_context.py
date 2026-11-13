import itertools

import context

TEXTO_CMN = """CMN – Conselho Monetário Nacional
• Órgão máximo do SFN, responsável por formular a política de moeda e de crédito.
• Composição: Ministro da Fazenda, Presidente do BACEN e Ministro do Planejamento
Objetivos:
e Equilibrar os meios de pagamento às necessidades da economia;
"""


def _par(termos):
    return {(t["kind"], t["text"]) for t in termos}


def test_extract_terms_de_um_slide_de_texto():
    t = _par(context.extract_terms(TEXTO_CMN))
    assert {("sigla", "CMN"), ("sigla", "SFN"), ("sigla", "BACEN")} <= t
    assert {("nome", "Conselho Monetário Nacional"), ("nome", "Ministro da Fazenda"),
            ("nome", "Ministro do Planejamento")} <= t
    assert ("nome", "Presidente") not in t and ("sigla", "ME") not in t
    assert ("termo", "Equilibrar") in t  # o marcador "e" lido pelo OCR não vira termo nem atrapalha a palavra


def test_extract_terms_primeira_linha_e_titulo():
    termos = context.extract_terms(TEXTO_CMN)
    assert {t["text"] for t in termos if t["titulo"]} == {"CMN", "Conselho Monetário Nacional"}


def test_extract_terms_titulos_em_caixa_alta():
    t = _par(context.extract_terms("INVESTIDOR QUALIFICADO\n✖\nINVESTIDOR PROFISSIONAL"))
    assert t == {("nome", "Investidor qualificado"), ("nome", "Investidor profissional")}
    t = _par(context.extract_terms("DICAS DE PROVA – MÓDULO II\nMERCADO DE CAPITAIS\nB3\nCAPM\nGordon\nSELIC e CDB"))
    assert {("sigla", "B3"), ("sigla", "CAPM"), ("sigla", "SELIC"), ("sigla", "CDB")} <= t
    assert ("nome", "Mercado de capitais") in t
    assert not any(k == "sigla" and x in {"II", "DICAS", "PROVA", "MÓDULO"} for k, x in t)  # romano e título não são sigla


def test_extract_terms_numero_de_lei_inclusive_em_caixa_alta():
    t = _par(context.extract_terms("Lei Complementar nº 179, de 24 de fevereiro de 2021"))
    assert ("numero", "Lei Complementar 179") in t
    t = _par(context.extract_terms("LEI COMPLEMENTAR Nº 179, DE 24 DE FEVEREIRO DE 2021"))
    assert ("numero", "Lei Complementar 179") in t


def test_extract_terms_plural_de_sigla_e_romano():
    t = _par(context.extract_terms("As IMFs e o item II tratam de B3"))
    assert ("sigla", "IMF") in t and ("sigla", "B3") in t and not any(x == "II" for _, x in t)


def test_spell_e_pronounceable():
    assert context.spell("CMN") == "cê ême ene" and context.spell("B3") == "bê três"
    assert context.spell("SPB") == "ésse pê bê"
    for lida in ("BACEN", "SELIC", "ANBIMA", "COPOM", "CETIP"):
        assert context.pronounceable(lida), lida
    for soletrada in ("CMN", "CVM", "SPB", "B3", "BNDES", "CDI", "FGC"):
        assert not context.pronounceable(soletrada), soletrada


def _slides(textos):
    return [{"file": f"{i:04d}.png", "t": 10.0 * i, "t_end": 10.0 * i + 9.0, "texto": t} for i, t in enumerate(textos)]


def test_build_glossary_tira_o_boilerplate_e_pesa_titulo_e_tipo():
    rodape = "Curso Economia do Dia"  # aparece em todos os slides
    gl = context.build_glossary(_slides([f"{TEXTO_CMN}\n{rodape}", f"Tesouro Nacional\nDealers\n{rodape}",
                                         f"SPB\nSistema de Pagamentos Brasileiro\n{rodape}", f"B3\n{rodape}"]))
    todos = {t["text"] for g in gl for t in g["terms"]}
    assert "Curso Economia do Dia" not in todos and "Dealers" in todos
    cmn = {t["text"]: t for t in gl[0]["terms"]}
    assert cmn["CMN"]["weight"] > cmn["Ministro da Fazenda"]["weight"] > cmn["Equilibrar"]["weight"]  # sigla de título > nome > termo
    assert gl[0]["terms"] == sorted(gl[0]["terms"], key=lambda t: -t["weight"])
    assert cmn["BACEN"]["pron"] == ["bê á cê é ene", "bacen"] and cmn["CMN"]["pron"] == ["cê ême ene"]
    assert gl[0]["visits"] == [[0.0, 9.0]]


def test_build_glossary_com_poucos_slides_nao_apaga_tudo():
    gl = context.build_glossary(_slides([TEXTO_CMN, TEXTO_CMN]))  # 2 slides iguais: sem boilerplate (n < 3)
    assert gl[0]["terms"] and gl[1]["terms"]


def _g(texto, t, t_end, visits=None, kind="sigla", w=1.0):
    return {"slide": f"{texto}.png", "t": t, "t_end": t_end, "visits": visits or [[t, t_end]],
            "terms": [{"text": texto, "kind": kind, "weight": w, "pron": []}]}


def _cobre(plano, dur):
    assert plano[0]["t0"] == 0.0 and plano[-1]["t1"] == dur
    assert all(a["t1"] == b["t0"] for a, b in itertools.pairwise(plano))
    assert all(p["t1"] > p["t0"] for p in plano)


def _em(plano, t):
    return next(p for p in plano if p["t0"] <= t < p["t1"])


def test_plan_intervals_particiona_e_aplica_lead_lag_e_carry():
    gl = [_g("AAA", 10, 50), _g("BBB", 50, 90), _g("CCC", 120, 190, visits=[[120, 150], [170, 190]])]
    plano = context.plan_intervals(gl, 200.0, lead_s=2.0, lag_s=1.0)
    _cobre(plano, 200.0)
    assert _em(plano, 30)["slide"] == "AAA.png" and _em(plano, 70)["slide"] == "BBB.png"
    assert _em(plano, 7.9)["slide"] is None and _em(plano, 8.1)["slide"] == "AAA.png"  # lead: A entra 2 s antes
    assert _em(plano, 90.5)["slide"] == "BBB.png" and _em(plano, 91.5)["slide"] is None  # lag: B fica 1 s a mais
    assert _em(plano, 117.9)["slide"] is None and _em(plano, 118.1)["slide"] == "CCC.png"
    meio = {t["text"]: t["weight"] for t in _em(plano, 100)["terms"]}  # sem slide: herda B e ouve C ao longe
    assert meio == {"BBB": 1.0, "CCC": 0.5}
    assert _em(plano, 160)["slide"] is None and {t["text"] for t in _em(plano, 160)["terms"]} == {"CCC"}  # entre as visitas
    assert _em(plano, 199)["slide"] is None and {t["text"] for t in _em(plano, 199)["terms"]} == {"CCC"}  # cauda herda
    cabeca = {t["text"]: t["weight"] for t in _em(plano, 3)["terms"]}
    assert cabeca == {"AAA": 0.5}  # antes do 1º slide: só o vizinho, com peso menor


def test_plan_intervals_sem_carry_deixa_o_intervalo_sem_slide_vazio():
    gl = [_g("AAA", 10, 20), _g("BBB", 100, 110)]
    plano = context.plan_intervals(gl, 130.0, carry=False)
    # sem slide e sem carry: só os dois vizinhos, com peso menor (quem apresenta pode estar a caminho de B)
    assert {t["text"]: t["weight"] for t in _em(plano, 60)["terms"]} == {"AAA": 0.5, "BBB": 0.5}


def test_plan_intervals_absorve_intervalo_curto_e_sem_slides():
    gl = [_g("AAA", 10, 50), _g("BBB", 52, 90)]  # 2 s entre eles: some, com lead 0 e lag 0
    plano = context.plan_intervals(gl, 100.0, lead_s=0.0, lag_s=0.0, min_interval_s=4.0)
    _cobre(plano, 100.0)
    assert all(p["t1"] - p["t0"] >= 4.0 for p in plano)
    assert context.plan_intervals([], 50.0) == [{"t0": 0.0, "t1": 50.0, "slide": None, "terms": []}]


def test_plan_intervals_limita_os_termos_e_une_pelo_maior_peso():
    gl = [{"slide": "a", "t": 0, "t_end": 20, "visits": [[0, 20]],
           "terms": [{"text": f"T{i}", "kind": "termo", "weight": 1 - i / 100, "pron": []} for i in range(50)]}]
    plano = context.plan_intervals(gl, 20.0, max_terms=30)
    assert len(plano[0]["terms"]) == 30 and plano[0]["terms"][0]["text"] == "T0"


def test_hotwords_respeita_o_limite_e_a_ordem_de_peso():
    termos = [{"text": "pequeno", "kind": "termo", "weight": 0.2}, {"text": "CMN", "kind": "sigla", "weight": 1.0},
              {"text": "BACEN", "kind": "sigla", "weight": 0.9}]
    assert context.hotwords(termos) == "CMN, BACEN, pequeno"
    assert context.hotwords(termos, max_chars=10) == "CMN, BACEN"  # cabe exatamente
    assert context.hotwords(termos, max_chars=9) == "CMN"  # um caractere a menos e o segundo termo sai
    assert context.hotwords([]) == ""


def test_global_terms_une_slides_e_da_bonus_a_quem_se_repete():
    gl = [_g("CMN", 0, 10, w=0.8), _g("CVM", 10, 20, w=0.84), _g("CMN", 20, 30, w=0.7)]
    g = context.global_terms(gl, k=5)
    # CMN aparece em 2 slides: maior peso (0,8) × (1 + 0,1 ln 2) = 0,855 > CVM, que aparece em 1 (0,84)
    assert [t["text"] for t in g] == ["CMN", "CVM"] and abs(g[0]["weight"] - 0.8555) < 1e-4
    assert [t["text"] for t in context.global_terms(gl, k=1)] == ["CMN"]


def test_split_terms_divide_os_termos_humanos_pelo_que_esta_nos_slides():
    gl = [_g("CMN", 0, 10), _g("BACEN", 10, 20)]
    dentro, fora = context.split_terms(["cmn", "Bacen", "Selic", "Pimec"], gl)
    assert dentro == ["cmn", "Bacen"] and fora == ["Selic", "Pimec"]  # sem caixa nem acento


def test_read_timed_terms(tmp_path):
    arq = tmp_path / "aula01.termos.tsv"
    arq.write_text("# termos falados\n12,5\tBACEN\n3.0\tConselho Monetário Nacional  # no título\n\n40\tCMN\n", encoding="utf-8")
    assert context.read_timed_terms(arq) == [(3.0, "Conselho Monetário Nacional"), (12.5, "BACEN"), (40.0, "CMN")]
    ruim = tmp_path / "ruim.tsv"
    ruim.write_text("12\n", encoding="utf-8")
    import pytest

    with pytest.raises(ValueError, match="sem termo"):
        context.read_timed_terms(ruim)
