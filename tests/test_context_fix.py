import copy

import context_fix as fx


def _t(text, kind="sigla", w=1.0, pron=()):
    return {"text": text, "kind": kind, "weight": w, "pron": list(pron)}


BACEN = _t("BACEN", pron=["bê á cê é ene", "bacen"])
CMN = _t("CMN", pron=["cê ême ene"])
CVM = _t("CVM", pron=["cê vê ême"])


def _seg(*palavras, t0=10.0):
    words = [{"start": t0 + i * 0.5, "end": t0 + i * 0.5 + 0.4, "word": w, "probability": p}
             for i, (w, p) in enumerate(palavras)]
    return {"id": 1, "start": t0, "end": words[-1]["end"], "text": " ".join(w for w, _ in palavras),
            "avg_logprob": -0.2, "no_speech_prob": 0.0, "words": words}


def _plano(*termos):
    return [{"t0": 0.0, "t1": 100.0, "slide": "0001.png", "terms": list(termos)}]


def _corrige(seg, *termos, **kw):
    segs, log, resumo = fx.fix_segments([seg], _plano(*termos), **kw)
    return segs[0], log, resumo


# --- os três erros reais do VP (docs/plano_contexto_cruzado.md, S4) -------------------------------------
def test_r1_bassim_vira_bacen_e_preserva_a_pontuacao():
    seg, log, resumo = _corrige(_seg(("o", 0.9), ("presidente", 0.9), ("do", 0.9), ("Bassim.", 0.33)), BACEN, CMN)
    assert seg["text"] == "o presidente do BACEN." and seg["words"][-1]["word"] == "BACEN."
    assert seg["words"][-1]["orig"] == "Bassim." and seg["words"][-1]["start"] == 11.5
    assert [(r["regra"], r["antes"], r["depois"], r["p"], r["termo"], r["slide"]) for r in log] == [
        ("R1", "Bassim.", "BACEN.", 0.33, "BACEN", "0001.png")]
    assert log[0]["sim"] == 1.0 and log[0]["auditoria"] is None and resumo == {"segmentos_pulados": 0, "correcoes": 1}
    assert (log[0]["seg_antes"], log[0]["seg_depois"]) == ("o presidente do Bassim.", "o presidente do BACEN.")


def test_r2_cml_sem_vogais_e_baixa_confianca_vira_cmn():
    seg, log, _ = _corrige(_seg(("o", 0.9), ("papel", 0.9), ("de", 0.9), ("cml", 0.15)), BACEN, CMN)
    assert seg["text"] == "o papel de CMN" and log[0]["regra"] == "R2" and log[0]["depois"] == "CMN"


def test_r2_sigla_em_maiusculas_a_uma_edicao_nao_precisa_de_baixa_confianca():
    seg, log, _ = _corrige(_seg(("atividades", 0.9), ("do", 0.9), ("CMR.", 0.95)), BACEN, CMN)  # o erro confiante
    assert seg["text"] == "atividades do CMN." and log[0]["regra"] == "R2" and log[0]["p"] == 0.95


# --- o que NÃO pode mudar --------------------------------------------------------------------------------
def test_nao_troca_palavra_confiante_nem_sem_vogais_confiante():
    seg = _seg(("o", 0.9), ("presidente", 0.9), ("do", 0.9), ("Bassim.", 0.9))
    assert _corrige(seg, BACEN, CMN)[1] == []  # R1 só vale com p < tau
    assert _corrige(_seg(("o", 0.9), ("cml", 0.9)), CMN)[1] == []  # R2 sem vogais também tem gate


def test_nao_troca_termo_que_nao_esta_no_slide_daquele_instante():
    seg = _seg(("do", 0.9), ("Bassim.", 0.33))
    assert _corrige(seg)[1] == [] and _corrige(seg, CMN)[1] == []  # slide sem BACEN: nada a corrigir
    plano = [{"t0": 0.0, "t1": 50.0, "slide": "a", "terms": [BACEN]}, {"t0": 50.0, "t1": 100.0, "slide": "b", "terms": [CMN]}]
    fora = _seg(("do", 0.9), ("Bassim.", 0.33), t0=70.0)  # aos 70 s o slide é o de CMN
    dentro = _seg(("do", 0.9), ("Bassim.", 0.33), t0=20.0)
    assert fx.fix_segments([fora], plano)[1] == [] and len(fx.fix_segments([dentro], plano)[1]) == 1


def test_nao_mexe_no_que_ja_esta_certo_e_devolve_o_mesmo_segmento():
    seg = _seg(("o", 0.9), ("CMN", 0.3), ("define", 0.9), ("metas", 0.9))
    novo, log, _ = _corrige(seg, CMN, BACEN)
    assert novo is seg and log == []


def test_r2_nao_troca_uma_sigla_por_outra_que_tambem_esta_no_slide():
    novo, log, _ = _corrige(_seg(("a", 0.9), ("CVM", 0.9)), CMN, CVM)  # CVM está no slide: está certa
    assert log == [] and novo["text"] == "a CVM"
    assert _corrige(_seg(("a", 0.9), ("CVX", 0.9)), CMN)[1] == []  # 2 edições de CMN: não troca


def test_palavra_com_probabilidade_zero_conta_como_a_menos_confiante():
    seg, log, _ = _corrige(_seg(("do", 0.9), ("Bassim", 0.0)), BACEN)
    assert seg["text"] == "do BACEN" and log[0]["p"] == 0.0


# --- R3 e spans de várias palavras -----------------------------------------------------------------------
def test_r3_letras_soletradas_viram_a_sigla():
    seg, log, _ = _corrige(_seg(("o", 0.9), ("c", 0.5), ("m", 0.5), ("n.", 0.5)), CMN)
    assert seg["text"] == "o CMN." and log[0]["regra"] == "R3" and log[0]["antes"] == "c m n."
    seg, _, _ = _corrige(_seg(("o", 0.9), ("cê", 0.9), ("ême", 0.9), ("ene", 0.9), ("define", 0.9)), CMN)
    assert seg["text"] == "o CMN define"  # nome das letras, sem gate de confiança
    b3 = _t("B3", pron=["bê três"])
    assert _corrige(_seg(("na", 0.9), ("bê", 0.9), ("três", 0.9)), b3)[0]["text"] == "na B3"


def test_r1_com_duas_palavras_que_soam_como_um_termo():
    seg, log, _ = _corrige(_seg(("o", 0.9), ("ba", 0.4), ("cem", 0.6), ("define", 0.9)), BACEN)
    assert seg["text"] == "o BACEN define" and log[0]["antes"] == "ba cem"
    assert (seg["words"][1]["start"], seg["words"][1]["end"]) == (10.5, 11.4)  # do início de "ba" ao fim de "cem"


def test_r1_nome_de_slide_e_nao_so_sigla():
    dealers = _t("Dealers", kind="nome", pron=())
    seg, log, _ = _corrige(_seg(("os", 0.9), ("dilers", 0.35), ("compram", 0.9)), dealers)
    assert seg["text"] == "os Dealers compram" and log[0]["termo"] == "Dealers"


# --- robustez --------------------------------------------------------------------------------------------
def test_segmento_sem_words_ou_com_texto_diferente_e_pulado_e_contado():
    sem = {"id": 1, "start": 0, "end": 2, "text": "do Bassim", "words": None}
    seg = _seg(("do", 0.9), ("Bassim", 0.2))
    seg["text"] = "do Bassim, certo"  # o texto não bate com as palavras (ex.: o filtro mexeu no texto)
    segs, log, resumo = fx.fix_segments([sem, seg], _plano(BACEN))
    assert segs == [sem, seg] and log == [] and resumo == {"segmentos_pulados": 2, "correcoes": 0}


def test_nao_altera_a_entrada():
    seg = _seg(("o", 0.9), ("Bassim.", 0.33))
    antes = copy.deepcopy(seg)
    fx.fix_segments([seg], _plano(BACEN))
    assert seg == antes


def test_phonetic_key_e_similarity():
    assert fx.similarity("Bassim", "BACEN") == 1.0 and fx.similarity("Pimek", "PIMEC") == 1.0
    assert fx.similarity("celic", "selic") == 1.0 and fx.similarity("cemene", "cê ême ene") == 1.0
    assert fx.similarity("banco", "bacen") < 0.85 and fx.similarity("", "bacen") == 0.0
    assert fx.similarity("dilers", "Dealers") > 0.8
    assert fx.edit_distance("kitten", "sitting") == 3 and fx.edit_distance("", "abc") == 3


# --- achados do teste com áudio real (medium em CPU, trecho do VP): docs/decisoes.md, D16 ----------------
def test_nao_apaga_palavra_funcional_nem_corrige_palavra_comum_do_slide():
    composicao = _t("Composição", kind="termo")
    seg = _seg(("sobre", 0.9), ("a", 0.42), ("composição", 0.9), ("do", 0.9), ("CMN", 0.9))
    novo, log, _ = _corrige(seg, composicao, CMN)
    assert novo is seg and log == []  # "a composição" não vira "Composição": termo comum não é alvo, e o "a" não é do termo
    # mesmo que o termo fosse um nome, o "a" não faz falta para a semelhança: o span não vale
    nome = _t("Composição", kind="nome")
    assert _corrige(seg, nome, CMN, kinds=("sigla", "nome"))[1] == []
    fazenda = _t("Ministro da Fazenda", kind="nome")
    assert _corrige(_seg(("o", 0.3), ("ministro", 0.9), ("da", 0.9), ("fazenda", 0.9)), fazenda)[1] == []


def test_nasal_final_sem_a_consoante_e_a_mesma_sigla():
    seg, log, _ = _corrige(_seg(("o", 0.9), ("presidente", 0.9), ("do", 0.9), ("bassé", 0.41)), BACEN)  # lido no medium
    assert seg["text"] == "o presidente do BACEN" and log[0]["regra"] == "R1" and log[0]["antes"] == "bassé"
    assert fx.similarity("bassé", "BACEN") == 1.0 and fx.similarity("banco", "bacen") < 0.85
    assert fx.similarity("sem", "SBC") < 0.85  # a nasal opcional não aproxima qualquer coisa


def test_kinds_permite_incluir_termos_comuns_se_o_usuario_quiser():
    objetivos = _t("Objetivos", kind="termo")
    seg = _seg(("os", 0.9), ("objetivu", 0.3), ("são", 0.9))
    assert _corrige(seg, objetivos)[1] == []
    assert [r["depois"] for r in _corrige(seg, objetivos, kinds=("sigla", "nome", "termo"))[1]] == ["Objetivos"]
