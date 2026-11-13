import pytest

import metrics


def test_norm_mantem_acentos_e_remove_pontuacao():
    assert metrics.norm("  Olá, MUNDO!  Ação... ") == "olá mundo ação"
    assert metrics.norm("MPEG-DASH") == "mpeg dash"


def test_wer_e_cer_a_mao():
    assert metrics.wer("o banco central regula", "o banco central regula") == 0
    assert metrics.wer("o banco central regula", "o banco central") == pytest.approx(0.25)
    assert metrics.wer("Olá, mundo.", "olá mundo") == 0  # mesma normalização nos dois lados
    assert metrics.cer("abcd", "abce") == pytest.approx(0.25)


def test_referencia_vazia_e_erro():
    with pytest.raises(ValueError):
        metrics.wer("  ...  ", "algo")


def test_wer_detail():
    d = metrics.wer_detail("a b c d", "a x c")
    assert d["substitutions"] == 1 and d["deletions"] == 1 and d["insertions"] == 0
    assert d["ref_words"] == 4


def test_word_pr_multiconjunto():
    p, r = metrics.word_pr("a a b", "a b b c")
    assert p == pytest.approx(2 / 4)
    assert r == pytest.approx(2 / 3)


def test_rtf():
    assert metrics.rtf(30, 60) == 0.5
    with pytest.raises(ValueError):
        metrics.rtf(1, 0)


def test_term_accuracy():
    ref = "o jitter e o jitter afetam o RTP"
    hyp = "o gitter e o jitter afetam o RTP e o RTP"
    r = metrics.term_accuracy(ref, hyp, ["jitter", "RTP", "QoE"])
    assert r["terms_in_ref"] == 3
    assert r["terms_hit"] == 2  # um jitter + um RTP
    assert r["false_positives"] == 1  # RTP a mais
    assert metrics.term_accuracy("nada aqui", "nada", ["jitter"])["accuracy"] is None


def test_align_matches_wer_counts():
    from metrics import align, wer_detail

    ref, hyp = "A casa azul, é bonita demais.", "a cada azul bonita demais sim"
    chunks = align(ref, hyp)
    assert [c["op"] for c in chunks] == ["equal", "substitute", "equal", "delete", "equal", "insert"]
    assert chunks[1] == {"op": "substitute", "ref": ["casa"], "hyp": ["cada"]}
    d = wer_detail(ref, hyp)
    erros = sum(len(c["ref"]) if c["op"] != "insert" else len(c["hyp"]) for c in chunks if c["op"] != "equal")
    assert erros == d["substitutions"] + d["deletions"] + d["insertions"]


# --- F10: gabarito parcial de termos (camada A) ---------------------------------------------------------
def _hip(texto, t0=0.0, passo=1.0):
    return [(t0 + i * passo, w) for i, w in enumerate(metrics.norm(texto).split())]


def test_timed_term_accuracy_acerta_dentro_da_janela_e_conta_falso_positivo():
    ocorrencias = [(10.0, "BACEN"), (30.0, "CMN"), (50.0, "Conselho Monetário Nacional")]
    hip = _hip("o BACEN define e o cmn", t0=9.0) + _hip("conselho monetário nacional", t0=70.0)
    r = metrics.timed_term_accuracy(ocorrencias, hip, window_s=3.0)
    # BACEN em 10 (anotado 10): acerto; CMN em 14 (anotado 30): fora da janela; o nome em 70 (anotado 50): fora
    assert r["terms_in_ref"] == 3 and r["terms_hit"] == 1 and abs(r["accuracy"] - 1 / 3) < 1e-9
    assert r["false_positives"] == 2  # cmn@14 e o nome@70 existem na hipótese sem anotação por perto
    ok = metrics.timed_term_accuracy(ocorrencias, _hip("bacen", 10.0) + _hip("cmn", 30.5) +
                                     _hip("conselho monetário nacional", 49.0))
    assert ok["terms_hit"] == 3 and ok["false_positives"] == 0 and ok["accuracy"] == 1.0


def test_timed_term_accuracy_cada_ocorrencia_da_hipotese_vale_uma_anotacao_so():
    r = metrics.timed_term_accuracy([(10.0, "CMN"), (11.0, "CMN")], _hip("cmn", 10.5))
    assert r["terms_hit"] == 1 and r["false_positives"] == 0  # uma palavra na hipótese não cobre duas anotações
    assert metrics.timed_term_accuracy([], _hip("cmn"))["accuracy"] is None
    assert metrics.timed_term_accuracy([(1.0, "CMN")], [])["accuracy"] == 0.0


def test_word_flags_marca_substituidas_e_inseridas():
    ref = "o presidente do bacen define as metas"
    hip = metrics.norm("o presidente do Bassim define ai as metas").split()
    assert metrics.word_flags(ref, hip) == [True, True, True, False, True, False, True, True]  # "bassim" e "ai"
    assert metrics.word_flags(ref, []) == [] and all(metrics.word_flags(ref, ref.split()))
