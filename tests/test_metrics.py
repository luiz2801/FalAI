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
