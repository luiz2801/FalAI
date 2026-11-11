import textwrap

import stt


def _seg(i, start, end, text, words=True):
    ws = None
    if words:
        tokens = text.split()
        passo = (end - start) / len(tokens)
        ws = [{"start": start + k * passo, "end": start + (k + 1) * passo, "word": w, "probability": 1.0}
              for k, w in enumerate(tokens)]
    return {"id": i, "start": start, "end": end, "text": text, "words": ws}


def test_fmt():
    assert stt.fmt(0) == "00:00:00.000"
    assert stt.fmt(3723.4567) == "01:02:03.457"
    assert stt.fmt_srt(61.5) == "00:01:01,500"


def test_write_vtt_cabecalho_e_blocos(tmp_path):
    segs = [_seg(1, 0, 2, "olá turma"), _seg(2, 2.5, 4, "hoje vamos falar de codecs")]
    p = stt.write_vtt(segs, tmp_path / "a.vtt", max_chars=None)
    linhas = p.read_text(encoding="utf-8").split("\n")
    assert linhas[0] == "WEBVTT" and linhas[1] == ""
    assert linhas[2] == "00:00:00.000 --> 00:00:02.000" and linhas[3] == "olá turma"


def test_write_srt_numerado(tmp_path):
    p = stt.write_srt([_seg(1, 0, 1.5, "um dois")], tmp_path / "a.srt", max_chars=None)
    assert p.read_text(encoding="utf-8").startswith("1\n00:00:00,000 --> 00:00:01,500\num dois")


def test_blocos_respeitam_limites_e_nao_deixam_orfaos():
    texto = ("nesse vídeo a gente vai fazer uma introdução sobre o conceito de capital de giro "
             "e podem cair algumas questões de conta nesse momento da prova")
    blocos = list(stt.subtitle_blocks([_seg(1, 0, 9, texto)], max_chars=42))
    assert len(blocos) >= 2
    for ini, fim, t in blocos:
        assert len(textwrap.wrap(t, 42)) <= 2
        assert fim - ini <= 7.0 + 1e-9
        assert len(t) >= 20  # sem bloco de uma palavra só
    assert " ".join(t for _, _, t in blocos) == texto


def test_blocos_sem_tempo_por_palavra():
    blocos = list(stt.subtitle_blocks([_seg(1, 0, 10, "palavra " * 40, words=False)], max_chars=42))
    assert blocos[0][0] == 0 and abs(blocos[-1][1] - 10) < 1e-9


def test_clean_segments_remove_alucinacao_e_repeticao():
    segs = [_seg(1, 0, 1, "Bom dia a todos."), _seg(2, 1, 2, "Legendas pela comunidade Amara.org"),
            _seg(3, 2, 3, "Bom dia a todos."), _seg(4, 3, 4, "então então então então então certo")]
    mantidos, removidos = stt.clean_segments(segs)
    assert [s["text"] for s in mantidos] == ["Bom dia a todos.", "então certo"]
    assert {r["reason"] for r in removidos} == {"hallucination", "repetition"}
    assert [s["id"] for s in mantidos] == [1, 2]


def test_clean_mantem_frase_longa_que_cita_legenda():
    s = _seg(1, 0, 5, "Para acessibilidade, as legendas por comunidade são uma alternativa barata "
                      "que vamos discutir em detalhes hoje na aula de multimídia.")
    assert len(stt.clean_segments([s])[0]) == 1


def test_plain_text_paragrafos():
    segs = [_seg(1, 0, 1, "Primeira frase."), _seg(2, 1.2, 2, "Segunda frase."), _seg(3, 5, 6, "Depois da pausa.")]
    assert stt.plain_text(segs) == "Primeira frase. Segunda frase.\n\nDepois da pausa.\n"


def test_build_prompt_respeita_limite():
    p = stt.build_prompt([f"termo{i}" for i in range(500)])
    assert len(p) <= stt.MAX_PROMPT_CHARS + 1 and p.startswith("Aula:") and p.endswith(".")
