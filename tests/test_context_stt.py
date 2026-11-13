import numpy as np
import pytest

import context
import context_stt as cs

SR = cs.SR
DUR = 140.0


def _t(text, kind="sigla", w=1.0):
    return {"text": text, "kind": kind, "weight": w, "pron": []}


def _g(slide, t, t_end, *termos):
    return {"slide": slide, "t": t, "t_end": t_end, "visits": [[t, t_end]], "terms": list(termos)}


# slide A (10–50 s) fala de BACEN, B (50–90) de CMN, C (90–130) de CVM
GLOSSARIO = [_g("A", 10, 50, _t("BACEN")), _g("B", 50, 90, _t("CMN")), _g("C", 90, 130, _t("CVM"))]


def _audio(dur=DUR):
    return np.arange(int(dur * SR), dtype=np.float32) / SR  # o valor de cada amostra é o seu instante


def _roteiro():
    """O que a pessoa diz: uma palavra a cada 2 s, duas palavras que cruzam a fronteira de 50 s, e o mesmo
    erro ("Bassim") em dois slides diferentes: aos 20 s (slide do BACEN) e aos 110 s (slide da CVM)."""
    r = [(2.0 * i, 2.0 * i + 0.4, f"w{i}", 0.9) for i in range(70)]
    r[10] = (20.0, 20.4, "Bassim", 0.3)
    r[55] = (110.0, 110.4, "Bassim", 0.3)
    r += [(49.0, 49.9, "borda1", 0.9), (49.6, 50.6, "borda2", 0.9)]  # pontos médios 49,45 e 50,1
    return sorted(r)


class AsrFalso:
    """Devolve as palavras do roteiro que cabem na fatia, em tempo relativo, agrupadas de 3 em 3. Com `vies`,
    troca a palavra pelo termo se o termo estiver nos hotwords da chamada (efeito do contexto, simulado)."""

    def __init__(self, vies=None, extra=(), tempo=0.5):
        self.vies, self.extra, self.tempo, self.chamadas = vies or {}, list(extra), tempo, []

    def __call__(self, audio, texto):
        a = round(float(audio[0]), 3)
        b = a + len(audio) / SR
        self.chamadas.append((a, round(b, 3), texto))
        ws = []
        for s, e, w, p in _roteiro():
            if s >= a - 1e-4 and e <= b + 1e-4:
                if texto and w in self.vies and self.vies[w] in texto:
                    w, p = self.vies[w], 0.95
                ws.append({"start": round(s - a, 3), "end": round(e - a, 3), "word": w, "probability": p})
        segs = []
        for i in range(0, len(ws), 3):
            g = ws[i:i + 3]
            segs.append({"id": i // 3 + 1, "start": g[0]["start"], "end": g[-1]["end"],
                         "text": " ".join(x["word"] for x in g), "avg_logprob": -0.2, "no_speech_prob": 0.0, "words": g})
        for x in self.extra:
            if x["start"] >= a and x["end"] <= b:
                segs.append({**x, "start": x["start"] - a, "end": x["end"] - a, "words": None})
        return segs, self.tempo


def _palavras(r):
    return [(w["start"], w["word"]) for s in r["segs"] for w in s["words"]]


def _esperado():
    return sorted((round(s, 3), w) for s, _, w, _ in _roteiro())


def test_b0_com_um_intervalo_e_sem_termos_e_igual_ao_que_o_asr_devolve():
    asr = AsrFalso()
    r = cs.transcribe_context(_audio(), "B0", asr)
    assert r["n_calls"] == 1 and asr.chamadas == [(0.0, DUR, None)] and r["tempo_s"] == 0.5
    assert sorted(_palavras(r)) == _esperado()
    assert [s["id"] for s in r["segs"]] == list(range(1, len(r["segs"]) + 1)) and r["corrections"] == []


def test_l_cada_intervalo_recebe_os_termos_do_seu_slide_e_nenhuma_palavra_some_ou_duplica():
    asr = AsrFalso(vies={"Bassim": "BACEN"})
    r = cs.transcribe_context(_audio(), "L", asr, GLOSSARIO)
    assert r["n_calls"] == 5 == len(asr.chamadas) and r["tempo_s"] == 2.5  # cabeça, A, B, C e cauda
    # a palavra da fronteira cai uma vez só, no intervalo do ponto médio (49,45 -> A; 50,1 -> B)
    por_slide = {iv["slide"]: iv for iv in r["plan"]}
    assert por_slide["A"]["t0"] == 8.0 and por_slide["B"]["t0"] == 50.0 and por_slide["C"]["t1"] == 131.0
    # fatias com margem de 2 s: A vai de 6 a 52
    assert (6.0, 52.0) in [(a, b) for a, b, _ in asr.chamadas]
    palavras = _palavras(r)
    assert len(palavras) == len(_roteiro()) == len(set(palavras))  # nem perdeu, nem repetiu
    por_tempo = dict(palavras)
    assert por_tempo[20.0] == "BACEN"  # o slide do BACEN estava na tela
    assert por_tempo[110.0] == "Bassim"  # aos 110 s o slide é o da CVM: o contexto não vale
    hot = {tuple(c[:2]): c[2] for c in asr.chamadas}
    assert "BACEN" in hot[(6.0, 52.0)] and "BACEN" not in hot[(88.0, 133.0)]


def test_l_nenhuma_palavra_se_perde_nem_se_repete_mesmo_sem_vies():
    r = cs.transcribe_context(_audio(), "L", AsrFalso(), GLOSSARIO)
    assert sorted(_palavras(r)) == _esperado()


def test_g_e_o_sao_uma_chamada_com_o_vocabulario_inteiro():
    asr = AsrFalso(vies={"Bassim": "BACEN"})
    g = cs.transcribe_context(_audio(), "G", asr, GLOSSARIO)
    assert g["n_calls"] == 1 and set(asr.chamadas[0][2].split(", ")) == {"BACEN", "CMN", "CVM"}
    assert {w for _, w in _palavras(g)} >= {"BACEN"} and "Bassim" not in {w for _, w in _palavras(g)}  # os dois viram BACEN
    asr = AsrFalso(vies={"Bassim": "BACEN"})
    o = cs.transcribe_context(_audio(), "O", asr, oracle_terms=["BACEN", "Selic"])
    assert o["n_calls"] == 1 and asr.chamadas[0][2] == "BACEN, Selic" and "Bassim" not in {w for _, w in _palavras(o)}


def test_p_placebo_nao_leva_nenhum_termo_real_e_nao_muda_nada():
    outro = [_g("x", 0, 10, *[_t(f"ZZ{c}") for c in "ABCDEFGH"])]
    asr = AsrFalso(vies={"Bassim": "BACEN"})
    r = cs.transcribe_context(_audio(), "P", asr, GLOSSARIO, placebo_glossary=outro, oracle_terms=["Selic"])
    assert r["n_calls"] == 5
    textos = [c[2] for c in asr.chamadas if c[2]]
    assert textos and all(all(t.startswith("ZZ") for t in x.split(", ")) for x in textos)  # só placebo
    reais = {iv["slide"]: len(iv["terms"]) for iv in context.plan_intervals(GLOSSARIO, DUR)}
    assert [len(iv["terms"]) for iv in r["plan"]] == [len(iv["terms"]) for iv in context.plan_intervals(GLOSSARIO, DUR)] \
        and reais["A"] == 2  # mesmo tamanho do glossário real
    assert "Bassim" in {w for _, w in _palavras(r)}  # sem BACEN no prompt, o erro fica


def test_l_mais_f_corrige_so_onde_o_slide_tem_o_termo():
    r = cs.transcribe_context(_audio(), "L+F", AsrFalso(), GLOSSARIO)  # ASR sem viés: o erro chega ao pós-processamento
    por_tempo = {w["start"]: w["word"] for s in r["segs"] for w in s["words"]}
    assert por_tempo[20.0] == "BACEN" and por_tempo[110.0] == "Bassim"
    assert [(c["antes"], c["depois"], c["slide"]) for c in r["corrections"]] == [("Bassim", "BACEN", "A")]
    assert r["fix_summary"]["correcoes"] == 1
    assert cs.transcribe_context(_audio(), "L", AsrFalso(), GLOSSARIO)["corrections"] == []


def test_vazamento_do_prompt_e_removido_so_com_cara_de_nao_fala():
    base = {"avg_logprob": -0.3, "words": None}
    eco = {**base, "id": 1, "start": 60.0, "end": 62.0, "text": "CMN CVM BACEN", "no_speech_prob": 0.9}
    fala = {**base, "id": 2, "start": 70.0, "end": 72.0, "text": "CMN CVM BACEN", "no_speech_prob": 0.0}  # mesma frase, falada
    curto = {**base, "id": 3, "start": 80.0, "end": 81.0, "text": "CMN", "no_speech_prob": 0.9}
    r = cs.transcribe_context(_audio(), "L", AsrFalso(extra=[eco, fala, curto]), GLOSSARIO)
    assert [x["text"] for x in r["removed"] if x.get("reason") == "vazamento"] == ["CMN CVM BACEN"]
    textos = [s["text"] for s in r["segs"]]
    assert textos.count("CMN CVM BACEN") == 1 and "CMN" in textos  # a falada e a de uma palavra ficam
    # B0 não tem termos: não há o que vazar, e o filtro nem roda
    assert not [x for x in cs.transcribe_context(_audio(), "B0", AsrFalso(extra=[eco]))["removed"]
                if x.get("reason") == "vazamento"]


def test_clean_false_nao_filtra_nada():
    eco = {"id": 1, "start": 60.0, "end": 62.0, "text": "Legendas pela comunidade Amara.org", "avg_logprob": -0.9,
           "no_speech_prob": 0.9, "words": None}
    assert any("Amara" in s["text"] for s in cs.transcribe_context(_audio(), "B0", AsrFalso(extra=[eco]), clean=False)["segs"])
    assert not any("Amara" in s["text"] for s in cs.transcribe_context(_audio(), "B0", AsrFalso(extra=[eco]))["segs"])


def test_recorta_por_palavra_e_cai_para_segmento_sem_tempos_por_palavra():
    seg = {"id": 1, "start": 1.0, "end": 5.0, "text": "a b c d", "avg_logprob": -0.1, "no_speech_prob": 0.0,
           "words": [{"start": 1.0, "end": 1.5, "word": "a", "probability": 1.0},
                     {"start": 2.0, "end": 2.5, "word": "b", "probability": 1.0},
                     {"start": 3.0, "end": 3.5, "word": "c", "probability": 1.0},
                     {"start": 4.0, "end": 4.5, "word": "d", "probability": 1.0}]}
    r = cs._recorta([seg], offset=10.0, t0=12.0, t1=14.0, ultimo=False)  # b e c, em tempo absoluto 12,25 e 13,25
    assert [w["word"] for w in r[0]["words"]] == ["b", "c"] and r[0]["text"] == "b c"
    assert (r[0]["start"], r[0]["end"]) == (12.0, 13.5)
    inteiro = cs._recorta([seg], offset=10.0, t0=0.0, t1=99.0, ultimo=True)[0]
    assert inteiro["text"] == "a b c d"  # com todas as palavras, o texto original é preservado
    sem = {"id": 2, "start": 1.0, "end": 3.0, "text": "xx", "words": None}
    assert len(cs._recorta([sem], 10.0, 10.0, 12.5, False)) == 1  # ponto médio 12
    assert cs._recorta([sem], 10.0, 13.0, 20.0, False) == []


def test_erros_de_uso_e_taxa_de_alteracao():
    a = _audio(20.0)
    for modo, kw in [("X", {}), ("G", {}), ("L", {}), ("P", {"glossary": GLOSSARIO}), ("O", {})]:
        with pytest.raises(ValueError):
            cs.transcribe_context(a, modo, AsrFalso(), **kw)
    base = [{"text": "o presidente do Bassim"}]
    assert cs.changed_fraction(base, [{"text": "o presidente do BACEN"}]) == 0.25
    assert cs.changed_fraction(base, base) == 0.0 and cs.changed_fraction([{"text": ""}], base) == 0.0
