"""F10 (FalAI Contexto), Fase 4: corrige palavras do STT com os termos do slide que está na tela.

Sem modelo. Trabalha sobre os segmentos do `stt.transcribe` (com `words` e `probability`) e sobre o plano de
intervalos de `context.plan_intervals`, e **só olha os termos do slide visível naquele instante**. Cada troca
vai para o registro de correções (para auditoria humana, regra 11). Regras (docs/plano_contexto_cruzado.md, 5.5):

  R1  palavra(s) de baixa confiança (p < tau) cuja pronúncia (chave fonética do português) é parecida o
      bastante com uma sigla ou um nome do slide: "Bassim" -> "BACEN", "bassé" -> "BACEN";
  R2  sigla quase igual a uma do slide, a 1 edição: token em maiúsculas ("CMR" -> "CMN"), sem gate de
      confiança, ou token sem vogais com p < tau ("cml" -> "CMN");
  R3  letras soletradas em palavras separadas que formam uma sigla do slide: "c m n", "cê ême ene" -> "CMN".

Palavra com p >= tau nunca muda por R1, e termo que não está no slide daquele instante nunca entra.
"""

from __future__ import annotations

import re

from context import DIGIT_NAMES, LETTER_NAMES, key, strip_accents

_NASAL = re.compile(r"([aeiou])[mn](?![aeiou])")
_PONTUACAO = re.compile(r"^([^\w]*)(.*?)([^\w]*)$", re.DOTALL)
# nome da letra (sem acento) -> letra; nome do dígito -> dígito; e a própria letra ou dígito
_NOME_LETRA = {strip_accents(v): k for k, v in {**LETTER_NAMES, **DIGIT_NAMES}.items()}
_NOME_LETRA.update({c: c for c in "abcdefghijklmnopqrstuvwxyz0123456789"})


def phonetic_key(texto: str) -> str:
    """Chave fonética grosseira do português do Brasil: letras que soam igual dão a mesma chave. Cobre c/ç/ss/s/z/x,
    qu/gu, ch/lh/nh, g/j, h mudo, redução de vogais átonas (e/i, o/u) e nasal final (-em/-im/-en). Serve só para
    comparar palavras entre si (distância de edição), não é uma transcrição fonética."""
    t = strip_accents(texto.lower().replace("ç", "s"))
    t = re.sub(r"[^a-z]", "", t)
    t = re.sub(r"qu(?=[ei])", "k", t).replace("qu", "k").replace("q", "k")
    t = re.sub(r"gu(?=[ei])", "g", t)
    t = re.sub(r"(?:sc|xc)(?=[ei])", "s", t)
    for a, b in (("ch", "x"), ("lh", "l"), ("nh", "n"), ("ss", "s"), ("rr", "r")):
        t = t.replace(a, b)
    t = re.sub(r"c(?=[ei])", "s", t).replace("ck", "k").replace("c", "k")
    t = re.sub(r"g(?=[ei])", "j", t)
    for a, b in (("h", ""), ("y", "i"), ("w", "u"), ("z", "s"), ("x", "s"), ("ou", "o"), ("e", "i"), ("o", "u")):
        t = t.replace(a, b)
    t = _NASAL.sub(r"\1N", t)
    return re.sub(r"(.)\1+", r"\1", t)


def edit_distance(a: str, b: str) -> int:
    """Distância de Levenshtein (inserção, remoção e troca valem 1)."""
    anterior = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        atual = [i]
        for j, cb in enumerate(b, 1):
            atual.append(min(anterior[j] + 1, atual[j - 1] + 1, anterior[j - 1] + (ca != cb)))
        anterior = atual
    return anterior[-1]


def similarity(a: str, b: str) -> float:
    """Semelhança (0 a 1) entre duas palavras pela chave fonética: 1 − edições / tamanho da maior."""
    ka, kb = phonetic_key(a), phonetic_key(b)
    if not ka or not kb:
        return 0.0
    s = 1.0 - edit_distance(ka, kb) / max(len(ka), len(kb))
    # o ASR às vezes escreve a vogal nasal sem a consoante ("bassé" por "Bacen"): a nasal conta como opcional
    ka2, kb2 = ka.replace("N", ""), kb.replace("N", "")
    if ka2 and kb2:
        s = max(s, 1.0 - edit_distance(ka2, kb2) / max(len(ka2), len(kb2)))
    return s


def _divide(palavra: str) -> tuple[str, str, str]:
    m = _PONTUACAO.match(palavra)
    return m.group(1), m.group(2), m.group(3)


def _p(palavra: dict) -> float:
    """Probabilidade da palavra; sem o campo, vale 1 (confiante). 0,0 é 0,0: o pior caso, não o melhor."""
    p = palavra.get("probability")
    return 1.0 if p is None else float(p)


def _ws(texto: str) -> str:
    return re.sub(r"\s+", " ", texto).strip()


def _termos_em(plano: list[dict], t: float) -> tuple[list[dict], str | None]:
    for p in plano:
        if p["t0"] <= t < p["t1"]:
            return p["terms"], p.get("slide")
    return (plano[-1]["terms"], plano[-1].get("slide")) if plano and t >= plano[-1]["t1"] else ([], None)


def _formas(termo: dict) -> list[str]:
    return [termo["text"], *termo.get("pron", [])]


def _sigla_like(core: str) -> bool:
    """Token que parece sigla: maiúsculas (3+ letras) ou só consoantes (3+ letras)."""
    return core.isalpha() and len(core) >= 3 and (core.isupper() or not re.search(r"[aeiouáéíóúâêôãõ]", core.lower()))


def _r3(palavras: list[dict], i: int, siglas: list[dict]):
    """Letras soletradas em palavras separadas formando uma sigla do slide (2 a 6 palavras)."""
    for n in range(min(6, len(palavras) - i), 1, -1):
        letras = []
        for w in palavras[i:i + n]:
            nome = key(_divide(w["word"])[1]).replace(" ", "")
            if nome not in _NOME_LETRA:
                break
            letras.append(_NOME_LETRA[nome])
        else:
            juntas = "".join(letras)
            for t in siglas:
                if key(t["text"]).replace(" ", "") == juntas:
                    return t, n, 1.0
    return None


def _r2(core: str, p: float, tau: float, siglas: list[dict]):
    """Sigla a 1 edição de uma do slide. Maiúsculas: sem gate. Sem vogais: só com p < tau."""
    if not _sigla_like(core) or (not core.isupper() and p >= tau):
        return None
    for t in siglas:
        alvo = key(t["text"]).replace(" ", "")
        if core.lower() != alvo and abs(len(core) - len(alvo)) <= 1 and edit_distance(core.lower(), alvo) == 1:
            return t, 1, round(1 - 1 / max(len(core), len(alvo)), 4)
    return None


def _r1(palavras: list[dict], i: int, tau: float, s_min: float, max_span: int, termos: list[dict], kinds):
    """Palavra(s) de baixa confiança foneticamente parecida(s) com um termo do slide (melhor semelhança).
    Só termos dos tipos `kinds` (padrão: siglas e nomes; palavras comuns do slide, como "Composição", não são
    alvo: o risco de estragar texto certo é maior que o ganho). Um span de várias palavras só vale se cada
    palavra faz falta: se tirar a primeira ou a última mantém a semelhança (a menos de 0,05), a palavra é
    estranha ao termo e não pode ser apagada ("a composição" não vira "Composição")."""
    melhor = None
    for n in range(min(max_span, len(palavras) - i), 0, -1):
        span = palavras[i:i + n]
        if min(_p(w) for w in span) >= tau:
            continue
        cores = [_divide(w["word"])[1] for w in span]
        texto = " ".join(cores)
        if len(phonetic_key(texto)) < 4:
            continue
        for t in termos:
            if t["kind"] not in kinds or key(t["text"]) == key(texto):
                continue
            formas = _formas(t)
            s = max(similarity(texto, f) for f in formas)
            if s < s_min:
                continue
            if n > 1 and any(max(similarity(" ".join(parte), f) for f in formas) >= s - 0.05
                             for parte in (cores[1:], cores[:-1])):
                continue
            if melhor is None or s > melhor[2]:
                melhor = (t, n, s)
    return melhor


def fix_segments(segs: list[dict], plano: list[dict], tau: float = 0.5, s_min: float = 0.85,
                 max_span: int = 3, kinds: tuple[str, ...] = ("sigla", "nome")) -> tuple[list[dict], list[dict], dict]:
    """Aplica R3, R2 e R1 (nessa ordem) às palavras de cada segmento. Devolve (segmentos corrigidos, registro
    das correções, resumo). Segmentos sem `words`, ou cujo texto não bate com as palavras, ficam como estão e
    entram na contagem `segmentos_pulados`. Não altera os segmentos de entrada."""
    saida, registro, pulados = [], [], 0
    for seg in segs:
        palavras = seg.get("words")
        if not palavras or _ws(seg["text"]) != _ws(" ".join(w["word"] for w in palavras)):
            pulados += 1
            saida.append(seg)
            continue
        novas, i, n_antes = [], 0, len(registro)
        while i < len(palavras):
            meio = (palavras[i]["start"] + palavras[i]["end"]) / 2
            termos, slide = _termos_em(plano, meio)
            siglas = [t for t in termos if t["kind"] == "sigla"]
            w = palavras[i]
            core = _divide(w["word"])[1]
            achado, regra = None, None
            if siglas and (achado := _r3(palavras, i, siglas)):
                regra = "R3"
            elif siglas and core and (achado := _r2(core, _p(w), tau, siglas)):
                regra = "R2"
            elif termos and (achado := _r1(palavras, i, tau, s_min, max_span, termos, kinds)):
                regra = "R1"
            if not achado:
                novas.append(w)
                i += 1
                continue
            termo, n, sim = achado
            span = palavras[i:i + n]
            antes = " ".join(x["word"] for x in span)
            depois = _divide(span[0]["word"])[0] + termo["text"] + _divide(span[-1]["word"])[2]
            p_min = min(_p(x) for x in span)
            novas.append({**span[0], "end": span[-1]["end"], "word": depois, "orig": antes, "probability": p_min})
            registro.append({"t": round(span[0]["start"], 2), "antes": antes, "depois": depois,
                             "p": p_min, "sim": round(sim, 4),
                             "regra": regra, "slide": slide, "termo": termo["text"], "auditoria": None})
            i += n
        if any("orig" in w for w in novas):
            texto = " ".join(w["word"] for w in novas)
            for r in registro[n_antes:]:  # quem audita precisa ver a frase inteira, antes e depois
                r["seg_antes"], r["seg_depois"] = seg["text"], texto
            saida.append({**seg, "words": novas, "text": texto})
        else:
            saida.append(seg)
    return saida, registro, {"segmentos_pulados": pulados, "correcoes": len(registro)}
