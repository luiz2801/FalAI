"""F10 (FalAI Contexto): glossário por slide e plano de intervalos de contexto.

Funções puras: nada aqui carrega modelo nem lê vídeo. A entrada é o texto de cada slide (do OCR) e os
instantes em que o slide esteve na tela (`frames.json`). A saída é, para cada slide, os termos que vale a
pena passar ao Whisper (siglas, nomes e termos), e um plano que diz quais termos valem em cada instante.

    uv run python src/context.py glossary data/raw/aula01.mp4            # slides + OCR + glossário
    uv run python src/context.py glossary data/raw/aula01.mp4 --ocr-json ocr.json   # sem rodar o OCR

Contratos e decisões: docs/plano_contexto_cruzado.md (seções 5.3 e 8) e guia.md (4.11 e 5.12).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
import unicodedata
from pathlib import Path

KIND_WEIGHT = {"sigla": 1.0, "nome": 0.9, "numero": 0.6, "termo": 0.5}
MAX_HOTWORD_CHARS = 600  # ~224 tokens em PT-BR com folga; o faster-whisper trunca o hotwords em 223 tokens

_UP, _LOW = "A-ZÁÀÂÃÉÊÍÓÔÕÚÇ", "a-záàâãéêíóôõúç"
_SIGLA_RE = re.compile(rf"(?<![\w])([{_UP}][{_UP}0-9]{{1,6}})s?(?![\w])")
_NOME_RE = re.compile(rf"(?<![\w])[{_UP}][{_LOW}]+(?:\s+(?:(?:de|da|do|das|dos|e)\s+)?[{_UP}][{_LOW}]+){{1,4}}")
_NUM_RE = re.compile(r"(?<![\w])(Lei|Resolu[çc][ãa]o|Circular|Decreto|Instru[çc][ãa]o|Portaria)(\s+Complementar)?"
                     r"\s*(?:n[ºo°.]*\s*)?(\d[\d.]*)", re.IGNORECASE)
_PALAVRA_RE = re.compile(rf"(?<![\w])[{_UP}{_LOW}]{{6,}}(?![\w])")
_ROMANOS = {"I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"}
_STOP = {"porque", "quando", "também", "sendo", "pelos", "pelas", "entre", "sobre", "assim", "ainda", "outros",
         "outras", "todos", "todas", "esses", "essas", "estas", "estes", "aquele", "aquela", "mesmo", "mesma",
         "menos", "através", "dentro", "durante", "partir", "ações"}

LETTER_NAMES = {"a": "á", "b": "bê", "c": "cê", "d": "dê", "e": "é", "f": "éfe", "g": "gê", "h": "agá",
                "i": "i", "j": "jota", "k": "cá", "l": "éle", "m": "ême", "n": "ene", "o": "ó", "p": "pê",
                "q": "quê", "r": "érre", "s": "ésse", "t": "tê", "u": "u", "v": "vê", "w": "dáblio",
                "x": "xis", "y": "ípsilon", "z": "zê"}
DIGIT_NAMES = {"0": "zero", "1": "um", "2": "dois", "3": "três", "4": "quatro", "5": "cinco", "6": "seis",
               "7": "sete", "8": "oito", "9": "nove"}


# ------------------------------------------------------------------ texto
def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def key(s: str) -> str:
    """Chave de comparação: minúsculas, sem acento, só letras e dígitos separados por espaço."""
    return re.sub(r"[^a-z0-9]+", " ", strip_accents(s).lower()).strip()


def spell(sigla: str) -> str:
    """'CMN' -> 'cê ême ene'; 'B3' -> 'bê três' (a sigla soletrada em português)."""
    return " ".join(LETTER_NAMES.get(c) or DIGIT_NAMES[c] for c in key(sigla).replace(" ", ""))


def pronounceable(sigla: str) -> bool:
    """A sigla também se lê como palavra (BACEN, SELIC, ANBIMA)? Alterna consoante e vogal: ≥ 4 letras,
    ≥ 2 vogais e nenhuma sequência de 3 consoantes. CMN, CVM, SPB, B3 e BNDES são soletradas."""
    s = key(sigla).replace(" ", "")
    return (len(s) >= 4 and s.isalpha() and sum(c in "aeiou" for c in s) >= 2
            and not re.search(r"[^aeiou]{3,}", s))


def _limpa_linha(linha: str) -> str:
    linha = re.sub(r"^[\s•·▪■□○●◦\-–—*»«>]+", "", linha.strip())
    # o OCR lê o marcador como "e", "o", "°"... antes de uma palavra capitalizada
    linha = re.sub(rf"^[eo°¢@©®]\s+(?=[{_UP}])", "", linha)
    return re.sub(r"\s+", " ", linha).strip()


def _so_maiusculas(linha: str) -> bool:
    letras = [c for c in linha if c.isalpha()]
    return len(letras) >= 4 and all(c.isupper() for c in letras)


def extract_terms(text: str) -> list[dict]:
    """Termos de um slide, sem repetir: [{"text", "kind", "titulo"}], na ordem em que aparecem.

    kind: `sigla` (CMN, BACEN, B3), `nome` (Conselho Monetário Nacional), `numero` (Lei Complementar 179) e
    `termo` (palavra de 6+ letras fora de uma lista de palavras funcionais). `titulo` marca o que está na
    primeira linha. Linha só em maiúsculas (título de caixa alta) não gera siglas: com 2 a 4 palavras vira
    um `nome` ("INVESTIDOR QUALIFICADO" -> "Investidor qualificado"); mais longa que isso é ignorada. Uma
    palavra sozinha em maiúsculas na linha ("CAPM") é sigla."""
    vistos: set[tuple[str, str]] = set()
    termos: list[dict] = []

    def add(texto: str, kind: str, titulo: bool) -> None:
        k = (key(texto), kind)
        if k[0] and k not in vistos:
            vistos.add(k)
            termos.append({"text": texto, "kind": kind, "titulo": titulo})

    primeira = True
    for bruta in text.splitlines():
        linha = _limpa_linha(bruta)
        if not linha:
            continue
        titulo, primeira = primeira, False
        palavras = re.findall(r"[A-Za-zÀ-ÿ0-9]+", linha)
        for m in _NUM_RE.finditer(linha):  # vale também em linha de caixa alta ("LEI COMPLEMENTAR Nº 179")
            tipo = m.group(1).capitalize() + (" Complementar" if m.group(2) else "")
            add(f"{tipo} {m.group(3).rstrip('.')}", "numero", titulo)
        if _so_maiusculas(linha):
            uteis = [p for p in palavras if p not in _ROMANOS and not p.isdigit()]
            if len(uteis) == 1 and 3 <= len(uteis[0]) <= 7:
                add(uteis[0], "sigla", titulo)  # sigla sozinha na linha ("CAPM")
            elif 2 <= len(uteis) <= 4:
                add(" ".join(uteis).capitalize(), "nome", titulo)
            continue
        for m in _SIGLA_RE.finditer(linha):
            s = m.group(1)
            if s in _ROMANOS or s.isdigit() or (len(s) < 3 and not any(c.isdigit() for c in s)):
                continue
            add(s, "sigla", titulo)
        for m in _NOME_RE.finditer(linha):
            add(m.group(0), "nome", titulo)
        if titulo and len(palavras) == 1 and palavras[0][0].isupper() and not palavras[0].isupper():
            add(palavras[0], "nome", titulo)  # título de uma palavra só: "Dealers"
        nomes = " ".join(key(t["text"]) for t in termos if t["kind"] == "nome")
        for m in _PALAVRA_RE.finditer(linha):
            w = m.group(0)
            if not w.isupper() and key(w) not in _STOP and key(w) not in nomes:
                add(w, "termo", titulo)
    return termos


# ------------------------------------------------------------------ glossário
def build_glossary(slides: list[dict], boiler_frac: float = 0.5, max_terms: int = 40) -> list[dict]:
    """Glossário de cada slide. `slides`: [{"file", "t", "t_end", "visits"?, "texto"}] (frames.json + OCR).

    Devolve [{"slide", "t", "t_end", "visits", "terms": [{"text", "kind", "weight", "pron"}]}], com os
    termos em ordem decrescente de peso. O peso combina o tipo, a raridade do termo entre os slides do
    vídeo e estar no título. Termo presente em ≥ `boiler_frac` dos slides (rodapé, nome do curso) sai,
    desde que haja ≥ 3 slides. Slide sem texto fica na lista, com `terms` vazio (a linha do tempo não muda)."""
    por_slide = [extract_terms(s.get("texto") or "") for s in slides]
    n = len(slides)
    df: dict[str, int] = {}
    for termos in por_slide:
        for k in {key(t["text"]) for t in termos}:
            df[k] = df.get(k, 0) + 1
    saida = []
    for s, termos in zip(slides, por_slide, strict=True):
        itens = []
        for t in termos:
            k = key(t["text"])
            if n >= 3 and df[k] / n >= boiler_frac:
                continue
            raridade = 1.0 if n <= 1 else 1.0 - (df[k] - 1) / (n - 1)
            peso = KIND_WEIGHT[t["kind"]] * (0.6 + 0.4 * raridade) * (1.15 if t["titulo"] else 1.0)
            pron = []
            if t["kind"] == "sigla":
                pron = [spell(t["text"])] + ([key(t["text"])] if pronounceable(t["text"]) else [])
            itens.append({"text": t["text"], "kind": t["kind"], "weight": round(min(peso, 1.0), 4), "pron": pron})
        itens.sort(key=lambda x: -x["weight"])
        saida.append({"slide": s["file"], "t": s["t"], "t_end": s["t_end"],
                      "visits": s.get("visits") or [[s["t"], s["t_end"]]], "terms": itens[:max_terms]})
    return saida


def _mescla(listas: list[tuple[list[dict], float]], max_terms: int) -> list[dict]:
    """Une listas de termos [(termos, escala)], ficando com o maior peso de cada termo."""
    melhor: dict[str, dict] = {}
    for termos, escala in listas:
        for t in termos:
            k = key(t["text"])
            peso = round(t["weight"] * escala, 4)
            if k not in melhor or peso > melhor[k]["weight"]:
                melhor[k] = {**t, "weight": peso}
    return sorted(melhor.values(), key=lambda x: -x["weight"])[:max_terms]


def plan_intervals(glossary: list[dict], duration_s: float, lead_s: float = 2.0, lag_s: float = 1.0,
                   carry: bool = True, min_interval_s: float = 4.0, neighbor_weight: float = 0.5,
                   max_terms: int = 30) -> list[dict]:
    """Particiona [0, duration_s] em intervalos contíguos, cada um com os termos do slide visível.

    O slide vale de `t − lead_s` até `t_end + lag_s` (quem apresenta cita o termo antes de trocar o slide
    e demora a largar o anterior); entre dois slides distantes, o espaço do meio é um intervalo sem slide
    que herda os termos do último (`carry`) ou fica vazio. Slides vizinhos entram com `neighbor_weight` do
    peso. Intervalos com menos de `min_interval_s` são absorvidos pelo anterior (ou pelo seguinte).
    Cada visita a um slide (`visits`) vira um intervalo. Devolve [{"t0", "t1", "slide", "terms"}]."""
    dur = float(duration_s)
    visitas = sorted((max(0.0, float(a)), min(dur, float(b)), gi) for gi, g in enumerate(glossary)
                     for a, b in (g.get("visits") or [[g["t"], g["t_end"]]]) if min(dur, float(b)) > max(0.0, float(a)))
    if not visitas:
        return [{"t0": 0.0, "t1": dur, "slide": None, "terms": []}]
    # fronteiras entre visitas consecutivas
    inicio, fim = [0.0] * len(visitas), [0.0] * len(visitas)
    inicio[0] = max(0.0, visitas[0][0] - lead_s)
    for i, (a, b, _) in enumerate(visitas):
        if i + 1 == len(visitas):
            fim[i] = min(dur, b + lag_s)
            break
        folga = visitas[i + 1][0] - b
        if folga >= lead_s + lag_s:
            fim[i], inicio[i + 1] = b + lag_s, visitas[i + 1][0] - lead_s
        else:
            meio = b + (folga * lag_s / (lead_s + lag_s) if lead_s + lag_s > 0 else 0.0)
            fim[i] = inicio[i + 1] = meio
    brutos: list[list] = []  # [t0, t1, índice do glossário ou None]
    if inicio[0] > 0:
        brutos.append([0.0, inicio[0], None])
    for i, (_, _, gi) in enumerate(visitas):
        brutos.append([inicio[i], fim[i], gi])
        if i + 1 < len(visitas) and fim[i] < inicio[i + 1]:
            brutos.append([fim[i], inicio[i + 1], None])
    if fim[-1] < dur:
        brutos.append([fim[-1], dur, None])
    # termos de cada intervalo
    ultimo: int | None = None
    for b in brutos:
        if b[2] is not None:
            ultimo = b[2]
        elif carry and ultimo is not None:
            b.append(ultimo)  # intervalo sem slide: herda o último
    # absorve intervalos curtos
    i = 0
    while len(brutos) > 1 and i < len(brutos):
        if brutos[i][1] - brutos[i][0] < min_interval_s:
            j = i - 1 if i > 0 else i + 1
            a, b = sorted((i, j))
            brutos[a][1] = brutos[b][1]
            brutos[a][0] = min(brutos[a][0], brutos[b][0])
            if brutos[a][2] is None:
                brutos[a][2] = brutos[b][2]
            del brutos[b]
            i = max(0, a - 1)
            continue
        i += 1
    # monta os termos, com os vizinhos
    seq = [b[2] if b[2] is not None else (b[3] if len(b) > 3 else None) for b in brutos]
    saida = []
    for idx, b in enumerate(brutos):
        gi = seq[idx]
        listas: list[tuple[list[dict], float]] = []
        if gi is not None:
            listas.append((glossary[gi]["terms"], 1.0))
        for viz in (idx - 1, idx + 1):
            if 0 <= viz < len(brutos) and seq[viz] is not None and seq[viz] != gi:
                listas.append((glossary[seq[viz]]["terms"], neighbor_weight))
        saida.append({"t0": round(b[0], 3), "t1": round(b[1], 3),
                      "slide": glossary[b[2]]["slide"] if b[2] is not None else None,
                      "terms": _mescla(listas, max_terms)})
    saida[0]["t0"], saida[-1]["t1"] = 0.0, round(dur, 3)
    return saida


def global_terms(glossary: list[dict], k: int = 30) -> list[dict]:
    """Os k termos de maior peso do vídeo todo (modo G). Termo em vários slides ganha um pequeno bônus."""
    juntos: dict[str, dict] = {}
    for g in glossary:
        for t in g["terms"]:
            kk = key(t["text"])
            if kk in juntos:
                juntos[kk]["n"] += 1
                juntos[kk]["weight"] = max(juntos[kk]["weight"], t["weight"])
            else:
                juntos[kk] = {**t, "n": 1}
    pontuados = [{**t, "weight": round(min(1.0, t["weight"] * (1 + 0.1 * math.log(t["n"]))), 4)}
                 for t in juntos.values()]
    for t in pontuados:
        t.pop("n", None)
    return sorted(pontuados, key=lambda x: -x["weight"])[:k]


def hotwords(terms: list[dict], max_chars: int = MAX_HOTWORD_CHARS) -> str:
    """Termos separados por vírgula, em ordem de peso, até `max_chars` (o faster-whisper trunca em 223 tokens)."""
    saida = ""
    for t in sorted(terms, key=lambda x: -x["weight"]):
        candidato = f"{saida}, {t['text']}" if saida else t["text"]
        if len(candidato) > max_chars:
            break
        saida = candidato
    return saida


def placebo_like(terms: list[dict], other_glossary: list[dict], exclude: set[str], seed: int) -> list[dict]:
    """Glossário placebo (modo P): para cada termo real, um termo do mesmo tipo vindo de OUTRO vídeo e fora
    de `exclude` (chaves dos termos que de fato aparecem no trecho: `terms.txt` e o glossário real). Mesmo
    tamanho e mesma mistura de tipos, sem nenhuma relação com o que é dito. Determinístico pela seed."""
    pool: dict[str, list[dict]] = {}
    for g in other_glossary:
        for t in g["terms"]:
            if key(t["text"]) not in exclude:
                pool.setdefault(t["kind"], []).append(t)
    rng = random.Random(seed)
    for lista in pool.values():
        lista.sort(key=lambda t: key(t["text"]))  # ordem estável antes de sortear
        rng.shuffle(lista)
    usados: set[str] = set()
    saida = []
    for t in terms:
        while pool.get(t["kind"]):
            c = pool[t["kind"]].pop()
            if key(c["text"]) not in usados:
                usados.add(key(c["text"]))
                saida.append({**c, "weight": t["weight"]})
                break
    return saida


def shift_glossary(glossary: list[dict], offset_s: float, duration_s: float) -> list[dict]:
    """Leva o glossário do tempo do vídeo para o do trecho: subtrai `offset_s` e recorta em [0, duration_s].
    Slides que não aparecem no trecho saem."""
    saida = []
    for g in glossary:
        visitas = [[max(0.0, a - offset_s), min(duration_s, b - offset_s)] for a, b in g["visits"]
                   if b - offset_s > 0 and a - offset_s < duration_s]
        if visitas:
            saida.append({**g, "visits": [[round(a, 3), round(b, 3)] for a, b in visitas],
                          "t": visitas[0][0], "t_end": visitas[-1][1]})
    return saida


def split_terms(terms: list[str], glossary: list[dict]) -> tuple[list[str], list[str]]:
    """Divide os termos humanos (`terms.txt`) em (os que aparecem em algum slide do glossário, os que não).
    Alimenta `acerto_termos_slide` e `acerto_termos_fora`; este último é o controle negativo: o contexto do
    slide não tem como ajudar o que não está na tela. Compara por `key` (sem caixa nem acento)."""
    nos_slides = {key(t["text"]) for g in glossary for t in g["terms"]}
    return [t for t in terms if key(t) in nos_slides], [t for t in terms if key(t) not in nos_slides]


def read_timed_terms(path: str | Path) -> list[tuple[float, str]]:
    """Gabarito parcial de termos (`data/gt/<video>.termos.tsv`): uma linha `t_s<TAB>termo` por ocorrência
    falada, escrita por quem **ouviu** o áudio; '#' comenta. Devolve [(t_s, termo)] em ordem de tempo."""
    saida = []
    for linha in Path(path).read_text(encoding="utf-8").splitlines():
        linha = linha.split("#", 1)[0].strip()
        if not linha:
            continue
        t, _, termo = linha.partition("\t")
        if not termo.strip():
            raise ValueError(f"linha sem termo em {path}: {linha!r} (use t_s<TAB>termo)")
        saida.append((float(t.replace(",", ".")), termo.strip()))
    return sorted(saida)


def save_glossary(path: str | Path, glossary: list[dict]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(glossary, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_glossary(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ------------------------------------------------------------------ CLI
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("glossary", help="slides + OCR + glossário de um vídeo")
    g.add_argument("video")
    g.add_argument("--out", help="pasta (padrão: data/derived/<vídeo>)")
    g.add_argument("--ocr-json", help="usa este OCR (lista de slides com 'texto') em vez de rodar o OCR")
    g.add_argument("--motor", default="tesseract", choices=["tesseract", "easyocr"])
    g.add_argument("--psm", type=int, default=6)
    g.add_argument("--panel", choices=["auto", "off"], default="auto")
    a = ap.parse_args()

    import extract

    root = Path(__file__).resolve().parent.parent
    video = Path(a.video)
    out = Path(a.out) if a.out else root / "data" / "derived" / video.stem
    frames_dir = out / "frames"
    if not (frames_dir / "frames.json").exists():
        print("Slides...")
        extract.slides(video, frames_dir, panel=a.panel)
    slides = json.loads((frames_dir / "frames.json").read_text(encoding="utf-8"))
    panel = json.loads((frames_dir / "panel.json").read_text(encoding="utf-8"))
    t0 = time.perf_counter()
    if a.ocr_json:
        ocr = json.loads(Path(a.ocr_json).read_text(encoding="utf-8"))
        slides = ocr["slides"] if isinstance(ocr, dict) else ocr
        meta = {"motor": "externo", "tempo_s": 0.0}
    else:
        import ocr as ocr_mod

        if motivo := ocr_mod.status(a.motor):
            raise SystemExit(f"OCR indisponível: {motivo}")
        print(f"OCR ({a.motor}) em {len(slides)} slides...")
        box, title_box = ocr_mod.title_region(panel)
        slides = ocr_mod.ocr_slides(frames_dir, slides, box or panel.get("ocr_box"), motor=a.motor,
                                    title_box=title_box, psm=a.psm)
        meta = {"motor": a.motor, "psm": a.psm, "tempo_s": round(sum(s["ocr_tempo_s"] for s in slides), 3)}
    glossario = build_glossary(slides)
    meta["glossario_s"] = round(time.perf_counter() - t0 - meta["tempo_s"], 3)
    (out / "ocr.json").write_text(json.dumps({**meta, "slides": slides}, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
    save_glossary(out / "glossario.json", glossario)
    for g_ in glossario:
        print(f"  {g_['slide']}  {g_['t']:7.1f}-{g_['t_end']:7.1f} s  " + ", ".join(t["text"] for t in g_["terms"][:8]))
    print(f"glossário em {out / 'glossario.json'}")


if __name__ == "__main__":
    main()
