"""OCR dos slides (F2, etapa 3): Tesseract (`por`) e EasyOCR (opcional). Guia: docs/plano_contexto_cruzado.md, 5.2.

Os dois motores seguem a mesma interface: (imagem ou caminho, caixa) -> {"texto", "words", "tempo_s"}.
Tempo do OCR (guia.md, armadilha 7.1.6): conta o pré-processamento e o reconhecimento; **não** conta a
leitura do arquivo. Igual para os dois motores.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np

MOTORES = ("tesseract", "easyocr")
_easy_reader = None  # criado sob demanda: o Reader carrega o modelo e exige GPU só de importar (armadilha 7.2.1)


def status(motor: str = "tesseract", lang: str = "por") -> str | None:
    """Motivo pelo qual o motor não pode rodar (com o comando que resolve), ou None se pode."""
    if motor == "tesseract":
        if not shutil.which("tesseract"):
            return "o Tesseract não está no PATH (Arch: sudo pacman -S tesseract tesseract-data-por)"
        import pytesseract

        try:
            idiomas = pytesseract.get_languages(config="")
        except (OSError, RuntimeError) as e:  # binário ausente ou quebrado
            return f"não consegui listar os idiomas do Tesseract: {e}"
        return None if lang in idiomas else f"o Tesseract não tem o idioma '{lang}' (instale tesseract-data-por)"
    if motor == "easyocr":
        try:
            import easyocr  # noqa: F401
        except ImportError:
            return "o easyocr não está instalado (traz o PyTorch; ver README)"
        return None
    return f"motor desconhecido: {motor}"


def _load(img) -> np.ndarray:
    if isinstance(img, np.ndarray):
        return img
    import cv2

    quadro = cv2.imread(str(img))
    if quadro is None:  # cv2.imread devolve None e o OCR falharia com uma mensagem confusa (armadilha 7.2.7)
        raise FileNotFoundError(f"não consegui ler a imagem {img}")
    return quadro


def crop_box(img: np.ndarray, box) -> np.ndarray:
    """Recorta a caixa (x0, y0, x1, y1), em frações do quadro. `box` vazio devolve o quadro inteiro."""
    if not box:
        return img
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box
    return img[int(y0 * h):int(np.ceil(y1 * h)), int(x0 * w):int(np.ceil(x1 * w))]


def preprocess(img: np.ndarray, invert: str | bool = "auto", target_height: int = 1400,
               max_scale: float = 3.0) -> tuple[np.ndarray, float]:
    """Cinza, texto escuro sobre fundo claro (inverte se o fundo for escuro) e ampliação até ~target_height
    px de altura (no máximo max_scale×): o texto dos slides de 720p é pequeno para o Tesseract.
    Devolve (imagem em cinza, fator de escala)."""
    import cv2

    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    if invert == "auto":
        invert = float(np.median(g)) < 128
    if invert:
        g = 255 - g
    escala = min(max_scale, max(1.0, target_height / g.shape[0]))
    if escala > 1.0:
        g = cv2.resize(g, None, fx=escala, fy=escala, interpolation=cv2.INTER_CUBIC)
    return g, float(escala)


def lines_from_data(data: dict, scale: float = 1.0, min_conf: float = 0.0) -> tuple[str, list[dict]]:
    """Texto (uma linha por linha do Tesseract, na ordem de leitura dele) e palavras com confiança, a partir
    do dict de `pytesseract.image_to_data`. As coordenadas voltam à escala da imagem original. conf < 0 (o
    Tesseract marca assim o que não é palavra) e texto vazio saem."""
    linhas: dict[tuple, list[dict]] = {}
    for i, txt in enumerate(data["text"]):
        txt = (txt or "").strip()
        conf = float(data["conf"][i])
        if not txt or conf < 0 or conf < min_conf:
            continue
        chave = (int(data["block_num"][i]), int(data["par_num"][i]), int(data["line_num"][i]))
        linhas.setdefault(chave, []).append({
            "text": txt, "conf": round(conf, 1), "x": round(data["left"][i] / scale), "y": round(data["top"][i] / scale),
            "w": round(data["width"][i] / scale), "h": round(data["height"][i] / scale)})
    ordem = sorted(linhas)
    texto = "\n".join(" ".join(w["text"] for w in sorted(linhas[k], key=lambda w: w["x"])) for k in ordem)
    return texto, [w for k in ordem for w in sorted(linhas[k], key=lambda w: w["x"])]


def lines_from_easy(resultado: list, scale: float = 1.0, min_conf: float = 0.0) -> tuple[str, list[dict]]:
    """O mesmo para o EasyOCR: resultado de `readtext` = [(caixa de 4 pontos, texto, confiança 0–1)]. Junta
    na mesma linha as caixas cujo centro vertical difere menos que 0,6 da altura mediana."""
    itens = []
    for pontos, txt, conf in resultado:
        txt = (txt or "").strip()
        if not txt or conf * 100 < min_conf:
            continue
        xs, ys = [p[0] for p in pontos], [p[1] for p in pontos]
        itens.append({"text": txt, "conf": round(float(conf) * 100, 1), "x": round(min(xs) / scale),
                      "y": round(min(ys) / scale), "w": round((max(xs) - min(xs)) / scale),
                      "h": round((max(ys) - min(ys)) / scale)})
    if not itens:
        return "", []
    altura = float(np.median([w["h"] for w in itens])) or 1.0
    linhas: list[list[dict]] = []
    for w in sorted(itens, key=lambda w: w["y"] + w["h"] / 2):
        if linhas and abs((w["y"] + w["h"] / 2) - np.mean([v["y"] + v["h"] / 2 for v in linhas[-1]])) < 0.6 * altura:
            linhas[-1].append(w)
        else:
            linhas.append([w])
    ordenadas = [sorted(linha, key=lambda w: w["x"]) for linha in linhas]
    return "\n".join(" ".join(w["text"] for w in linha) for linha in ordenadas), [w for l in ordenadas for w in l]


def ocr_tesseract(img, box=None, psm: int = 6, lang: str = "por", pre: bool = True, min_conf: float = 0.0) -> dict:
    """OCR com o Tesseract. `box`: caixa em frações do quadro (None = quadro inteiro). `psm` 6 = um bloco
    uniforme de texto; 4 = colunas. Devolve {"texto", "words", "tempo_s"}."""
    import cv2
    import pytesseract
    from pytesseract import Output

    im = crop_box(_load(img), box)
    t0 = time.perf_counter()
    prep, escala = preprocess(im) if pre else (cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), 1.0)
    data = pytesseract.image_to_data(prep, lang=lang, config=f"--psm {psm}", output_type=Output.DICT)
    tempo = time.perf_counter() - t0
    texto, palavras = lines_from_data(data, escala, min_conf)
    return {"texto": texto, "words": palavras, "tempo_s": tempo}


def ocr_easyocr(img, box=None, pre: bool = True, min_conf: float = 0.0, gpu: bool = True, **_ignorado) -> dict:
    """OCR com o EasyOCR (`["pt"]`). O Reader é criado na primeira chamada e reaproveitado. Mesma interface
    do Tesseract. Não foi exercitado com o pacote real nesta máquina (não instalado): conferir no uso."""
    global _easy_reader
    import cv2
    import easyocr

    if _easy_reader is None:
        _easy_reader = easyocr.Reader(["pt"], gpu=gpu)
    im = crop_box(_load(img), box)
    t0 = time.perf_counter()
    prep, escala = preprocess(im) if pre else (cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), 1.0)
    resultado = _easy_reader.readtext(prep, detail=1, paragraph=False)
    tempo = time.perf_counter() - t0
    texto, palavras = lines_from_easy(resultado, escala, min_conf)
    return {"texto": texto, "words": palavras, "tempo_s": tempo}


def title_region(panel: dict | None) -> tuple[tuple | None, tuple | None]:
    """(caixa do corpo, caixa do título) a partir do panel.json. O título fica na fita texturizada que cobre a borda
    de cima do painel: no mesmo OCR do corpo ele se perde, e sozinho (psm 7) sai bem. Sem painel: (None, None)."""
    if not panel or not panel.get("roi"):
        return None, None
    y0 = panel["roi"][1]
    ox0, oy0, ox1, _ = panel.get("ocr_box") or panel["roi"]
    ty0 = max(0.0, oy0 - 0.12)
    return tuple(panel["roi"]), ((ox0, ty0, ox1, y0) if y0 - ty0 > 0.04 else None)


def ocr_slides(frames_dir: str | Path, slides: list[dict], box=None, motor: str = "tesseract", title_box=None,
               **kw) -> list[dict]:
    """OCR de cada slide de `frames.json`. Devolve os mesmos dicts com `texto`, `words`, `ocr_conf` (média das
    confianças das palavras, ou None) e `ocr_tempo_s`. A ordem dos slides é mantida."""
    if motor not in MOTORES:
        raise ValueError(f"motor desconhecido: {motor}")
    fn = ocr_tesseract if motor == "tesseract" else ocr_easyocr
    saida = []
    for s in slides:
        r = fn(Path(frames_dir) / s["file"], box, **kw)
        if title_box:  # título numa passada própria, como uma linha só
            t = fn(Path(frames_dir) / s["file"], title_box, **({**kw, "psm": 7} if motor == "tesseract" else kw))
            r = {"texto": (t["texto"].strip() + "\n" + r["texto"]).strip(), "words": t["words"] + r["words"],
                 "tempo_s": r["tempo_s"] + t["tempo_s"]}
        confs = [w["conf"] for w in r["words"]]
        saida.append({**s, "texto": r["texto"], "words": r["words"],
                      "ocr_conf": round(sum(confs) / len(confs), 1) if confs else None,
                      "ocr_tempo_s": round(r["tempo_s"], 4)})
    return saida
