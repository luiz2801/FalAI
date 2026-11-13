import shutil

import numpy as np
import pytest

import ocr


def test_crop_box_em_fracoes_e_sem_caixa():
    img = np.arange(100 * 200 * 3, dtype=np.uint32).astype(np.uint8).reshape(100, 200, 3)
    assert ocr.crop_box(img, None) is img and ocr.crop_box(img, ()) is img
    c = ocr.crop_box(img, (0.1, 0.2, 0.6, 0.9))
    assert c.shape == (70, 100, 3) and np.array_equal(c, img[20:90, 20:120])


def test_preprocess_inverte_fundo_escuro_e_amplia_imagem_pequena():
    escuro = np.zeros((300, 600, 3), np.uint8)
    escuro[100:120, 50:300] = 255  # texto claro em fundo preto
    g, escala = ocr.preprocess(escuro)
    assert np.median(g) > 200 and g[330, 500] < 50  # texto escuro em fundo claro (coordenadas ×3)
    assert escala == 3.0 and g.shape == (900, 1800)  # 1400/300 > 3: limitado a 3×
    claro = np.full((1500, 600, 3), 240, np.uint8)
    g2, escala2 = ocr.preprocess(claro)
    assert escala2 == 1.0 and g2.shape == (1500, 600) and np.median(g2) == 240  # alta o bastante: nada muda
    _, escala3 = ocr.preprocess(np.full((700, 900, 3), 255, np.uint8), invert=False)
    assert escala3 == 2.0


def _data(*palavras):
    """Monta o dict de pytesseract.image_to_data: (texto, conf, bloco, parágrafo, linha, x, y)."""
    cols = ["text", "conf", "block_num", "par_num", "line_num", "left", "top"]
    d = {c: [p[i] for p in palavras] for i, c in enumerate(cols)}
    d["width"], d["height"] = [40] * len(palavras), [20] * len(palavras)
    return d


def test_lines_from_data_ordem_de_leitura_escala_e_filtros():
    d = _data(("", -1, 1, 1, 0, 0, 0), ("Órgão", 91.4, 1, 1, 1, 100, 60), ("máximo", 88.0, 1, 1, 1, 200, 60),
              ("CMN", 96.2, 1, 1, 0, 100, 20), ("ruído", 12.0, 2, 1, 1, 10, 300), ("  ", 50, 2, 1, 1, 0, 0))
    texto, palavras = ocr.lines_from_data(d, scale=2.0)
    assert texto == "CMN\nÓrgão máximo\nruído"  # linha 0 (CMN) antes da linha 1; vazio e conf −1 saem
    assert [w["text"] for w in palavras] == ["CMN", "Órgão", "máximo", "ruído"]
    assert palavras[1]["x"] == 50 and palavras[1]["y"] == 30 and palavras[1]["w"] == 20  # voltou à escala original
    texto2, _ = ocr.lines_from_data(d, min_conf=50)
    assert "ruído" not in texto2 and "CMN" in texto2


def test_lines_from_easy_agrupa_caixas_na_mesma_linha():
    def caixa(x, y, w=80, h=20):
        return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]

    res = [(caixa(300, 12), "Nacional", 0.9), (caixa(10, 10), "Conselho", 0.95), (caixa(10, 80), "Objetivos", 0.8),
           (caixa(10, 40), "", 0.9), (caixa(150, 11), "Monetário", 0.4)]
    texto, palavras = ocr.lines_from_easy(res)
    assert texto == "Conselho Monetário Nacional\nObjetivos" and palavras[0]["conf"] == 95.0
    assert ocr.lines_from_easy(res, min_conf=50)[0] == "Conselho Nacional\nObjetivos"
    assert ocr.lines_from_easy([]) == ("", [])


def test_status_motor_desconhecido_e_imagem_inexistente(tmp_path):
    assert "desconhecido" in ocr.status("xyz")
    with pytest.raises(FileNotFoundError):
        ocr.ocr_tesseract(tmp_path / "nao_existe.png")  # falha clara, antes de qualquer chamada ao binário
    with pytest.raises(ValueError):
        ocr.ocr_slides(tmp_path, [], motor="xyz")


@pytest.mark.skipif(shutil.which("tesseract") is None or ocr.status("tesseract") is not None,
                    reason="Tesseract com o idioma 'por' não está instalado")
def test_ocr_tesseract_le_texto_sintetico(tmp_path):
    import cv2

    img = np.zeros((400, 1200, 3), np.uint8)  # texto claro em fundo preto, como nos slides do curso de teste
    cv2.putText(img, "CMN BACEN Conselho Monetario Nacional", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)
    cv2.putText(img, "Objetivos do SFN", (30, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)
    arq = tmp_path / "0000.png"
    cv2.imwrite(str(arq), img)
    r = ocr.ocr_tesseract(arq)
    assert "BACEN" in r["texto"].upper() and "CMN" in r["texto"].upper() and r["tempo_s"] > 0
    assert r["words"] and all(0 <= w["conf"] <= 100 for w in r["words"])
    saida = ocr.ocr_slides(tmp_path, [{"file": "0000.png", "t": 0.0, "t_end": 5.0}])
    assert saida[0]["t_end"] == 5.0 and "BACEN" in saida[0]["texto"].upper() and saida[0]["ocr_conf"] > 0
