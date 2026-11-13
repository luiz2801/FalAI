import json
import subprocess

import numpy as np

import extract


def _video(tmp_path):
    """6 s alternando 2 cores a cada 2 s (três 'slides')."""
    p = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y",
                    "-f", "lavfi", "-i", "color=c=red:s=320x180:d=2:r=10",
                    "-f", "lavfi", "-i", "color=c=blue:s=320x180:d=2:r=10",
                    "-f", "lavfi", "-i", "color=c=red:s=320x180:d=2:r=10",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=6",
                    "-filter_complex", "[0][1][2]concat=n=3:v=1:a=0[v]", "-map", "[v]", "-map", "3",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-shortest", str(p)], check=True)
    return p


def test_keyframes_detecta_trocas_e_cria_pasta(tmp_path):
    v = _video(tmp_path)
    out = tmp_path / "nao_existe" / "frames"  # pasta inexistente é criada
    frames = extract.keyframes(v, out, step_s=1.0, thr=0.04)
    assert [round(f["t"]) for f in frames] == [0, 2, 4]
    assert json.loads((out / "frames.json").read_text()) == frames
    assert all((out / f["file"]).exists() for f in frames)


def test_audio_16k_mono(tmp_path):
    v = _video(tmp_path)
    wav = extract.extract_audio(v, tmp_path / "a.wav")
    from degrade import read_wav

    x, taxa = read_wav(wav)
    assert taxa == 16000 and abs(len(x) / taxa - 6) < 0.1
    trecho = extract.clip_audio(v, tmp_path / "c.wav", 1.0, 2.0)
    assert abs(len(read_wav(trecho)[0]) / 16000 - 2.0) < 0.05
    arr = extract.load_audio(v)
    assert arr.dtype == np.float32 and abs(len(arr) / 16000 - 6) < 0.1


# --- slides completos (extract.SlideTracker / extract.slides) -------------------------------------

def _slide(titulo, bullets):
    import cv2

    img = np.full((540, 960, 3), 255, np.uint8)
    cv2.putText(img, titulo, (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (60, 0, 160), 3)
    for i, b in enumerate(bullets):
        cv2.putText(img, f"- {b}", (60, 160 + 70 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (120, 70, 20), 2)
    return img


def _roteiro():
    """(duração_s, quadro): slide A revelando 3 bullets, slide B, volta ao A completo."""
    a = ["Decidir se registra", "Registrar a transacao", "Conferir a equacao"]
    return [(2, _slide("Transacoes", [])), (3, _slide("Transacoes", a[:1])), (3, _slide("Transacoes", a[:2])),
            (3, _slide("Transacoes", a)), (4, _slide("Balanco", ["Ativo = Passivo + PL"])),
            (3, _slide("Transacoes", a))]


def _alimenta(tracker, roteiro, step=0.5, mexe=None):
    t, k = 0.0, 0
    for dur, img in roteiro:
        for _ in range(int(dur / step)):
            quadro = img.copy() if mexe is None else mexe(img.copy(), k)
            tracker.feed(t, quadro)
            t, k = t + step, k + 1
    return tracker.finish(t)


def test_slides_guarda_o_quadro_completo_de_cada_slide():
    res = _alimenta(extract.SlideTracker(), _roteiro())
    assert len(res) == 2  # A (revisitado, sem duplicar) e B
    a, b = res
    assert np.array_equal(a["image"], _roteiro()[3][1])  # o quadro com os 3 bullets, não o primeiro
    assert a["n_stages"] == 4 and a["t"] == 0.0 and abs(a["t_complete"] - 8.0) < 1.01
    assert len(a["visits"]) == 2 and a["visits"][1][0] >= 13  # a volta ao slide A vira visita
    assert np.array_equal(b["image"], _roteiro()[4][1]) and abs(b["t"] - 11.0) < 1.01


def test_slides_ignora_professor_no_canto():
    rng = np.random.default_rng(0)

    def professor(img, _k):  # webcam: ruído que muda a cada amostra, 15% da largura × 25% da altura
        img[380:515, 780:935] = rng.integers(0, 255, (135, 155, 3), dtype=np.uint8)
        return img

    res = _alimenta(extract.SlideTracker(), _roteiro(), mexe=professor)
    assert len(res) == 2 and res[0]["n_stages"] == 4


def test_slides_ignora_relance_transitorio():
    roteiro = _roteiro()
    flash = np.zeros((540, 960, 3), np.uint8)  # 1 amostra de tela preta no meio do slide A
    roteiro[3:4] = [(1.5, roteiro[3][1]), (0.5, flash), (1, roteiro[3][1])]
    res = _alimenta(extract.SlideTracker(), roteiro)
    assert len(res) == 2


def test_slides_grava_arquivos_e_json(tmp_path):
    import cv2

    p = tmp_path / "s.mp4"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 5, (960, 540))
    for dur, img in _roteiro()[:5]:
        for _ in range(dur * 5):
            w.write(img)
    w.release()
    out = tmp_path / "nao_existe" / "slides"
    res = extract.slides(p, out, step_s=0.5)
    assert [r["file"] for r in res] == ["0000.png", "0001.png"]
    assert json.loads((out / "frames.json").read_text()) == res
    final = cv2.imread(str(out / "0000.png"))
    assert final.shape == (540, 960, 3) and final.sum() < _roteiro()[1][1].sum()  # tem os 3 bullets (mais tinta)


def test_slides_ignora_professor_em_tela_cheia():
    rng = np.random.default_rng(1)
    foto = cv2_blur(rng.integers(0, 255, (540, 960, 3), dtype=np.uint8))  # imagem natural: muita "tinta"
    a = _slide("Transacoes", ["Decidir se registra"])
    res = _alimenta(extract.SlideTracker(), [(4, a), (4, foto), (4, a)])
    assert len(res) == 1 and len(res[0]["visits"]) == 2


def cv2_blur(img):
    import cv2

    return cv2.GaussianBlur(img, (0, 0), 6)


def test_slides_volta_mais_completa_substitui_a_parcial():
    a = ["Decidir se registra", "Registrar a transacao", "Conferir a equacao"]
    parcial, completo = _slide("Transacoes", a[:1]), _slide("Transacoes", a)
    outro = _slide("Balanco", ["Ativo = Passivo + PL"])
    res = _alimenta(extract.SlideTracker(), [(4, parcial), (4, outro), (4, completo)])
    assert len(res) == 2 and np.array_equal(res[0]["image"], completo) and len(res[0]["visits"]) == 2


# --- layout em painel (F10, Fase 1): moldura texturizada + painel preto + fita no topo ---------------

def _moldura(linhas, titulo="Titulo da fita", painel=(0.10, 0.12, 0.90, 0.86), seed=3):
    """Quadro 960×540 no layout do curso de teste: moldura amarela texturizada, painel preto (~51% do
    quadro) e fita cinza texturizada sobre a borda de cima do painel. A textura é a mesma em todos os
    quadros (seed fixa), como no vídeo real."""
    import cv2

    rng = np.random.default_rng(seed)
    h, w = 540, 960
    img = np.clip(np.array([30, 170, 190], np.int16) + rng.integers(-40, 40, (h, w, 1), dtype=np.int16),
                  0, 255).astype(np.uint8)
    x0, y0, x1, y1 = int(painel[0] * w), int(painel[1] * h), int(painel[2] * w), int(painel[3] * h)
    img[y0:y1, x0:x1] = 0
    fx0, fy0, fx1, fy1 = int(0.17 * w), int(0.03 * h), int(0.85 * w), int(0.22 * h)
    img[fy0:fy1, fx0:fx1] = np.clip(120 + rng.integers(-30, 30, (fy1 - fy0, fx1 - fx0, 1)), 0, 255)
    cv2.putText(img, titulo, (fx0 + 40, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    for i, linha in enumerate(linhas):
        cv2.putText(img, f"- {linha}", (int(0.14 * w), int(0.32 * h) + 62 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2)
    return img


_BULLETS_A = ["Orgao maximo do SFN que formula politicas", "Composicao: Fazenda, BACEN, Planejamento",
              "Equilibrar os meios de pagamento", "Regular o valor interno e externo da moeda"]


def _mediana_analise(img):
    import cv2

    pequeno = cv2.resize(img, extract.ANALYSIS_SIZE, interpolation=cv2.INTER_AREA)
    return int(np.median(cv2.cvtColor(pequeno, cv2.COLOR_BGR2GRAY)))


def test_slides_painel_perto_de_50_por_cento_nao_perde_o_slide():
    """Regressão de S2 (docs/plano_contexto_cruzado.md): o fundo liso era medido no quadro inteiro, e a
    mediana do quadro cai da cor do painel para a da moldura quando o texto novo tira o painel de 50%."""
    roi = (0.14, 0.26, 0.86, 0.82)
    estagios = [_moldura(_BULLETS_A[:k]) for k in (1, 2, 3, 4)]
    assert _mediana_analise(estagios[0]) == 0 and _mediana_analise(estagios[3]) > 0  # a precondição do defeito
    b = _moldura(["Banco Central do Brasil", "Executa as politicas do CMN"], titulo="Outro slide")
    res = _alimenta(extract.SlideTracker(roi=roi), [(3, estagios[0]), (3, estagios[1]), (3, estagios[2]),
                                                    (3, estagios[3]), (4, b)])
    assert len(res) == 2
    assert res[0]["n_stages"] == 4 and np.array_equal(res[0]["image"], estagios[3])  # o quadro completo
    assert np.array_equal(res[1]["image"], b)


def test_slides_com_painel_ainda_rejeita_camera_em_tela_cheia():
    rng = np.random.default_rng(1)
    camera = cv2_blur(rng.integers(0, 255, (540, 960, 3), dtype=np.uint8))
    a = _moldura(_BULLETS_A[:2])
    res = _alimenta(extract.SlideTracker(roi=(0.14, 0.26, 0.86, 0.82)), [(4, a), (4, camera), (4, a)])
    assert len(res) == 1 and len(res[0]["visits"]) == 2


def _iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def _camera():
    rng = np.random.default_rng(1)
    return cv2_blur(rng.integers(0, 255, (540, 960, 3), dtype=np.uint8))


def test_panel_detecta_o_painel_da_moldura():
    frames = [_moldura(_BULLETS_A[:k]) for k in (1, 2, 3, 4)] + [_moldura(["Outro"], titulo="B")]
    frames += [_camera(), _camera()]  # a câmera em tela cheia não pode atrapalhar
    det = extract.panel_from_frames(frames)
    assert det is not None and det["votes"] >= 4
    # painel preto: x 0,10–0,90 e y 0,12–0,86. A fita (y 0,03–0,22) cobre a borda de cima, então o
    # retângulo liso (o do rastreio) começa abaixo dela, e a caixa do OCR (a região inteira) inclui a fita
    assert _iou(det["roi"], (0.10, 0.22, 0.90, 0.86)) > 0.85
    assert det["ocr_box"][1] < det["roi"][1] - 0.04


def test_panel_nao_inventa_painel_em_slide_de_tela_cheia():
    cheios = [_slide("Transacoes", [f"item {k}"]) for k in range(5)]
    assert extract.panel_from_frames(cheios) is None


def test_panel_nao_confunde_faixa_de_titulo_com_moldura():
    def com_faixa(titulo):
        import cv2

        img = np.full((540, 960, 3), 255, np.uint8)
        img[:90] = (160, 60, 20)  # faixa de título azul, de ponta a ponta
        cv2.putText(img, titulo, (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
        cv2.putText(img, "- um bullet qualquer", (60, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (60, 60, 60), 2)
        return img

    assert extract.panel_from_frames([com_faixa(f"Slide {k}") for k in range(5)]) is None


def test_panel_so_camera_nao_tem_painel():
    assert extract.panel_from_frames([_camera()] * 6) is None


def _video_moldura(tmp_path):
    import cv2

    p = tmp_path / "moldura.mp4"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 5, (960, 540))
    b = _moldura(["Banco Central do Brasil", "Executa as politicas do CMN"], titulo="Outro slide")
    for img, dur in [(_moldura(_BULLETS_A[:k]), 3) for k in (1, 2, 3, 4)] + [(b, 4)]:
        for _ in range(dur * 5):
            w.write(img)
    w.release()
    return p


def test_slides_acha_o_painel_sozinho_e_grava_panel_json(tmp_path):
    v = _video_moldura(tmp_path)
    det = extract.find_panel(v)
    assert det is not None and _iou(det["roi"], (0.10, 0.22, 0.90, 0.86)) > 0.8
    out = tmp_path / "auto"
    res = extract.slides(v, out, step_s=0.5)  # padrão: panel="auto"
    assert len(res) == 2 and res[0]["n_stages"] >= 3
    guardado = json.loads((out / "panel.json").read_text())
    assert guardado["modo"] == "auto" and _iou(guardado["roi"], det["roi"]) > 0.99
    # sem o painel, a moldura inteira conta como tinta (> max_ink) e nenhum slide é aceito (S1 e S2)
    assert extract.slides(v, tmp_path / "off", step_s=0.5, panel="off") == []
    assert json.loads((tmp_path / "off" / "panel.json").read_text())["modo"] == "off"
