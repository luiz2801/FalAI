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
