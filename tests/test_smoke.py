"""Testes de fumaça da etapa 0.

Não usam nenhum vídeo da disciplina: o áudio e a imagem são gerados aqui.
"""

import shutil
import subprocess

import pytest


def test_ffmpeg_gera_wav(tmp_path):
    assert shutil.which("ffmpeg"), "ffmpeg não está no PATH"
    out = tmp_path / "tom.wav"
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
            "-ar", "16000", "-ac", "1", str(out),
        ],
        check=True,
    )
    assert out.stat().st_size > 1000


def test_tesseract_tem_portugues():
    assert shutil.which("tesseract"), "tesseract não está no PATH"
    res = subprocess.run(["tesseract", "--list-langs"], capture_output=True, text=True, check=True)
    assert "por" in (res.stdout + res.stderr).split()


def test_jiwer_wer():
    import jiwer

    assert jiwer.wer("o banco central regula", "o banco central regula") == 0
    assert jiwer.wer("o banco central regula", "o banco central") == pytest.approx(0.25)


def test_faster_whisper_importa():
    from faster_whisper import WhisperModel

    assert WhisperModel is not None


def test_opencv_importa():
    import cv2

    assert cv2.__version__


def test_gpu_cuda_visivel():
    ctranslate2 = pytest.importorskip("ctranslate2")
    if ctranslate2.get_cuda_device_count() == 0:
        pytest.skip("sem GPU CUDA visível no contêiner (rodar com --gpus all, ver README)")
    assert ctranslate2.get_cuda_device_count() >= 1
