"""Extração de áudio e de keyframes dos slides (F1 e F2, guia.md seção 4.1)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000


def _ffmpeg(*args: str) -> subprocess.CompletedProcess:
    proc = subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y", *args],
                          capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg falhou: {proc.stderr.decode(errors='replace')[-800:]}")
    return proc


def extract_audio(video: str | Path, out_wav: str | Path) -> Path:
    """Áudio mono, 16 kHz, PCM 16 bits (o formato de entrada do Whisper)."""
    Path(out_wav).parent.mkdir(parents=True, exist_ok=True)
    _ffmpeg("-i", str(video), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(out_wav))
    return Path(out_wav)


def clip_audio(media: str | Path, out_wav: str | Path, start_s: float, dur_s: float) -> Path:
    """Recorta o trecho com gabarito. Corta o áudio antes de transcrever, para que
    gabarito e hipótese cubram exatamente o mesmo intervalo."""
    Path(out_wav).parent.mkdir(parents=True, exist_ok=True)
    _ffmpeg("-ss", str(start_s), "-t", str(dur_s), "-i", str(media), "-vn", "-ac", "1",
            "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(out_wav))
    return Path(out_wav)


def load_audio(media: str | Path) -> np.ndarray:
    """Decodifica qualquer mídia para float32, 16 kHz, mono, em memória.

    O faster-whisper recebe esse array em vez do caminho: assim ele não usa o PyAV
    (o faster-whisper 1.2.1 é incompatível com o PyAV 19)."""
    proc = _ffmpeg("-i", str(media), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-")
    return np.frombuffer(proc.stdout, dtype=np.float32)


def media_duration(media: str | Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                          str(media)], capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def keyframes(video: str | Path, out_dir: str | Path, step_s: float = 1.0, thr: float = 0.04) -> list[dict]:
    """Amostra um quadro a cada step_s segundos; guarda quando a diferença média para o último
    keyframe (quadro 320×180 em cinza, normalizada em 0–1) passa de thr.

    Grava NNNN.png (quadro original) e frames.json ([{"t", "file"}]) e devolve a lista."""
    import cv2

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)  # cv2.imwrite não cria diretório
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV não abriu {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    passo = max(1, round(fps * step_s))
    frames, ultimo, n = [], None, 0
    while True:
        ok = cap.grab()  # leitura sequencial: evita o seek a cada amostra (armadilha 7.2.4)
        if not ok:
            break
        if n % passo == 0:
            ok, quadro = cap.retrieve()
            if not ok:
                break
            pequeno = cv2.cvtColor(cv2.resize(quadro, (320, 180), interpolation=cv2.INTER_AREA),
                                   cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            if ultimo is None or float(np.mean(np.abs(pequeno - ultimo))) > thr:
                arquivo = out_dir / f"{len(frames):04d}.png"
                if not cv2.imwrite(str(arquivo), quadro):
                    raise RuntimeError(f"falha ao gravar {arquivo}")
                frames.append({"t": round(n / fps, 3), "file": arquivo.name})
                ultimo = pequeno
        n += 1
    cap.release()
    (out_dir / "frames.json").write_text(json.dumps(frames, ensure_ascii=False, indent=2), encoding="utf-8")
    return frames
