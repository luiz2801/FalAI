"""Degradações controladas para os testes de robustez (RQ1 e RQ2, guia.md seção 4.4)."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Lê um WAV PCM 16 bits mono. Devolve (amostras em [-1, 1], taxa)."""
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1:
            raise ValueError(f"{path}: esperado WAV PCM 16 bits mono (use extract.extract_audio)")
        taxa = w.getframerate()
        dados = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return dados.astype(np.float32) / 32768.0, taxa


def write_wav(path: str | Path, x: np.ndarray, rate: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(x, -1.0, 1.0) * 32767.0).round().astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def snr_db(clean: np.ndarray, noisy: np.ndarray) -> float:
    """SNR medido entre o sinal limpo e a versão com ruído."""
    ruido = noisy.astype(np.float64) - clean.astype(np.float64)
    return 10 * np.log10(np.mean(clean.astype(np.float64) ** 2) / np.mean(ruido ** 2))


def add_noise_array(x: np.ndarray, snr: float, seed: int = 0) -> np.ndarray:
    """Soma ruído branco gaussiano com potência = potência do sinal / 10^(SNR/10); recorta em [-1, 1]."""
    rng = np.random.default_rng(seed)
    p_sinal = float(np.mean(x.astype(np.float64) ** 2))
    if p_sinal == 0:
        raise ValueError("sinal sem energia: SNR indefinido")
    ruido = rng.normal(0.0, np.sqrt(p_sinal / 10 ** (snr / 10)), size=x.shape)
    return np.clip(x + ruido, -1.0, 1.0).astype(np.float32)


def add_noise(in_wav: str | Path, out_wav: str | Path, snr: float, seed: int = 0) -> float:
    """Gera a versão com ruído de um WAV. Devolve o SNR efetivo medido (após recorte e
    quantização), que deve ficar a ±0,5 dB do alvo (guia.md, armadilha 14)."""
    x, taxa = read_wav(in_wav)
    y = add_noise_array(x, snr, seed)
    write_wav(out_wav, y, taxa)
    y_lido, _ = read_wav(out_wav)
    return snr_db(x, y_lido)


def degrade_image(img: np.ndarray, kind: str, level: float) -> np.ndarray:
    """Degrada um quadro (RQ2). kind: 'resize' (fator), 'blur' (kernel ímpar), 'jpeg' (qualidade)."""
    import cv2

    if kind == "resize":
        h, w = img.shape[:2]
        menor = cv2.resize(img, (max(1, int(w * level)), max(1, int(h * level))), interpolation=cv2.INTER_AREA)
        return cv2.resize(menor, (w, h), interpolation=cv2.INTER_LINEAR)
    if kind == "blur":
        k = int(level)
        if k < 1 or k % 2 == 0:
            raise ValueError("blur: o kernel deve ser ímpar e positivo")
        return cv2.GaussianBlur(img, (k, k), 0)
    if kind == "jpeg":
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(level)])
        if not ok:
            raise RuntimeError("falha ao codificar JPEG")
        return cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    raise ValueError(f"degradação desconhecida: {kind!r} (use resize, blur ou jpeg)")
