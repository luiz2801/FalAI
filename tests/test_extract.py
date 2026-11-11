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
