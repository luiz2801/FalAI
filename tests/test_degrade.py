import numpy as np
import pytest

import degrade


def _tom(tmp_path, seg=2.0, taxa=16000):
    t = np.arange(int(seg * taxa)) / taxa
    x = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    p = tmp_path / "tom.wav"
    degrade.write_wav(p, x, taxa)
    return p


@pytest.mark.parametrize("snr", [20, 10, 0])
def test_add_noise_snr_medido_proximo_do_alvo(tmp_path, snr):
    medido = degrade.add_noise(_tom(tmp_path), tmp_path / f"r{snr}.wav", snr, seed=0)
    assert abs(medido - snr) <= 0.5


def test_add_noise_mesma_semente_mesmo_resultado(tmp_path):
    entrada = _tom(tmp_path)
    degrade.add_noise(entrada, tmp_path / "a.wav", 10, seed=0)
    degrade.add_noise(entrada, tmp_path / "b.wav", 10, seed=0)
    degrade.add_noise(entrada, tmp_path / "c.wav", 10, seed=1)
    a, b, c = (degrade.read_wav(tmp_path / f"{n}.wav")[0] for n in "abc")
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_degrade_image_mantem_tamanho_e_recusa_tipo_desconhecido():
    img = (np.random.default_rng(0).random((90, 160, 3)) * 255).astype(np.uint8)
    for kind, level in [("resize", 0.5), ("blur", 9), ("jpeg", 30)]:
        assert degrade.degrade_image(img, kind, level).shape == img.shape
    with pytest.raises(ValueError):
        degrade.degrade_image(img, "sepia", 1)
    with pytest.raises(ValueError):
        degrade.degrade_image(img, "blur", 4)
