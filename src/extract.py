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


# --------------------------------------------------------------------------------------------------
# Slides completos (F2): um quadro por slide, o último antes da troca (docs/decisoes.md, 2026-11-12)
# --------------------------------------------------------------------------------------------------

ANALYSIS_SIZE = (480, 270)  # largura × altura da análise; só para comparar quadros
GRID = (16, 9)              # células (colunas × linhas) para detectar regiões instáveis (professor)
TITLE_KEPT = 0.95          # fração do cabeçalho que tem de coincidir para ser "o mesmo título"
INK_DELTA = 0.12            # diferença para o fundo (0–1) a partir da qual um pixel conta como "tinta"


def _rect_mask(rects, shape) -> np.ndarray:
    """Máscara booleana (altura, largura) com os retângulos (x0, y0, x1, y1) em frações do quadro."""
    m = np.zeros(shape, dtype=bool)
    h, w = shape
    for x0, y0, x1, y1 in rects:
        m[int(y0 * h):int(np.ceil(y1 * h)), int(x0 * w):int(np.ceil(x1 * w))] = True
    return m


class SlideTracker:
    """Agrupa os quadros amostrados em episódios de slide e guarda o quadro mais completo de cada um.

    Cada quadro vira uma máscara de "tinta" (pixels diferentes do fundo). A comparação com o quadro
    candidato do episódio usa duas medidas, que separam bullet novo de slide novo:
      mantido = fração da tinta do candidato que continua presente (com tolerância de 1 px);
      acrescido = fração da tinta atual que é nova.
    mantido >= keep_thr: é o mesmo slide; com acrescido > grow_eps, o quadro atual vira o candidato
    (o slide cresceu). mantido < keep_thr, ou o cabeçalho mudou: slide novo, e o candidato do episódio
    anterior é gravado (o último quadro estável antes da troca: o "buffer reverso").

    Só quadros estáveis (iguais aos `min_stable` amostras anteriores) são considerados; assim, fades,
    mão e cursor passando não entram. Células da grade que mudam o tempo todo (professor num canto) são
    ignoradas na comparação, e `ignore` permite fixar regiões à mão. Slides revisitados não geram
    arquivo novo: a visita é registrada em `visits`.
    """

    def __init__(self, out_dir: str | Path | None = None, keep_thr: float = 0.9, grow_eps: float = 0.01,
                 static_eps: float = 0.02, min_stable: int = 2, min_ink: float = 0.002,
                 min_slide_s: float = 2.0, max_ink: float = 0.30, min_flat: float = 0.35, title_frac: float = 0.22, roi=(0, 0, 1, 1), ignore=()):
        import cv2

        self._cv2 = cv2
        self.out_dir = Path(out_dir) if out_dir else None
        if self.out_dir:
            self.out_dir.mkdir(parents=True, exist_ok=True)  # cv2.imwrite não cria diretório
        self.keep_thr, self.grow_eps, self.static_eps = keep_thr, grow_eps, static_eps
        self.min_stable, self.min_ink, self.min_slide_s, self.max_ink = min_stable, min_ink, min_slide_s, max_ink
        self.min_flat = min_flat
        w, h = ANALYSIS_SIZE
        self._shape = (h, w)
        self._fixed_valid = _rect_mask([roi], self._shape) & ~_rect_mask(ignore, self._shape)
        self._title = np.zeros(self._shape, dtype=bool)
        y0, y1 = int(roi[1] * h), int(np.ceil(roi[3] * h))
        self._title[y0:y0 + max(1, int(title_frac * (y1 - y0)))] = True
        self._kernel = np.ones((3, 3), np.uint8)
        self._cell_h, self._cell_w = h // GRID[1], w // GRID[0]
        self._ema = np.zeros((GRID[1], GRID[0]), dtype=np.float32)
        self._volatile = np.zeros((GRID[1], GRID[0]), dtype=bool)
        self._valid_cache: np.ndarray | None = None
        self._prev: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._stable, self._run_start = 0, 0.0
        self._ep: dict | None = None
        self.slides: list[dict] = []

    # --- análise de quadro -------------------------------------------------------------------------
    def _ink(self, frame: np.ndarray):
        cv2 = self._cv2
        g = cv2.cvtColor(cv2.resize(frame, ANALYSIS_SIZE, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        g = g.astype(np.int16)
        ink = np.abs(g - int(np.median(g))) > INK_DELTA * 255
        return ink, cv2.dilate(ink.astype(np.uint8), self._kernel).astype(bool), g

    def _valid(self) -> np.ndarray:
        if self._valid_cache is None:
            vol = np.kron(self._volatile, np.ones((self._cell_h, self._cell_w), dtype=bool))
            valid = self._fixed_valid & ~vol
            # Se quase tudo estiver "instável" (ex.: troca de slide com fade), a máscara não é confiável
            self._valid_cache = valid if valid.sum() >= 0.5 * self._fixed_valid.sum() else self._fixed_valid
        return self._valid_cache

    @staticmethod
    def _kept(a: np.ndarray, b_dil: np.ndarray, valid: np.ndarray) -> float:
        n = int((a & valid).sum())
        return 1.0 if n == 0 else float((a & b_dil & valid).sum()) / n

    @staticmethod
    def _added(a: np.ndarray, b_dil: np.ndarray, valid: np.ndarray) -> float:
        n = int((a & valid).sum())
        return 0.0 if n == 0 else float((a & ~b_dil & valid).sum()) / n

    def _update_volatility(self, g, pg) -> None:
        # variação de cinza (e não de tinta): o vídeo do professor satura a máscara de tinta
        changed = np.abs(g - pg) > INK_DELTA * 255 * 0.7
        gh, gw = GRID[1], GRID[0]
        cells = changed[:gh * self._cell_h, :gw * self._cell_w].reshape(gh, self._cell_h, gw, self._cell_w)
        flagged = cells.mean(axis=(1, 3)) > 0.03
        if flagged.mean() > 0.5:  # mudança global (troca de slide): não é professor
            return
        self._ema = 0.8 * self._ema + 0.2 * flagged
        novo = np.where(self._ema > 0.4, True, np.where(self._ema < 0.15, False, self._volatile))
        if (novo != self._volatile).any():
            self._volatile, self._valid_cache = novo, None

    # --- episódios ---------------------------------------------------------------------------------
    def feed(self, t: float, frame: np.ndarray) -> None:
        """Entrega o quadro amostrado no instante t (s). Chame em ordem crescente de t."""
        ink, dil, g = self._ink(frame)
        if self._prev is None:
            self._prev, self._stable, self._run_start = (ink, dil, g), 0, t
        else:
            pink, pdil, pg = self._prev
            self._update_volatility(g, pg)
            valid = self._valid()
            moved = max(1 - self._kept(pink, dil, valid), self._added(ink, pdil, valid))
            if moved < self.static_eps:
                self._stable += 1
            else:
                self._stable, self._run_start = 0, t
            self._prev = (ink, dil, g)
        if self._stable == self.min_stable:  # a corrida de quadros iguais acaba de ficar estável
            self._consider(self._run_start, frame, ink, dil, g)

    def _consider(self, t: float, frame, ink, dil, g) -> None:
        valid = self._valid()
        flat = float((np.abs(g - int(np.median(g))) <= 3).mean())  # fração de fundo liso (slide alto, foto baixo)
        frac = int((ink & valid).sum()) / max(int(valid.sum()), 1)
        if frac < self.min_ink or frac > self.max_ink or flat < self.min_flat:  # sem conteúdo, ou imagem natural (professor em tela cheia)
            self._commit(t)
            return
        ep = self._ep
        if ep is None:
            self._open(t, frame, ink, dil)
            return
        band = self._title & valid
        n_title = int((ep["ink"] & band).sum())
        title_changed = n_title >= 20 and self._kept(ep["ink"] & band, dil, band) < TITLE_KEPT
        if self._kept(ep["ink"], dil, valid) >= self.keep_thr and not title_changed:
            if self._added(ink, ep["dil"], valid) > self.grow_eps:  # o slide cresceu: este quadro é mais completo
                ep.update(image=frame.copy(), ink=ink, dil=dil, t_complete=t, n_stages=ep["n_stages"] + 1)
        else:
            self._commit(t)
            self._open(t, frame, ink, dil)

    def _open(self, t, frame, ink, dil) -> None:
        self._ep = {"t_start": t, "t_complete": t, "n_stages": 1, "image": frame.copy(), "ink": ink, "dil": dil}

    def _commit(self, t_end: float) -> None:
        ep, self._ep = self._ep, None
        if ep is None or t_end - ep["t_start"] < self.min_slide_s:  # relance transitório
            return
        valid = self._valid()
        band = self._title & valid
        for s in self.slides:  # slide revisitado: registra a visita, sem arquivo novo
            novo_em_s = self._kept(ep["ink"], s["dil"], valid) >= self.keep_thr  # o novo cabe no antigo
            s_em_novo = self._kept(s["ink"], ep["dil"], valid) >= self.keep_thr  # o antigo cabe no novo
            mesmo_titulo = (self._kept(ep["ink"] & band, s["dil"], band) >= TITLE_KEPT
                            and self._kept(s["ink"] & band, ep["dil"], band) >= TITLE_KEPT)
            if not ((novo_em_s and s_em_novo) or ((novo_em_s or s_em_novo) and mesmo_titulo)):
                continue
            s["visits"].append([round(ep["t_start"], 3), round(t_end, 3)])
            s["t_end"] = round(t_end, 3)
            if s_em_novo and not novo_em_s:  # a volta trouxe o slide mais completo: este quadro substitui o anterior
                s.update(ink=ep["ink"], dil=ep["dil"], t_complete=round(ep["t_complete"], 3),
                         n_stages=max(s["n_stages"], ep["n_stages"]))
                self._write(s, ep["image"])
            return
        slide = {"t": round(ep["t_start"], 3), "t_complete": round(ep["t_complete"], 3), "t_end": round(t_end, 3),
                 "n_stages": ep["n_stages"], "visits": [[round(ep["t_start"], 3), round(t_end, 3)]],
                 "ink": ep["ink"], "dil": ep["dil"]}
        slide["file"] = f"{len(self.slides):04d}.png"
        self._write(slide, ep["image"])
        self.slides.append(slide)

    def _write(self, slide: dict, image: np.ndarray) -> None:
        if not self.out_dir:
            slide["image"] = image
            return
        arquivo = self.out_dir / slide["file"]
        if not self._cv2.imwrite(str(arquivo), image):
            raise RuntimeError(f"falha ao gravar {arquivo}")

    def finish(self, t_end: float) -> list[dict]:
        """Fecha o último episódio e devolve os slides (sem as máscaras internas)."""
        self._commit(t_end)
        return [{k: v for k, v in s.items() if k not in ("ink", "dil")} for s in self.slides]


def slides(video: str | Path, out_dir: str | Path, step_s: float = 0.5, **kwargs) -> list[dict]:
    """Um quadro por slide, o mais completo (o último antes da troca), mesmo que o professor
    revele os bullets aos poucos. Leitura sequencial, amostrando a cada step_s segundos.

    Grava NNNN.png (quadro original) e frames.json e devolve a lista, com, por slide:
    t (início do slide, para o clique da página), t_complete (quando ficou completo), t_end,
    n_stages (estados estáveis: 1 + bullets revelados), file e visits ([[início, fim], ...],
    mais de um se o professor voltou ao slide). kwargs vão para SlideTracker
    (keep_thr, grow_eps, min_stable, min_slide_s, roi, ignore...)."""
    import cv2

    out_dir = Path(out_dir)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV não abriu {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    passo = max(1, round(fps * step_s))
    tracker, n = SlideTracker(out_dir, **kwargs), 0
    while cap.grab():  # leitura sequencial: evita o seek a cada amostra
        if n % passo == 0:
            ok, quadro = cap.retrieve()
            if not ok:
                break
            tracker.feed(n / fps, quadro)
        n += 1
    cap.release()
    resultado = tracker.finish(n / fps)
    (out_dir / "frames.json").write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    return resultado


def _rect(texto: str) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = (float(v) for v in texto.split(","))
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError(f"retângulo inválido: {texto} (use x0,y0,x1,y1 entre 0 e 1, com x0<x1 e y0<y1)")
    return x0, y0, x1, y1


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Extrai um quadro completo por slide de um vídeo de aula.")
    ap.add_argument("video")
    ap.add_argument("--out", help="pasta de saída (padrão: data/derived/<nome do vídeo>/frames)")
    ap.add_argument("--step-s", type=float, default=0.5, help="intervalo de amostragem, em segundos")
    ap.add_argument("--keep-thr", type=float, default=0.9, help="tinta mantida para ser o mesmo slide")
    ap.add_argument("--min-slide-s", type=float, default=2.0, help="duração mínima de um slide")
    ap.add_argument("--roi", type=_rect, help="região analisada: x0,y0,x1,y1 em frações do quadro")
    ap.add_argument("--ignore", type=_rect, action="append", default=[],
                    help="região a ignorar (professor, player): x0,y0,x1,y1; pode repetir")
    a = ap.parse_args()
    video = Path(a.video)
    out = Path(a.out) if a.out else Path(__file__).resolve().parent.parent / "data" / "derived" / video.stem / "frames"
    extra = {"roi": a.roi} if a.roi else {}
    res = slides(video, out, a.step_s, keep_thr=a.keep_thr, min_slide_s=a.min_slide_s, ignore=a.ignore, **extra)
    print(f"{len(res)} slides em {out}")
    for i, s in enumerate(res):
        print(f"{s['file']}  início {s['t']:7.1f} s  completo {s['t_complete']:7.1f} s  fim {s['t_end']:7.1f} s  "
              f"estágios {s['n_stages']}  visitas {len(s['visits'])}")


if __name__ == "__main__":
    main()
