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
                 min_slide_s: float = 2.0, max_ink: float = 0.30, min_flat: float = 0.35, title_frac: float = 0.22, roi=(0, 0, 1, 1), ignore=(),
                 trace=None):
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
        # Referência do fundo (mediana de cinza) e do "fundo liso": só a região analisada. Com o quadro
        # inteiro, um painel que ocupa ~50% do quadro (moldura texturizada em volta) fazia a mediana saltar
        # de cor quando o texto novo tirava o painel de 50%, e o filtro min_flat derrubava o slide
        # (docs/decisoes.md, D14). Sem roi nem ignore, a região é o quadro inteiro: resultado idêntico ao antigo.
        self._ref = self._fixed_valid if self._fixed_valid.any() else np.ones(self._shape, dtype=bool)
        self._ref_all = bool(self._ref.all())
        self._trace = trace  # opcional: função(dict) chamada a cada amostra estável (diagnóstico)
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
        ref = g if self._ref_all else g[self._ref]
        ink = np.abs(g - int(np.median(ref))) > INK_DELTA * 255
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
        ref = g if self._ref_all else g[self._ref]
        flat = float((np.abs(ref - int(np.median(ref))) <= 3).mean())  # fração de fundo liso (slide alto, foto baixo)
        frac = int((ink & valid).sum()) / max(int(valid.sum()), 1)
        # sem conteúdo, ou imagem natural (professor em tela cheia)
        rejeitado = frac < self.min_ink or frac > self.max_ink or flat < self.min_flat
        if self._trace:
            ep = self._ep
            self._trace({"t": t, "tinta": frac, "liso": flat, "rejeitado": rejeitado,
                         "mantido": None if ep is None else self._kept(ep["ink"], dil, valid),
                         "inicio": None if ep is None else ep["t_start"]})
        if rejeitado:
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


# --------------------------------------------------------------------------------------------------
# Painel do slide (F10, Fase 1): a região do quadro onde o slide aparece, sem moldura, miniatura nem rodapé
# --------------------------------------------------------------------------------------------------

PANEL_BLOCK = 6         # lado do bloco de análise (px em ANALYSIS_SIZE)
PANEL_FLAT_RANGE = 10   # amplitude de cinza (máx − mín) abaixo da qual um bloco é "liso"
PANEL_GRAY_BIN = 32     # largura da faixa de cinza: blocos lisos de tons diferentes são regiões diferentes
PANEL_MAX_STD = 8.0     # desvio máximo (níveis de cinza) do tom da região: a parede do estúdio tem gradiente
PANEL_COVER = 0.60      # uma linha/coluna só é do painel se a região a cobrir em ≥ 60% (fita e miniatura ficam de fora)
PANEL_MIN_AREA = 0.12   # fração mínima do quadro para o retângulo liso valer como painel
PANEL_MIN_INSET = 0.04  # o painel tem de ficar afastado ≥ 4% de pelo menos 3 das 4 bordas do quadro
PANEL_MAX_AREA = 0.85   # acima disso é slide em tela cheia: não há painel a recortar
PANEL_AGREE_IOU = 0.85  # dois quadros "votam" no mesmo painel se os retângulos tiverem IoU ≥ isto


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Preenche os buracos fechados de uma máscara (uint8 0/1): texto e diagramas dentro do painel. O que
    toca a borda não é buraco: a fita de título e a miniatura da câmera, que avançam sobre o painel a
    partir de fora, continuam de fora."""
    import cv2

    h, w = mask.shape
    pad = np.zeros((h + 2, w + 2), np.uint8)
    pad[1:-1, 1:-1] = mask
    fora = pad.copy()
    cv2.floodFill(fora, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 2)
    return (fora[1:-1, 1:-1] != 2).astype(np.uint8)


def _panel_candidate(gray: np.ndarray):
    """Num quadro em cinza (ANALYSIS_SIZE): a maior região lisa e de um só tom (faixa de cinza), com os
    buracos fechados preenchidos. Devolve (roi, caixa, (linhas, colunas)) em blocos, ou None:
      caixa  caixa da região inteira (inclui a fita de título que cobre a borda do painel);
      roi    a caixa aparada às linhas e colunas que a região cobre em ≥ 60%: sem a fita no topo, sem a
             miniatura da câmera nos cantos. Texto denso não encolhe o roi (os buracos são preenchidos).
    O painel de um slide é liso (fundo uniforme com texto esparso). A moldura, a fita de título e a
    miniatura da câmera têm textura, e as facetas lisas da moldura têm outro tom: ficam de fora."""
    import cv2

    b = PANEL_BLOCK
    gh, gw = gray.shape[0] // b, gray.shape[1] // b
    blocos = gray[:gh * b, :gw * b].reshape(gh, b, gw, b)
    liso = (blocos.max(axis=(1, 3)).astype(np.int16) - blocos.min(axis=(1, 3))) <= PANEL_FLAT_RANGE
    faixa = (blocos.mean(axis=(1, 3)) // PANEL_GRAY_BIN).astype(int)
    kernel = np.ones((5, 5), np.uint8)
    melhor, area_melhor = None, 0
    for v in np.unique(faixa[liso]):
        m = cv2.morphologyEx((liso & (faixa == v)).astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        n, rotulos, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < PANEL_MIN_AREA * gh * gw:
                continue
            reg = _fill_holes((rotulos == i).astype(np.uint8)).astype(bool)
            ys, xs = np.nonzero(reg)
            x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
            caixa = reg[y0:y1, x0:x1]
            linhas = np.flatnonzero(caixa.mean(axis=1) >= PANEL_COVER)
            if not len(linhas):
                continue
            top, bot = y0 + int(linhas[0]), y0 + int(linhas[-1]) + 1
            colunas = np.flatnonzero(reg[top:bot, x0:x1].mean(axis=0) >= PANEL_COVER)
            if not len(colunas):
                continue
            esq, dir_ = x0 + int(colunas[0]), x0 + int(colunas[-1]) + 1
            area = (dir_ - esq) * (bot - top)
            fracao = (esq / gw, top / gh, dir_ / gw, bot / gh)
            folgas = (fracao[0], fracao[1], 1 - fracao[2], 1 - fracao[3])
            # painel de verdade: não é o quadro todo, fica afastado de ≥ 3 das 4 bordas (moldura em volta) e
            # tem um tom só. Slide em tela cheia, faixa de título de ponta a ponta e parede de estúdio saem
            if (area > PANEL_MAX_AREA * gh * gw or sum(f >= PANEL_MIN_INSET for f in folgas) < 3
                    or float(blocos.mean(axis=(1, 3))[reg & liso].std()) > PANEL_MAX_STD):
                continue
            if area > area_melhor:
                melhor, area_melhor = ((esq, top, dir_, bot), (x0, y0, x1, y1), (gh, gw)), area
    return melhor if area_melhor >= PANEL_MIN_AREA * gh * gw else None


def _iou(a, b) -> float:
    iw, ih = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def panel_from_frames(frames, min_votes: int | None = None) -> dict | None:
    """Painel do slide a partir de quadros BGR amostrados ao longo do vídeo, ou None (usar o quadro inteiro).

    Cada quadro propõe um painel (`_panel_candidate`); vale o painel em que mais quadros concordam (IoU ≥
    PANEL_AGREE_IOU), com pelo menos `min_votes` votos (padrão: 4% dos quadros, mínimo 3). Cada quadro só
    propõe painel se ele ficar afastado de ≥ 3 das 4 bordas, ocupar menos de PANEL_MAX_AREA do quadro e
    tiver um tom só (`_panel_candidate`): slide em tela cheia, ou com faixa de título de ponta a ponta, não
    tem moldura e fica sem painel (comportamento antigo).

    Devolve {"roi", "ocr_box", "votes", "frames"}, retângulos (x0, y0, x1, y1) em frações do quadro:
      roi     retângulo liso, para o rastreio (`SlideTracker(roi=...)`);
      ocr_box caixa da região inteira, que inclui a fita de título que cobre a borda do painel."""
    import cv2

    cands = []
    for f in frames:
        g = cv2.cvtColor(cv2.resize(f, ANALYSIS_SIZE, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        c = _panel_candidate(g)
        if c:
            cands.append(c)
    votos_min = min_votes if min_votes is not None else max(3, round(0.04 * len(frames)))
    if len(cands) < votos_min:
        return None
    grupos = [[j for j, o in enumerate(cands) if _iou(c[0], o[0]) >= PANEL_AGREE_IOU] for c in cands]
    melhor = max(grupos, key=len)
    if len(melhor) < votos_min:
        return None
    gh, gw = cands[melhor[0]][2]
    rect = np.median([cands[j][0] for j in melhor], axis=0)
    caixa = np.median([cands[j][1] for j in melhor], axis=0)
    roi = (rect[0] / gw, rect[1] / gh, rect[2] / gw, rect[3] / gh)
    ocr = (caixa[0] / gw, caixa[1] / gh, caixa[2] / gw, caixa[3] / gh)
    return {"roi": tuple(round(float(v), 4) for v in roi), "ocr_box": tuple(round(float(v), 4) for v in ocr),
            "votes": len(melhor), "frames": len(frames)}


def find_panel(video: str | Path, n_samples: int = 60) -> dict | None:
    """Amostra `n_samples` quadros espaçados do vídeo (com seek: poucas amostras) e devolve
    `panel_from_frames`. None se o vídeo não tiver painel (ou não permitir seek)."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV não abriu {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    quadros = []
    if total > 0:
        for i in np.linspace(0.02 * total, 0.98 * total, n_samples).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, q = cap.read()
            if ok:
                quadros.append(q)
    cap.release()
    return panel_from_frames(quadros) if quadros else None


def slides(video: str | Path, out_dir: str | Path, step_s: float = 0.5, panel: str = "auto", **kwargs) -> list[dict]:
    """Um quadro por slide, o mais completo (o último antes da troca), mesmo que o professor
    revele os bullets aos poucos. Leitura sequencial, amostrando a cada step_s segundos.

    Grava NNNN.png (quadro original), frames.json e panel.json e devolve a lista, com, por slide:
    t (início do slide, para o clique da página), t_complete (quando ficou completo), t_end,
    n_stages (estados estáveis: 1 + bullets revelados), file e visits ([[início, fim], ...],
    mais de um se o professor voltou ao slide). kwargs vão para SlideTracker
    (keep_thr, grow_eps, min_stable, min_slide_s, roi, ignore...).

    panel: "auto" (padrão) procura o painel do slide (`find_panel`) quando `roi` não é dado; sem painel
    (slide em tela cheia), analisa o quadro inteiro, como antes. "off" nunca procura. panel.json guarda
    o resultado: {"modo": "auto" | "manual" | "off", "roi", "ocr_box"}."""
    import cv2

    out_dir = Path(out_dir)
    achado = None
    if "roi" in kwargs:
        info = {"modo": "manual", "roi": list(kwargs["roi"]), "ocr_box": list(kwargs["roi"])}
    elif panel == "auto":
        achado = find_panel(video)
        if achado:
            kwargs["roi"] = achado["roi"]
        info = {"modo": "auto", "roi": list(achado["roi"]) if achado else None,
                "ocr_box": list(achado["ocr_box"]) if achado else None}
    else:
        info = {"modo": "off", "roi": None, "ocr_box": None}
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
    (out_dir / "panel.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
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
    ap.add_argument("--panel", choices=["auto", "off"], default="auto",
                    help="auto: procura o painel do slide (moldura, miniatura e rodapé ficam de fora); "
                         "off: analisa o quadro inteiro. --roi tem prioridade sobre os dois")
    ap.add_argument("--trace", help="t0,t1 (s): imprime a decisão do rastreio a cada amostra estável nesse intervalo "
                                    "(diagnóstico: tinta, fundo liso, motivo da rejeição)")
    a = ap.parse_args()
    video = Path(a.video)
    out = Path(a.out) if a.out else Path(__file__).resolve().parent.parent / "data" / "derived" / video.stem / "frames"
    extra = {"roi": a.roi} if a.roi else {}
    if a.trace:
        t0, t1 = (float(x) for x in a.trace.split(","))

        def trace(d):
            if t0 <= d["t"] <= t1:
                mantido = "-" if d["mantido"] is None else f"{d['mantido']:.3f}"
                print(f"  t={d['t']:7.1f} tinta={d['tinta']:6.1%} liso={d['liso']:6.1%} mantido={mantido}"
                      f"{'  REJEITADO' if d['rejeitado'] else ''}")

        extra["trace"] = trace
    res = slides(video, out, a.step_s, panel=a.panel, keep_thr=a.keep_thr, min_slide_s=a.min_slide_s,
                 ignore=a.ignore, **extra)
    info = json.loads((out / "panel.json").read_text(encoding="utf-8"))
    print(f"painel: {info['modo']}" + (f"  roi {tuple(info['roi'])}  ocr {tuple(info['ocr_box'])}" if info["roi"] else
                                       "  (quadro inteiro)"))
    print(f"{len(res)} slides em {out}")
    for i, s in enumerate(res):
        print(f"{s['file']}  início {s['t']:7.1f} s  completo {s['t_complete']:7.1f} s  fim {s['t_end']:7.1f} s  "
              f"estágios {s['n_stages']}  visitas {len(s['visits'])}")


if __name__ == "__main__":
    main()
