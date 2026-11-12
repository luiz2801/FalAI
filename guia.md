# guia.md — Mapa do projeto Tema 9

**Versão 0.3 — 2026-11-12**

> Complementa o [CLAUDE.md](CLAUDE.md) (regras, decisões e roadmap). Descreve **o que existe hoje** e o contrato de cada módulo, para não precisar reler o código inteiro. O que está marcado com ⏳ é planejado.
> Os detalhes de cada etapa (com trechos de código) estão no [guia de implementação](../Tema09_MVP_Guia_de_Implementacao.md), fora do repositório.

> ## Regra de manutenção (obrigatória)
>
> **Antes de modificar:** leia a seção do módulo, os contratos (seção 5), os parâmetros (seção 6) e as armadilhas (seção 7).
>
> **Ao modificar:** toda alteração de função, contrato, parâmetro, formato de arquivo ou comando entra neste guia **no mesmo passo**. Se código e guia divergirem, o código é a verdade e o guia deve ser corrigido.
>
> **Versão:** texto sem efeito de comportamento → não muda. Mudança de comportamento, contrato ou comando → *minor* (0.2, 0.3…). Mudança que invalida resultados já gerados (ex.: nova normalização de texto, que altera o WER) → *major* (1.0), e os `results/raw.csv` antigos precisam ser regerados.
>
> Este guia é uma foto do estado atual. Histórico de decisões vai em [docs/decisoes.md](docs/decisoes.md).

**Mudanças da 0.3:** CLI `python src/extract.py <vídeo>` e seções 3.1 (como rodar) e 3.2 (como rodar no Docker); `extract.slides()` e `SlideTracker` (um quadro completo por slide, mesmo com bullets revelados aos poucos); `frames.json` ganha `t_complete`, `t_end`, `n_stages` e `visits`; `pipeline.py` ganha `--frames-mode` (padrão `slides`; `diff` é o baseline antigo). Detalhes em docs/decisoes.md (2026-11-12).

**Mudanças da 0.2:** código das etapas 2, 4 e 5 e da F9, vindo da junção com o plano paralelo (docs/decisoes.md, 2026-11-11). Contrato do `raw.csv` ampliado (seção 5.5); novos arquivos de gabarito (5.7 a 5.9).

---

## 1. Estado em uma olhada

| Etapa | Módulos | Estado |
|---|---|---|
| 0. Ambiente | `pyproject.toml`, `uv.lock`, `.python-version`, `Dockerfile`, `docker-compose.yml`, `.dockerignore`, `README.md`, `.gitignore`, `tests/test_smoke.py` | 🟡 build OK no contêiner; falta o NVIDIA Container Toolkit (teste de GPU) e o Tesseract `por` no host |
| 1. Dataset e gabarito | `data/raw/`, `data/gt/` | ⏳ depende de licença e de transcrição manual. Há exemplos de formato em `data/gt/*.example.*` |
| 2. STT (F1, parte de F3 e F4) | `extract.py` (áudio), `stt.py`, `degrade.py` (ruído), `metrics.py` | ✅ código e testes. Testado de ponta a ponta com um vídeo e gabarito temporários, apagados depois |
| 3. OCR (F2, parte de F3 e F4) | `extract.slides` ✅, `extract.keyframes` ✅ (baseline), `degrade.degrade_image` ✅, `ocr.py` ⏳, `metrics.word_pr` ✅ | 🟡 falta o `ocr.py` e o Tesseract `por` |
| 4. Experimentos (F4) | `run_experiments.py` (RQ1, I5 e F9), `report.py` | 🟡 RQ1 pronta; RQ2 entra com o `ocr.py` |
| 5. Interface (F5) | `pipeline.py`, `web/index.html` | ✅ player com legenda, slides, transcrição, busca e tópicos clicáveis. Backup da demo ⏳ |
| 6. Artigo | `artigo/` | 🟡 esqueleto; `[PENDENTE]` modelo SBC |
| Desejável | F6 (busca com métricas), F7 (ablação) | ⏳ a busca da página já faz substring sobre fala e slides; faltam as consultas e `precision_at_k`, `mrr` |
| Opcional | F8 (HLS) ⏳; F9 (resumo por LLM) ✅ `summarize.py`, `eval_summaries.py` | |

### Ambiente verificado

| Item | Máquina principal (guia de implementação, §1.1) | Notebook usado na junção (2026-11-11) |
|---|---|---|
| Python | 3.14.8 instalado. **Usar 3.12** via `uv` | 3.14.10; 3.12.13 via `uv` |
| FFmpeg | 9.0.2 | instalado |
| Tesseract | **não instalado** | instalado, **sem** `por` |
| GPU | RTX 3060, 12 GB | nenhuma (CPU, `int8`) |
| CPU e RAM | 24 threads, 62 GB | i7-1260P, 16 threads, 15 GB |
| Ollama | — | rodando, **sem modelos** baixados |

O RTF só é comparável dentro da mesma máquina: rode todos os experimentos na principal e reporte o hardware (`results/hardware.json`, gerado pelo `run_experiments.py`).

---

## 2. Fluxo de dados

```
data/raw/aulaNN.mp4
  │
  ├─ extract.clip_audio (data/gt/clips.csv) ──► data/derived/aulaNN/aulaNN.clip.limpo.wav (16 kHz, mono)
  │        │
  │        degrade.add_noise (SNR 20, 10, 0 dB; seed 0) ──► aulaNN.clip.snr20.wav …
  │        │
  │        stt.transcribe (modelo × condição × 3 repetições; carga e aquecimento fora)
  │        ├─► data/derived/aulaNN/stt/aulaNN.<modelo>.<condição>.{vtt,srt,txt,json}  (análise de erros)
  │        └─► metrics.wer, cer, term_accuracy ──► results/raw.csv
  │
  ├─ extract.slides (step_s, keep_thr, ...) ──► data/derived/aulaNN/frames/NNNN.png (quadro completo) + frames.json
  │        (extract.keyframes = baseline por diferença de quadros)
  │        ├─ degrade.degrade_image (resize, blur, jpeg)
  │        └─ ocr.* ⏳ ──► slides.json (t, file, texto) + tempo ──► metrics.cer, word_pr ──► raw.csv
  │
  └─ F9: stt.transcribe (vídeo inteiro, cada modelo) ──► summarize.summarize (ollama | claude)
           ──► results/summaries/<v>__<modelo>__<llm>__r<k>.md ──► eval_summaries (às cegas) ──► raw.csv

results/raw.csv ──► report.py ──► results/tables/*.{csv,md}, results/figures/*.{png,pdf}

pipeline.py: aulaNN.mp4 ──► transcrição + keyframes (+ resumo) ──► web/data/aulaNN/ ──► web/index.html
```

**Dependências entre módulos** (a direção é sempre de cima para baixo):

- `metrics.py` não importa nada do projeto. Só funções puras.
- `degrade.py` e `extract.py` dependem só de bibliotecas externas (numpy, OpenCV, FFmpeg).
- `stt.py` usa `extract.load_audio`. `summarize.py` usa só `stt.fmt` (e o SDK `anthropic`, opcional).
- `run_experiments.py` orquestra a matriz. `eval_summaries.py` usa `run_experiments.row` e `append_rows`.
- `report.py` só lê `results/raw.csv`. Não recalcula métrica.
- `web/` só lê arquivos gerados. Não importa código Python.

---

## 3. Comandos

| Comando | Faz | Observação |
|---|---|---|
| `uv sync` | Instala as dependências fixadas | `uv sync --extra llm` inclui o SDK da Anthropic (F9 com Claude) |
| `uv run pytest -q` | Testes (sem rede, sem carregar modelos) | O teste do Tesseract falha até instalar `tesseract-data-por` |
| `HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose run --rm pipeline` | Os mesmos testes no contêiner | Ver README |
| `uv lock` | Atualiza o `uv.lock` depois de mudar uma versão no `pyproject.toml` | Não usar `pip install` avulso |
| `uv run python src/stt.py <mídia> --model small` | Transcreve uma mídia (`.vtt`, `.srt`, `.txt`, `.json`) | `--vocab data/gt/terms.txt` para o I5 |
| `uv run python src/pipeline.py data/raw/aulaNN.mp4 --model small` | Vídeo de ponta a ponta e pacote da demo em `web/data/` | `--summary ollama` ou `--summary claude` para a F9; sem LLM, o resumo é pulado |
| `uv run python src/extract.py <vídeo>` | Só os slides completos, sem STT (ver 3.1; no Docker, 3.2) | Grava em `data/derived/<vídeo>/frames/` |
| `cd web && python -m http.server` | Serve a demo | Porta 8000. `file://` não funciona (o `fetch` é bloqueado) |
| `uv run python src/run_experiments.py` | RQ1: `tiny`, `base`, `small` × limpo/20/10/0 dB × 3 repetições → `results/raw.csv` | `--models`, `--reps`, `--vocab` (I5), `--no-clean`, `--redo`. Pula o que já está no CSV |
| `uv run python src/run_experiments.py --summaries --backend ollama` | F9: transcreve os vídeos inteiros e gera 2 resumos por modelo | `--backend claude --confirm-paid` para a API paga |
| `uv run python src/eval_summaries.py prepare` / `consolidate` | F9: pacote às cegas para os avaliadores; depois grava as notas no `raw.csv` | Ver seção 4.9 |
| `uv run python src/report.py` | Tabelas e figuras a partir do CSV | Sem argumentos |
| `ollama pull qwen2.5:7b` | Baixa o LLM local padrão da F9 (~4,7 GB) | O usuário executa |
| `tesseract --list-langs \| grep por` | Confirma o idioma português do Tesseract | Instalação de sistema: o usuário executa |

---

### 3.1 Como rodar a extração de slides

**1. Só os slides** (rápido: 2 a 4 s por vídeo de 6 a 14 min, sem GPU):

```bash
uv run python src/extract.py "data/raw/aula01.mp4"
```

Saída em `data/derived/aula01/frames/` (o nome da pasta é o do vídeo, sem a extensão):

```
data/derived/aula01/frames/
├── 0000.png        # quadro COMPLETO de cada slide, na resolução original
├── 0001.png
└── frames.json     # tempos de cada slide (contrato 5.3)
```

No terminal sai uma linha por slide:

```
0001.png  início    10.5 s  completo    37.5 s  fim    42.0 s  estágios 5  visitas 1
```

Leitura: o slide entrou aos 10,5 s, ficou completo aos 37,5 s (o professor revelou 4 bullets, por isso 5 estágios) e saiu aos 42,0 s. `visitas` > 1 quer dizer que o professor voltou ao slide.

**2. Opções** (todas opcionais):

| Opção | Padrão | Use quando |
|---|---|---|
| `--out PASTA` | `data/derived/<vídeo>/frames` | Quer gravar em outro lugar (por exemplo, para comparar duas configurações) |
| `--step-s S` | 0,5 | Os bullets aparecem muito rápido (diminua) ou quer mais velocidade (aumente) |
| `--keep-thr X` | 0,9 | Slides diferentes estão sendo fundidos (suba) ou o mesmo slide sai duplicado (baixe) |
| `--min-slide-s S` | 2,0 | Slides de poucos segundos estão sendo perdidos (baixe) ou relances viram slide (suba) |
| `--ignore x0,y0,x1,y1` | nenhuma | Há algo fixo que mexe, como a webcam do professor num canto ou um relógio. Pode repetir |
| `--roi x0,y0,x1,y1` | quadro inteiro | Quer analisar só a área do slide (por exemplo, cortar a barra do player) |

As regiões são **frações do quadro** (0 a 1), com origem no canto superior esquerdo. Exemplo: webcam no canto inferior direito, ocupando os 25% da direita e os 30% de baixo:

```bash
uv run python src/extract.py data/raw/aula01.mp4 --ignore 0.75,0.7,1,1
```

O professor em movimento num canto já costuma ser ignorado sozinho (seção 4.1); `--ignore` é para quando isso não bastar. A região ignorada só sai da *comparação*: o PNG gravado continua inteiro.

**3. Vídeo de ponta a ponta** (transcrição, slides e página da demo):

```bash
uv run python src/pipeline.py data/raw/aula01.mp4 --model small
cd web && python -m http.server   # abrir http://localhost:8000
```

O pipeline usa `extract.slides` por padrão. Para comparar com o baseline antigo (primeiro quadro após cada mudança), use `--frames-mode diff`.

**4. Pelo Python** (por exemplo, num notebook ou num experimento):

```python
import sys; sys.path.insert(0, "src")
import extract

slides = extract.slides("data/raw/aula01.mp4", "data/derived/aula01/frames", step_s=0.5,
                        ignore=[(0.75, 0.7, 1, 1)])
for s in slides:
    print(s["file"], s["t"], s["t_complete"], s["n_stages"])
```

Para quadros que já estão em memória (testes, outro leitor de vídeo), use `extract.SlideTracker`: `feed(t, quadro_bgr)` a cada amostra e `finish(t_fim)` no final.

**5. Conferir o resultado.** Os números da CLI não dizem se o resultado está certo; olhe as imagens. Uma folha de contato rápida:

```bash
uv run python - <<'PY'
import cv2, glob, numpy as np
fs = sorted(glob.glob("data/derived/aula01/frames/0*.png"))
ims = [cv2.resize(cv2.imread(f), (480, 270)) for f in fs]
while len(ims) % 4: ims.append(np.zeros_like(ims[0]))
cv2.imwrite("contato.png", np.vstack([np.hstack(ims[i:i + 4]) for i in range(0, len(ims), 4)]))
PY
```

O que procurar e o que fazer:

| Sintoma | Causa provável | Ajuste |
|---|---|---|
| O slide saiu pela metade (faltam bullets) | O professor trocou de slide antes de terminar, ou o último estado durou menos de 1 s | Baixe `--step-s` para 0,25 |
| O mesmo slide saiu em 2 arquivos | Algo fixo no quadro (relógio, ícone) mudou entre as visitas | `--ignore` na região, ou baixe `--keep-thr` |
| Dois slides viraram um | Mesmo título e conteúdo parecido | Suba `--keep-thr` para 0,95 |
| Apareceu o professor como slide | Fundo pouco liso e tinta baixa (ver `min_flat`, seção 4.1) | Ajuste `min_flat` no código e registre em `docs/decisoes.md` |
| Cartões de logo (abertura e encerramento) aparecem | Parecem slides | Apague à mão por ora; ainda não há filtro |
| Nenhum slide, ou menos do que o esperado | Slide com fundo de imagem ou degradê | Veja a armadilha 16 (limites conhecidos) |

Cada mudança de limiar que alterar o resultado deve ir para `docs/decisoes.md` (CLAUDE.md, regra 7). As sugestões acima de `--step-s 0,25` e `--keep-thr 0,95` são orientação, não foram testadas em vídeo real.

**6. Regerar.** Rodar de novo **sobrescreve** `NNNN.png` e o `frames.json`, mas **não apaga** PNGs antigos. Se o número de slides diminuir, sobram arquivos que o `frames.json` não lista. Apague a pasta `frames/` antes de regerar, ou use `--out` com uma pasta nova.

### 3.2 Como rodar no Docker

Útil para quem não tem Python 3.12, `uv` ou FFmpeg na máquina. A imagem traz tudo (CUDA 12.4, FFmpeg, Tesseract com `por`, as dependências fixadas). **A extração de slides não usa GPU**, então não precisa do NVIDIA Container Toolkit.

> **Estado:** os comandos abaixo seguem o `Dockerfile` e o `docker-compose.yml`, mas **não foram executados**. A imagem `falai` ainda não foi construída nesta máquina. Ao testar pela primeira vez, corrija esta seção.

**1. Construir a imagem** (uma vez; demora e baixa vários GB, por causa da base CUDA e das bibliotecas de ML):

```bash
docker build -t falai:latest .
```

O código de `src/` e `tests/` é **montado como volume**, não copiado a cada uso: depois de mudar um `.py` não precisa reconstruir a imagem. Só reconstrua se mudar `pyproject.toml` ou `uv.lock`.

**2. Colocar o vídeo onde o contêiner enxerga.** O contêiner vê apenas `./data` e `./results` (e `./src`, `./tests`). Duas formas:

- **Dentro de `data/raw/`** (a mais simples): copie o vídeo para `data/raw/aula01.mp4`. A pasta está no `.gitignore` e no `.dockerignore`. *Link simbólico não funciona*: o destino fica fora do contêiner.
- **Montar a pasta onde o vídeo já está**, com `-v` (exemplos abaixo).

**3. Rodar só os slides, sem GPU** (recomendado):

```bash
# vídeo em data/raw/
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$PWD/src:/app/src:ro" -v "$PWD/data:/app/data" \
  falai:latest python src/extract.py data/raw/aula01.mp4

# vídeo em outra pasta do host, montada em /videos (somente leitura)
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$PWD/src:/app/src:ro" -v "$PWD/data:/app/data" \
  -v "$PWD/Financial Acumen for Non-Financial Managers:/videos:ro" \
  falai:latest python src/extract.py "/videos/Renda receita e despesas Coursera.mkv"
```

Os PNGs e o `frames.json` aparecem no host, em `data/derived/<nome do vídeo>/frames/`. As opções da seção 3.1 (`--ignore`, `--step-s`...) vão depois do nome do vídeo.

O `--user` faz os arquivos gerados pertencerem ao seu usuário, e não ao `root`. Sem ele, você teria de apagá-los com `sudo`.

**4. Com `docker compose`** (o serviço `pipeline` já monta `src`, `tests`, `data` e `results`):

```bash
export HOST_UID=$(id -u) HOST_GID=$(id -g)
docker compose run --rm pipeline python src/extract.py data/raw/aula01.mp4
docker compose run --rm -v "$PWD/Financial Acumen for Non-Financial Managers:/videos:ro" \
  pipeline python src/extract.py "/videos/Renda receita e despesas Coursera.mkv"
docker compose run --rm pipeline bash        # shell dentro do contêiner
```

O serviço reserva a GPU (`deploy.resources.reservations`). **Sem o NVIDIA Container Toolkit, o `compose run` falha** ao iniciar (erro de driver `nvidia`). Nesse caso, use o `docker run` do passo 3, ou instale o toolkit (README, "Em contêiner").

**5. Pipeline completo no contêiner** (STT + slides + página). Aqui a GPU ajuda, e o `docker run` precisa de `--gpus all`:

```bash
docker run --rm --gpus all --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$PWD/src:/app/src:ro" -v "$PWD/data:/app/data" -v "$PWD/web:/app/web" \
  falai:latest python src/pipeline.py data/raw/aula01.mp4 --model small
cd web && python -m http.server     # no host
```

O `web/` também precisa ser montado, porque o pipeline grava a demo em `web/data/`. Os modelos do Whisper são baixados a cada execução, já que o `HOME=/tmp` do contêiner não persiste; para guardá-los, monte um volume de cache (por exemplo, `-v "$HOME/.cache/huggingface:/tmp/.cache/huggingface"`).

**6. Testes no contêiner:** `docker compose run --rm pipeline` (roda `pytest -q`). Aqui o teste do Tesseract passa, porque a imagem traz o idioma `por`.

**Problemas comuns**

| Sintoma | Causa | Solução |
|---|---|---|
| `could not select device driver "nvidia"` | Sem NVIDIA Container Toolkit e usando `compose run` ou `--gpus` | Use `docker run` sem `--gpus` (slides não precisam), ou instale o toolkit |
| `permission denied` ao gravar em `data/` | Faltou `--user "$(id -u):$(id -g)"`, ou uma execução anterior criou a pasta como `root` | Passe `--user`; corrija o dono da pasta (`sudo chown -R $USER data`) |
| `OpenCV não abriu <vídeo>` | O caminho está errado *dentro* do contêiner | Use o caminho do contêiner (`/videos/...`, `data/raw/...`), não o do host |
| Mudei o código e nada mudou | `src/` não foi montado | Inclua `-v "$PWD/src:/app/src:ro"` |

## 4. Módulos

### 4.1 `src/extract.py` (etapas 2 e 3)

| Função | Faz |
|---|---|
| `extract_audio(video, out_wav)` | WAV PCM 16 bits, mono, 16 kHz |
| `clip_audio(media, out_wav, start_s, dur_s)` | Recorta o trecho com gabarito (corta o áudio antes de transcrever) |
| `load_audio(media) -> np.ndarray` | float32, 16 kHz, mono, em memória. É o que o `stt.transcribe` usa (armadilha 7.2.10) |
| `media_duration(media)` | Duração pelo `ffprobe` |
| `slides(video, out_dir, step_s=0.5, **kwargs) -> list[dict]` | Um quadro completo por slide (ver abaixo). Usado pelo `pipeline.py` |
| `SlideTracker(out_dir=None, ...)` | Núcleo de `slides`: `feed(t, quadro)` e `finish(t_fim)`. Aceita quadros em memória (testes) |
| `keyframes(video, out_dir, step_s=1.0, thr=0.04) -> list[dict]` | Baseline: primeiro quadro após cada mudança (ver abaixo) |

**`slides`**: resolve o caso do professor que revela os bullets aos poucos (o `keyframes` guarda o primeiro quadro após a mudança, ou seja, o slide incompleto). Leitura sequencial, um quadro a cada `step_s`. Cada quadro vira uma máscara de "tinta" (pixels diferentes do fundo, em 480×270 cinza). Comparado ao quadro candidato do episódio: `mantido` = fração da tinta do candidato que continua (tolerância de 1 px); `acrescido` = fração da tinta atual que é nova.

- `mantido ≥ keep_thr` (0,9) e `acrescido > grow_eps` (0,01): o slide cresceu; o quadro atual vira o candidato (`n_stages` + 1).
- `mantido < keep_thr`, ou a faixa do cabeçalho (22% superiores) mudou (< 0,8 mantido): slide novo. O candidato do episódio anterior é gravado, que é o último quadro estável antes da troca ("buffer reverso").
- Só quadros **estáveis** são considerados (iguais aos `min_stable` = 2 amostras anteriores): fade, mão e cursor passando não entram.
- Episódios com menos de `min_slide_s` (2 s) são descartados. Tela sem conteúdo (menos de `min_ink` = 0,2% de tinta) ou com tinta demais (mais de `max_ink` = 30%: imagem natural, como o professor em tela cheia) fecha o episódio e não abre outro. Nos slides de teste a tinta ficou entre 3% e 14%; no professor, 42% e 59%.
- Também fecha o episódio o quadro com pouco **fundo liso** (`min_flat` = 0,35: fração de pixels a ±3 níveis da mediana). Só a tinta não bastava: o professor de camisa azul em fundo escuro tinha 29%. Medido em 62 quadros: professor 7% a 18%, slides e cartões ≥ 48%.
- O cabeçalho só conta como "o mesmo" se ≥ 95% dele coincidir (`TITLE_KEPT`). Com 0,8, "Revenue" e "Expenses" passavam por acaso (0,83 a 0,90).
- **Professor no canto:** células da grade 16×9 cuja luminância muda quase a cada amostra (EMA > 0,4) entram numa máscara de instabilidade e saem da comparação (o quadro gravado continua inteiro). `ignore=[(x0, y0, x1, y1), ...]` (frações do quadro) fixa regiões à mão; `roi` limita a região analisada (por exemplo, para cortar a barra do player).
- **Slide revisitado:** se o candidato for igual (nos dois sentidos), ou parcial/completo do mesmo título, de um slide já gravado, não gera arquivo novo; a visita entra em `visits`. Se a volta trouxer o slide **mais completo**, o quadro novo substitui o arquivo anterior e `t_complete` passa a ser o da volta.

Saída: `NNNN.png` (quadro original) e `frames.json` (contrato 5.3). Cria `out_dir` e confere o `cv2.imwrite`.

**`keyframes`** (baseline): amostra um quadro a cada `step_s` segundos com leitura sequencial (`cap.grab()`, sem seek). Reduz para 320×180 em cinza e guarda o quadro quando a diferença média para o último keyframe guardado (0–1) passa de `thr`. Grava `NNNN.png` (quadro original) e `frames.json` (`[{"t", "file"}]`) e devolve a lista. Cria `out_dir` se não existir e confere o retorno do `cv2.imwrite`.

### 4.2 `src/stt.py` (etapa 2)

| Função | Faz |
|---|---|
| `fmt(t)`, `fmt_srt(t)` | Segundos → `HH:MM:SS.mmm` (WebVTT) / `HH:MM:SS,mmm` (SRT) |
| `resolve_device(device="auto")` | `(device, compute_type)`: CUDA/`float16` se houver GPU, senão CPU/`int8` |
| `load_model(size, device, cpu_threads=0)` | `WhisperModel` com cache; devolve `(modelo, segundos_de_carga)` (0 se já em cache) |
| `unload_models()` | Esvazia o cache entre modelos |
| `transcribe(media, size, device="auto", initial_prompt=None, clean=True, word_timestamps=True) -> (segs, duracao_s, tempo_s)` | Ver abaixo |
| `clean_segments(segs) -> (mantidos, removidos)` | Remove alucinações conhecidas e repetições (cada removido leva `reason`) |
| `build_prompt(terms)`, `read_terms(path)` | I5: `initial_prompt` com o vocabulário, até ~600 caracteres |
| `subtitle_blocks(segs, max_chars=42, max_lines=2, max_dur=7.0)` | Blocos de legenda equilibrados; `max_chars=None` → um bloco por segmento |
| `write_vtt(segs, path, max_chars=42)`, `write_srt(...)` | Legendas (contrato 5.4) |
| `plain_text(segs)` | Transcrição completa em parágrafos, sem cabeçalho nem tempos |
| `save_outputs(segs, out_dir, stem, meta)` | `<stem>.vtt`, `.srt`, `.segments.srt` (para o LLM), `.txt`, `.json` |

**`transcribe`**: decodifica a mídia com `extract.load_audio` e transcreve com `language="pt"`, `beam_size=5`, `vad_filter=True`, `condition_on_previous_text=False`. Consome o gerador **dentro** do cronômetro; carga do modelo e decodificação ficam fora. `segs` é uma lista de `{"id", "start", "end", "text", "avg_logprob", "no_speech_prob", "words"}` (`words`: `{"start", "end", "word", "probability"}` ou `None`). Com `clean=True`, os segmentos removidos ficam em `transcribe.last_removed`. `duracao_s` vem de `info.duration`.

### 4.3 `src/ocr.py` ⏳ (etapa 3)

**`ocr_tesseract(path) -> (texto, tempo_s)`**: `pytesseract.image_to_string(cv2.imread(path), lang="por", config="--psm 6")`. Testar `--psm 3` e usar o melhor para os dois motores, ou relatar a diferença.

**`ocr_easyocr(path) -> (texto, tempo_s)`**: `" ".join(_reader.readtext(path, detail=0))`, com o `Reader` criado sob demanda (armadilha 7.2.1).

Ambas seguem a assinatura `(path) -> (texto, tempo_s)`. A saída preenche `texto` no `slides.json` (contrato 5.3); o `pipeline.py` já copia esse campo para a página.

### 4.4 `src/degrade.py` (etapas 2 e 3)

| Função | Faz |
|---|---|
| `read_wav(path) -> (x, taxa)`, `write_wav(path, x, taxa)` | WAV PCM 16 bits mono ↔ float32 em [-1, 1] (biblioteca `wave`, sem dependência nova) |
| `add_noise(in_wav, out_wav, snr, seed=0) -> snr_medido` | Ruído branco gaussiano com potência = potência do sinal / 10^(SNR/10), recorte em [-1, 1]. Devolve o SNR efetivo medido após recorte e quantização |
| `add_noise_array(x, snr, seed=0)`, `snr_db(limpo, ruidoso)` | Versões em memória |
| `degrade_image(img, kind, level)` | `"resize"` (fator), `"blur"` (kernel ímpar), `"jpeg"` (qualidade). `kind` desconhecido ou kernel par → `ValueError` |

### 4.5 `src/metrics.py` (etapa 2, ampliado na 3)

Funções puras. Nenhuma lê arquivo.

| Função | Faz |
|---|---|
| `norm(t) -> str` | Minúsculas, pontuação trocada por espaço (`[^\w\s]`), acentos mantidos, espaços colapsados. **É a única normalização do projeto.** |
| `wer(ref, hyp)`, `cer(ref, hyp)` | `jiwer` sobre os textos normalizados. Referência vazia → `ValueError` |
| `wer_detail(ref, hyp)` | WER com substituições, remoções, inserções e tamanho da referência (análise de erros, regra 11) |
| `word_pr(ref, hyp) -> (precisao, recall)` | Multiconjunto de palavras; ignora a ordem |
| `rtf(tempo_proc_s, duracao_s)` | Tempo de processamento / duração da mídia |
| `term_accuracy(ref, hyp, terms)` | I5: ocorrências de cada termo na hipótese, limitadas às da referência, sobre o total na referência; e falsos positivos |
| `precision_at_k`, `recall_at_k`, `mrr` ⏳ (F6) | Métricas de busca |

### 4.6 `src/run_experiments.py` (etapa 4)

**RQ1** (padrão): para cada vídeo com linha em `data/gt/clips.csv` **e** gabarito `data/gt/<video>.stt.txt`:

- gera o trecho limpo e as versões com ruído em `data/derived/<video>/` (reaproveitadas; seed 0; avisa se o SNR medido fugir ±0,5 dB do alvo);
- por modelo: mede a carga (linha `carga_modelo`), faz uma execução de aquecimento descartada e roda cada condição `--reps` vezes (padrão 3);
- grava por repetição as linhas `wer`, `cer` e, havendo `data/gt/terms.txt`, `acerto_termos`;
- guarda a hipótese da repetição 1 em `data/derived/<video>/stt/` para a análise de erros;
- `--vocab` acrescenta a condição `limpo_vocab` (I5); `--no-clean` desliga o filtro de alucinações e rotula o modelo como `<modelo>-semfiltro`;
- pula o que já está no CSV (`--redo` refaz).

**F9** (`--summaries`): transcreve cada vídeo inteiro de `data/raw/` com cada modelo (reaproveita em `data/derived/<video>/summary/`), gera `--reps` resumos (padrão 2) e grava `tempo_llm` e `custo_usd`. O backend `claude` exige `--confirm-paid` (regra 12).

Também grava `results/hardware.json`.

### 4.7 `src/report.py` (etapa 4)

Só lê `results/raw.csv` (com duplicatas, vale a última linha) e produz o que tiver dados:

| Saída | Conteúdo |
|---|---|
| `tables/tabela1_wer` | WER médio ± desvio (n), por modelo × condição |
| `tables/tabela3_tempo_stt` | Segundos de inferência por minuto de mídia e RTF, por modelo |
| `tables/tabela4_resumos` | F9: nota, fidelidade, cobertura, precisão dos timestamps, custo e tempo, por modelo × LLM |
| `figures/fig1_wer_snr` | WER × SNR, uma linha por modelo (robustez) |
| `figures/fig2_rtf_wer` | RTF × WER no áudio limpo, um ponto por modelo (trade-off) |
| `figures/fig3_termos_vocabulario` | I5: acerto de termos com e sem `initial_prompt` |
| `figures/fig4_wer_nota_resumo` | F9: WER do modelo × nota do resumo (pergunta secundária) |
| Tabela 2 (OCR) ⏳ | CER, precisão e recall por motor × degradação |

Tabelas em `.csv` e `.md`; figuras em `.png` (300 dpi) e `.pdf`. Nenhum número digitado à mão.

### 4.8 `web/index.html` e `src/pipeline.py` (etapa 5)

`pipeline.process(video, model="small", device="auto", vocab=None, summary=None, ...)`: transcrição, keyframes e, com `summary`, o resumo e os capítulos. Se o backend do LLM não estiver disponível, o resumo é pulado com o motivo. Depois copia para `web/data/<video>/`: `video.<ext>`, `subs.vtt`, `slides.json`, `transcript.json`, `frames/`, `manifest.json` e, havendo resumo, `chapters.vtt` e `summary.md`. Também atualiza `web/data/videos.json`.

A página lê esses arquivos e tem player com legenda (`<track>`), seletor de vídeo e abas **Tópicos** (capítulos do resumo), **Slides** (miniaturas), **Transcrição** (com o trecho atual destacado) e **Resumo**. A busca é substring sem diferença de caixa nem de acento, sobre tópicos, texto dos slides e transcrição; clicar em qualquer item faz `video.currentTime = t`. Funciona em tema claro e escuro.

### 4.9 `src/summarize.py` e `src/eval_summaries.py` (F9)

**`summarize(srt, backend="ollama", model=None, effort="medium", slides=None) -> (markdown, meta)`**

- Entrada: o `.segments.srt` (um bloco por segmento, com tempos).
- O prompt pede seis seções, com os tópicos no formato `- [hh:mm:ss] Título` e "use apenas informações presentes na transcrição".
- `ollama`: `POST /api/chat` em `OLLAMA_HOST` (padrão `localhost:11434`), `num_ctx=32768`, `temperature=0.2`, `seed=0`, modelo padrão `qwen2.5:7b`.
- `claude`: SDK oficial, streaming, `effort` (padrão `medium`), modelo padrão `claude-opus-5-5`, com `fallbacks: "default"` (beta `server-side-fallback-2026-07-01`) nos modelos que aceitam. Recusa vira erro explícito. Slides como imagens só neste backend.
- `meta` traz `backend`, `model`, `model_served`, `tokens_in`, `tokens_out`, `cost_usd` (0 no Ollama), `time_s` e `stop_reason`.

Também: `backend_status(backend, model)` (motivo da indisponibilidade ou `None`), `extract_topics(md) -> [(s, título)]`, `write_chapters_vtt(topics, path, duration_s)` e `export_pdf(md)` (pandoc + xelatex).

**`eval_summaries.py`**

- `prepare`: copia os resumos para `results/eval/blind/R001.md…` em ordem embaralhada (seed 42) e cria `key.csv` (não enviar), `scores.csv`, `fidelity.csv` e `topics.csv`.
- `topics.csv` traz a cobertura e a precisão dos timestamps (±30 s) **estimadas**, casando títulos por Jaccard das palavras de conteúdo (limiar 0,4) contra `data/gt/<video>.topics.txt`. Corrija nas colunas `*_manual`.
- `consolidate`: grava no `raw.csv` as linhas `tarefa=resumo` com `nota_media`, `fidelidade`, `cobertura` e `precisao_timestamps`.

---

## 5. Contratos de dados

**5.1 Gabarito de fala** `data/gt/<video>.stt.txt`: texto puro, transcrito à mão do zero, nas regras de normalização do README.

**5.2 Gabarito de slides** `data/gt/<video>.slides.json`: lista `[{"t": 125.0, "texto": "..."}]`, `t` em segundos. Use `texto`, a mesma chave da saída (docs/decisoes.md).

**5.3 Slides derivados** `frames.json` de `extract.slides`: `[{"t", "t_complete", "t_end", "n_stages", "visits", "file"}]` (`t` = início do slide, é para onde o clique salta; `t_complete` = quando ficou completo; `visits` = `[[início, fim], ...]`). O baseline `keyframes` grava só `{"t", "file"}`. `slides.json` da página: `[{"t", "file", "texto"}]` mais `t_complete`, `t_end` e `n_stages` quando existem; `texto` vazio até o OCR.

**5.4 Legenda** `.vtt`: `WEBVTT`, linha em branco, blocos `HH:MM:SS.mmm --> HH:MM:SS.mmm` com o texto nas linhas seguintes (no máximo 2 linhas de 42 caracteres). O `.srt` segue o mesmo corte, numerado. O `.segments.srt` tem um bloco por segmento e é a entrada do LLM.

**5.5 Resultados** `results/raw.csv`, um registro por execução × métrica:

```
video,tarefa,modelo,condicao,repeticao,metrica_erro,valor_erro,tempo_s,duracao_s,rtf
```

| `tarefa` | `condicao` | `metrica_erro` (com `valor_erro`) | Tempo |
|---|---|---|---|
| `stt` | `limpo`, `snr20`, `snr10`, `snr0`, `limpo_vocab` | `wer`, `cer`, `acerto_termos` | `tempo_s` (inferência), `duracao_s`, `rtf` em toda linha |
| `stt` | `carga` (vídeo `_`) | `carga_modelo` (sem valor) | `tempo_s` = carga do modelo |
| `resumo` | `<backend>:<modelo do LLM>` | `tempo_llm` (sem valor), `custo_usd` | `tempo_s` = tempo do LLM |
| `resumo` | idem | `nota_media`, `fidelidade`, `cobertura`, `precisao_timestamps` (de `eval_summaries.py`) | — |
| `ocr` ⏳ | `limpo`, `resize0.5`, `blur9`, `jpeg30` | `cer`, `precisao_palavras`, `recall_palavras` | |

Na tarefa `stt`, `modelo` pode ter o sufixo `-semfiltro` (`--no-clean`). Na tarefa `resumo`, `modelo` é o modelo de STT que gerou a transcrição. Separador vírgula; lido com `pandas.read_csv` sem opções.

**5.6 Nomes**: aulas são `aula01`, `aula02`… Modelos de STT são `tiny`, `base`, `small`, `medium`, `large-v3`.

**5.7 Trechos** `data/gt/clips.csv`: `video,start_s,dur_s` (ver `clips.example.csv`).

**5.8 Tópicos reais** `data/gt/<video>.topics.txt` (F9): uma linha `[hh:mm:ss] Título` por tópico (ver `aula01.topics.example.txt`).

**5.9 Vocabulário** `data/gt/terms.txt` (I5, opcional): um termo por linha; `#` comenta (ver `terms.example.txt`).

---

## 6. Parâmetros fixos

Todos valem igual para todos os modelos e motores de uma mesma comparação.

| Parâmetro | Valor | Onde | Observação |
|---|---|---|---|
| Idioma do STT | `pt` | `stt.LANGUAGE` | |
| `beam_size` | 5 | `stt.BEAM_SIZE` | |
| `vad_filter` | `True` | `stt.VAD_FILTER` | |
| `condition_on_previous_text` | `False` | `stt.transcribe` | `[PREMISSA]` reduz loops |
| `word_timestamps` | `True` | `stt.transcribe` | Necessário para a legenda; entra no tempo |
| Filtro de alucinações | ligado | `stt.clean_segments` | `[PREMISSA]` muda o WER; `--no-clean` para comparar |
| Dispositivo STT | `auto`: `cuda`/`float16`, senão `cpu`/`int8` | `stt.resolve_device` | Usar a mesma escolha em todos os experimentos |
| Modelos STT | `tiny`, `base`, `small` (`medium` se couber) | RQ1 | |
| `initial_prompt` | até ~600 caracteres | `stt.build_prompt` | Só na condição `limpo_vocab` |
| Legenda | 42 caracteres × 2 linhas, até 7 s | `stt.subtitle_blocks` | |
| Tesseract | `lang="por"`, `--psm 6` | `ocr.ocr_tesseract` ⏳ | Testar `--psm 3` |
| EasyOCR | `["pt"]`, `gpu=True` | `ocr.py` ⏳ | Ver armadilha 7.2.1 |
| `step_s` (slides) | 0,5 s | `extract.slides` | Ablação F7 |
| `keep_thr`, `grow_eps`, `min_stable`, `min_slide_s`, `max_ink`, `min_flat` (slides) | 0,9; 0,01; 2; 2,0 s; 0,30; 0,35 | `extract.SlideTracker` | `[PREMISSA]` sem calibração em vídeo real ainda. Ablação F7 |
| `step_s` (keyframes, baseline) | 1,0 s | `extract.keyframes` | |
| `thr` (keyframes, baseline) | 0,04 | `extract.keyframes` | Calibrar entre 0,03 e 0,05 olhando as imagens. Meta: ~1 keyframe por slide |
| Resolução de análise | 320×180 cinza (keyframes); 480×270 (slides) | `extract.py` | Só para comparar quadros |
| SNR | limpo, 20, 10, 0 dB | RQ1 | |
| Seed do ruído | 0 | `degrade.add_noise` | |
| Degradações de imagem | resize 0,5; blur 9; JPEG 30 | RQ2 | |
| Repetições | 3 (STT), 2 (resumo) | `run_experiments.py` | Média ± desvio |
| LLM local | `qwen2.5:7b`, `num_ctx` 32768, `temperature` 0,2, `seed` 0 | `summarize.py` | F9 |
| LLM pago | `claude-opus-5-5`, `effort` `medium` | `summarize.py` | Só com confirmação |
| Tolerância dos timestamps | ±30 s | `eval_summaries.py` | F9 |
| Trecho com gabarito | 2 a 3 min por vídeo, começando no início de uma frase | dataset | Armadilha 21 |
| Vídeos da matriz | 6, de 5 a 15 min | dataset | Vídeos longos (~45 min) ficam fora da matriz |

---

## 7. Armadilhas

### 7.1 Medição (afetam a nota de métricas)

1. **Gerador preguiçoso.** `transcribe()` não transcreve até ser consumido. O `stt.transcribe` consome dentro do cronômetro.
2. **Aquecimento.** A primeira execução inclui a inicialização (GPU, caches). O `run_experiments.py` faz uma execução de aquecimento descartada por modelo.
3. **Carregamento do modelo.** Medido à parte (linha `carga_modelo`) e feito uma vez por modelo.
4. **Parâmetros iguais.** `language`, `beam_size`, `vad_filter`, `condition_on_previous_text` e `word_timestamps` iguais em todos os modelos.
5. **Hardware.** Reportar no artigo. RTF só compara dentro da mesma máquina.
6. **Escopo do tempo de OCR.** Definir se a leitura da imagem entra no tempo e manter igual para os dois motores.

### 7.2 Código e ambiente

1. **`easyocr.Reader` no import** carrega o modelo e exige GPU só de importar `ocr.py`. Criar sob demanda, numa função com cache.
2. **`device="cuda"` fixo** falha sem CUDA. Por isso o padrão é `auto`.
3. **Bibliotecas CUDA ausentes** (cuBLAS, cuDNN): instalar as bibliotecas ou usar CPU. Usar a mesma opção em todos os experimentos.
4. **`cap.set(CAP_PROP_POS_FRAMES)` a cada passo** faz seek em cada amostra. O `keyframes` usa leitura sequencial.
5. **`cv2.imwrite` não cria diretório.** O `keyframes` cria `out_dir` e confere o retorno.
6. **`degrade_image` com `kind` desconhecido** levanta `ValueError`.
7. **`cv2.imread` com caminho inexistente** devolve `None`, e o pytesseract falha com mensagem pouco clara. Conferir antes.
8. **`lang="por"` sem o pacote.** Falha se faltar `tesseract-data-por`. Instalação de sistema: pedir ao usuário.
9. **Python 3.14.** Pacotes de ML podem não ter binários para essa versão. Usar 3.12.
10. **faster-whisper 1.2.1 × PyAV 19.** Passar o caminho do arquivo faz o faster-whisper abrir com o PyAV e falhar (`TypeError: metadata_errors`). O `stt.transcribe` passa o array do `extract.load_audio`.
11. **Ollama sem o modelo.** O servidor responde, mas `/api/chat` devolve 404. O `backend_status` confere a lista de modelos e mostra o `ollama pull` necessário.

### 7.3 Dados e métricas

12. **Gabarito corrigido a partir do modelo** enviesa o WER. Transcrever do zero.
13. **Duas normalizações diferentes** (uma no gabarito, outra na saída) inflam o WER. Usar só `norm()`.
14. **Referência vazia** em WER ou CER não tem significado. `metrics.wer` levanta `ValueError`.
15. **Ruído branco é simplificação.** O recorte em [-1, 1] pode alterar o SNR efetivo. O `add_noise` devolve o SNR medido, e o `run_experiments.py` avisa se ele fugir ±0,5 dB do alvo.
16. **Professor se movendo** gera keyframes falsos no baseline. No `slides`, a máscara de instabilidade cobre o professor num canto; se ele ocupar a tela inteira (câmera em tela cheia), o quadro pode virar "slide". Saídas: `ignore`/`roi`, ou agrupar quadros consecutivos cujo OCR tenha ≥ 90% de igualdade.
    - **Limites conhecidos do `slides`:** (a) um overlay fixo que só acrescenta tinta (ícone de pausa do player) pode virar o quadro candidato; (b) um slide diferente com o mesmo título e conteúdo contido no outro seria fundido; (c) fundo com imagem ou degradê ruidoso satura a máscara de tinta; (d) cartões de logo (abertura e encerramento) saem como slides; (e) os limiares foram ajustados em vídeos de um único curso (Coursera/Wharton, 12 aulas), mesmo visual e mesmo professor, sem gabarito de tempo: pode não generalizar.
17. **Amostra pequena.** 6 vídeos e 2 a 3 min de gabarito por vídeo. Reportar desvio e tamanho da amostra, sem tratar o resultado como prova.
18. **Saída de IA sem leitura.** Classificar os erros (nomes próprios, siglas, números, sotaque, ruído). As hipóteses ficam em `data/derived/<video>/stt/`.
19. **Alucinações do Whisper em silêncio** ("Legendas pela comunidade Amara.org", loops de repetição) inflam o WER com inserções. O `clean_segments` remove as conhecidas e registra o que removeu (`removed` no `.json`).
20. **Não determinismo do LLM.** Dois resumos da mesma transcrição diferem. Por isso há 2 repetições e média ± desvio.
21. **Trecho começando no meio da frase.** No teste da junção, o `base` apagou os primeiros ~15 s de um trecho limpo que começava no meio de uma frase, e o WER subiu de 0,22 (com ruído de 20 dB) para 0,43 (limpo). Comece cada trecho em `clips.csv` no início de uma frase.

### 7.4 Entrega

22. **`fetch` não funciona em `file://`.** Servir com `python -m http.server`.
23. **Demo ao vivo pode falhar.** Vídeo de backup gravado antes.
24. **Vídeos na pasta `web/`.** `web/data/` é gerado e fica fora do git.

---

## 8. Testes

| Arquivo | Cobre |
|---|---|
| `tests/test_smoke.py` | FFmpeg, Tesseract `por`, jiwer, faster-whisper, OpenCV e GPU (pulado sem CUDA) |
| `tests/test_metrics.py` | `norm` com acentos; WER e CER à mão; referência vazia; `wer_detail`; `word_pr`; `rtf`; `term_accuracy` |
| `tests/test_stt_vtt.py` | `fmt`; `write_vtt` (cabeçalho e blocos); `write_srt`; blocos dentro dos limites e sem órfãos; `clean_segments`; `plain_text`; `build_prompt` |
| `tests/test_degrade.py` | `add_noise`: SNR medido a ±0,5 dB do alvo em 20, 10 e 0 dB; mesma semente, mesmo resultado; `degrade_image` mantém o tamanho e recusa `kind` desconhecido |
| `tests/test_extract.py` | `keyframes` em vídeo sintético (3 "slides" → 3 keyframes); `out_dir` inexistente; áudio 16 kHz mono; recorte. `SlideTracker`/`slides`: bullets revelados (quadro final completo, volta ao slide sem duplicar), professor no canto (ruído), professor em tela cheia, volta mais completa substitui a parcial, relance transitório, gravação de arquivos e `frames.json` |
| `tests/test_summarize.py` | `extract_topics`; capítulos; backend desconhecido; requisição ao Ollama (simulada); requisição ao Claude com o SDK real e transporte HTTP simulado (fallback, streaming, custo) |

Regras: testes sem rede e sem carregar modelo do Whisper ou do EasyOCR.

---

## 9. Receitas rápidas

| Quero… | Mexa em… |
|---|---|
| Adicionar um modelo de STT | `--models` do `run_experiments.py` e a matriz da RQ1 do CLAUDE.md |
| Ajustar a sensibilidade de troca de slide | `extract.SlideTracker` (`keep_thr`, `grow_eps`, `step_s`; no baseline, `keyframes` `thr`). Registrar em `docs/decisoes.md` |
| Adicionar um motor de OCR | `ocr.py` (função `(path) -> (texto, tempo_s)`), `run_experiments.py` e teste |
| Adicionar uma condição de ruído ou degradação | `degrade.py`, `run_experiments.CONDITIONS` e a seção 5.5 |
| Adicionar uma métrica | `metrics.py`, uma linha em `raw.csv` e uma tabela ou figura em `report.py` |
| Mudar a normalização do texto | `metrics.norm`. **Mudança *major*:** invalida o `raw.csv` atual |
| Adicionar uma aula | `data/raw/` (com licença anotada no README), gabarito em `data/gt/`, linha em `clips.csv` |
| Trocar o LLM local | `--llm-model` (e `ollama pull`), ou `summarize.DEFAULT_MODELS` |
| Criar a busca da F6 | `data/gt/consultas.json` ⏳ e `metrics` (`precision_at_k`, `mrr`) |

---

## 10. O que ainda não existe

- `src/ocr.py` e a RQ2 (etapa 3).
- Vídeos com licença, gabaritos, `clips.csv` e `results/raw.csv` reais (etapa 1).
- Modelo SBC do artigo (pendência 4 do CLAUDE.md).
- F6 (consultas e métricas de busca), F7 (ablação) e F8 (HLS).
- Backup gravado da demo.
