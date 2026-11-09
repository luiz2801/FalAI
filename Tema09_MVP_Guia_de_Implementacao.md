# Tema 9 — Guia de implementação do MVP

Complementa o [planejamento](Tema09_IA_Sistemas_Multimidia_Planejamento.md). Este guia cobre os requisitos **F1 a F5** (obrigatórios) e aponta onde encaixar F6 e F7.

> **Aviso:** os trechos de código abaixo são ponto de partida e **não foram executados**. A etapa 0 e a etapa 2 incluem testes para validá-los. As versões das bibliotecas devem ser fixadas no `pyproject.toml` assim que o ambiente funcionar.

---

## 0. O que é o MVP

**Entrada:** vídeos de aula (MP4). **Saída:**

| Entrega | Requisito |
|---|---|
| Transcrição PT-BR e legenda `.vtt` por vídeo | F1 |
| Texto dos slides (OCR) com marca de tempo | F2 |
| Comparação de ≥ 2 modelos de STT e ≥ 2 motores de OCR | F3 |
| Tabelas de WER, CER e tempo de inferência, com ruído/degradação | F4 |
| Página web: vídeo + legenda + painel de slides clicável | F5 |

**Pronto quando:**

- [ ] `python src/run_experiments.py` gera `results/raw.csv` sem intervenção manual.
- [ ] `python src/report.py` gera as tabelas e figuras do artigo a partir do CSV.
- [ ] A página web abre com legenda sincronizada e clique no slide leva ao instante do vídeo.
- [ ] Existe um README com os comandos para reproduzir tudo.

---

## 1. Ambiente (etapa 0, ~1 dia)

### 1.1 O que já foi verificado nesta máquina

| Item | Situação |
|---|---|
| Python | 3.14.8 (muito novo, ver abaixo) |
| `uv` | instalado |
| FFmpeg | instalado (9.0.2) |
| Tesseract | **não instalado** |
| GPU | NVIDIA RTX 3060, 12 GB (cabe até o modelo Whisper *medium*) |
| CPU / RAM | 24 threads / 62 GB |

Os colegas devem rodar a mesma verificação (`ffmpeg -version`, `nvidia-smi`, `python3 --version`). **Reporte o hardware no artigo**, porque o RTF só é comparável dentro da mesma máquina.

### 1.2 Instalação

```bash
# Tesseract + idioma português (Arch; em outras distros, procure os pacotes equivalentes)
sudo pacman -S tesseract tesseract-data-por

# Use Python 3.12 em vez do 3.14: pacotes de ML (torch, ctranslate2) costumam
# demorar a ter binários para versões muito novas do Python.
uv venv --python 3.12
source .venv/bin/activate
uv pip install faster-whisper jiwer opencv-python-headless pytesseract \
               easyocr soundfile numpy pandas matplotlib
```

### 1.3 Testes de fumaça (não pule)

```bash
tesseract --list-langs | grep por        # deve listar "por"
python -c "import faster_whisper, cv2, jiwer, easyocr, pytesseract; print('ok')"
```

Rode o Whisper em um áudio curto na GPU. Se aparecer erro de biblioteca CUDA (cuBLAS/cuDNN), há duas saídas:

1. instalar as bibliotecas CUDA exigidas pelo faster-whisper;
2. usar `device="cpu", compute_type="int8"` (mais lento, mas serve para o MVP).

Seja qual for a escolha, **use a mesma em todos os experimentos**.

### 1.4 Estrutura do repositório

```
tema9/
├── README.md
├── pyproject.toml
├── data/
│   ├── raw/        # vídeos originais (não versionar no git)
│   ├── gt/         # gabaritos: <video>.stt.txt e <video>.slides.json
│   └── derived/    # áudio, áudio com ruído, frames, .vtt, .json
├── src/
│   ├── extract.py          # FFmpeg: áudio + keyframes com timestamp
│   ├── stt.py              # transcrição + escrita de .vtt
│   ├── ocr.py              # Tesseract e EasyOCR
│   ├── degrade.py          # ruído no áudio, blur/redução na imagem
│   ├── metrics.py          # normalização, WER, CER, precisão/recall, RTF
│   ├── run_experiments.py  # roda a matriz e grava results/raw.csv
│   └── report.py           # tabelas e figuras
├── web/index.html
└── results/
```

---

## 2. Dataset e gabarito (etapa 1, em paralelo com a 0 — é o gargalo)

### 2.1 Vídeos

- **6 vídeos** de aula, de 5 a 15 min cada (os de 45 min só servem para medir tempo e para a demo).
- Fontes: gravar uma aula/palestra própria ou usar vídeo com licença livre. **Anote a licença** de cada vídeo.
- Se gravar pessoas (professor, colegas), **peça autorização** de uso da imagem e da voz.
- Misture condições: sala silenciosa, sala com eco, sotaques diferentes, slides limpos e slides filmados de longe.

### 2.2 Gabarito de fala (RQ1)

- Escolha **1 trecho de 2 a 3 min por vídeo** e transcreva **do zero, à mão**.
- **Não corrija a saída de um modelo.** Isso enviesa o WER a favor desse modelo.
- Regras fixas, escritas no README e seguidas por todos: números (algarismos ou por extenso), como tratar hesitações ("é...", "né") e siglas.
- Custo estimado: ~15 a 20 min de trabalho por trecho.

### 2.3 Gabarito de slides (RQ2)

- Se você tem o **PDF/PPTX original** da aula, extraia o texto de cada slide (por exemplo com `pdftotext`) e use como gabarito. É o caminho mais barato e confiável.
- Se não tem, digite à mão o texto de 10 a 15 slides por vídeo.
- Grave em `data/gt/<video>.slides.json`: `[{"t": 125.0, "text": "..."}, ...]`.

---

## 3. F1: transcrição e legenda (etapa 2, ~2 dias)

### 3.1 Extrair o áudio

```bash
ffmpeg -i data/raw/aula01.mp4 -vn -ac 1 -ar 16000 data/derived/aula01.wav
# recortar o trecho anotado (ex.: de 5:00 a 7:30)
ffmpeg -i data/derived/aula01.wav -ss 300 -t 150 data/derived/aula01.clip.wav
```

### 3.2 Transcrever

```python
# src/stt.py
import time
from faster_whisper import WhisperModel

def fmt(t):
    h, r = divmod(t, 3600); m, s = divmod(r, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"

def transcribe(wav, size, device="cuda", compute="float16"):
    model = WhisperModel(size, device=device, compute_type=compute)
    t0 = time.perf_counter()
    segs, info = model.transcribe(wav, language="pt", beam_size=5, vad_filter=True)
    segs = list(segs)                      # transcribe() devolve um gerador: consuma
    dt = time.perf_counter() - t0          # antes de parar o cronômetro
    return segs, info.duration, dt

def write_vtt(segs, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for s in segs:
            f.write(f"{fmt(s.start)} --> {fmt(s.end)}\n{s.text.strip()}\n\n")
```

**Armadilhas de medição de tempo** (afetam diretamente a nota de métricas):

- `transcribe()` é preguiçoso. Sem `list(segs)`, você mede quase zero.
- A primeira execução inclui inicialização da GPU. Faça **1 execução de aquecimento** descartada.
- Meça separadamente o **tempo de carregar o modelo** e o **tempo de inferência**. Reporte só o segundo como RTF.
- Mantenha igual entre modelos: `language`, `beam_size`, `vad_filter`.

### 3.3 Modelos a comparar (F3)

`tiny`, `base`, `small` (e `medium`, se a GPU permitir). A pergunta é onde o ganho de WER deixa de compensar o custo de tempo.

### 3.4 Medir erro

```python
# src/metrics.py
import re, jiwer
from collections import Counter

def norm(t):
    t = t.lower()
    t = re.sub(r"[^\w\s]", " ", t)         # mantém acentos
    return re.sub(r"\s+", " ", t).strip()

def wer(ref, hyp): return jiwer.wer(norm(ref), norm(hyp))
def cer(ref, hyp): return jiwer.cer(norm(ref), norm(hyp))

def word_pr(ref, hyp):
    """Precisão/recall de palavras, ignorando a ordem (bom para slides)."""
    r, h = Counter(norm(ref).split()), Counter(norm(hyp).split())
    inter = sum((r & h).values())
    return inter / max(sum(h.values()), 1), inter / max(sum(r.values()), 1)

def rtf(tempo_proc_s, duracao_s): return tempo_proc_s / duracao_s
```

**Definições para a defesa:**

- **WER** = (substituições + remoções + inserções) / nº de palavras do gabarito.
- **CER** = o mesmo, em nível de caractere.
- **RTF** = tempo de processamento / duração da mídia. RTF < 1 significa mais rápido que o tempo real.

### 3.5 Teste de ruído (F4)

```python
# src/degrade.py
import numpy as np, soundfile as sf

def add_noise(in_wav, out_wav, snr_db, seed=0):
    x, sr = sf.read(in_wav)
    rng = np.random.default_rng(seed)
    p_noise = np.mean(x**2) / (10 ** (snr_db / 10))
    y = np.clip(x + rng.normal(0, np.sqrt(p_noise), x.shape), -1, 1)
    sf.write(out_wav, y, sr)
```

- Condições: **limpo, SNR 20 dB, 10 dB e 0 dB**.
- Fixe a semente para reprodutibilidade.
- Ruído branco é uma simplificação. Se tiver tempo, misture também um ruído real (sala de aula, ventilador). Registre essa limitação no artigo.

**Definição de pronto da etapa:** dado um trecho com gabarito, um comando imprime WER e RTF para cada modelo e cada condição de ruído.

---

## 4. F2: keyframes e OCR (etapa 3, ~2 dias)

### 4.1 Extrair keyframes com marca de tempo

```python
# src/extract.py
import cv2, json, numpy as np

def keyframes(video, out_dir, step_s=1.0, thr=0.04):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(fps * step_s))
    prev, out = None, []
    for i in range(0, n, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2GRAY)
        if prev is None or np.mean(cv2.absdiff(g, prev)) / 255 > thr:
            path = f"{out_dir}/{len(out):04d}.png"
            cv2.imwrite(path, frame)
            out.append({"t": round(i / fps, 2), "file": path})
            prev = g
    json.dump(out, open(f"{out_dir}/frames.json", "w"), indent=1)
    return out
```

- **Calibre `thr` olhando as imagens.** Trocas de slide com fundo claro geram diferença pequena; comece em 0,03 a 0,05 e ajuste até ficar com ~1 imagem por slide.
- Falsos positivos vêm do professor se mexendo. Duas saídas: recortar a região do slide ou juntar frames consecutivos cujo texto de OCR seja ≥ 90% igual.
- Variar `step_s` e `thr` é a ablação F7.

### 4.2 Dois motores de OCR (F3)

```python
# src/ocr.py
import time, cv2, pytesseract, easyocr

_reader = easyocr.Reader(["pt"], gpu=True)

def ocr_tesseract(path):
    t0 = time.perf_counter()
    txt = pytesseract.image_to_string(cv2.imread(path), lang="por", config="--psm 6")
    return txt, time.perf_counter() - t0

def ocr_easyocr(path):
    t0 = time.perf_counter()
    txt = " ".join(_reader.readtext(path, detail=0))
    return txt, time.perf_counter() - t0
```

Teste `--psm 3` contra `--psm 6` no Tesseract e use o melhor para os dois ou relate a diferença. (Alternativa ao EasyOCR: PaddleOCR, com instalação mais pesada.)

### 4.3 Degradações para robustez

```python
def degrade_image(img, kind, level):
    import cv2
    if kind == "resize":   # level = 0.5, 0.25
        h, w = img.shape[:2]
        small = cv2.resize(img, (int(w*level), int(h*level)))
        return cv2.resize(small, (w, h))           # volta ao tamanho original
    if kind == "blur":     # level = 5, 9 (tamanho do kernel, ímpar)
        return cv2.GaussianBlur(img, (level, level), 0)
    if kind == "jpeg":     # level = 30 (qualidade)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, level])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
```

### 4.4 Medir

Para cada slide com gabarito, associe o keyframe mais próximo no tempo e calcule **CER**, **precisão** e **recall** de palavras (`word_pr`, que ignora a ordem de leitura) e o tempo por imagem.

**Definição de pronto da etapa:** dado um vídeo, um comando produz `slides.json` (`t`, `file`, `texto`) para cada motor e imprime CER/precisão/recall frente ao gabarito.

---

## 5. F4: rodar os experimentos (etapa 4, ~2 dias)

### 5.1 Matriz

| | Fatores | Execuções (6 vídeos) |
|---|---|---|
| **RQ1 (STT)** | 3 modelos × 4 condições (limpo, 20, 10, 0 dB) | 72 |
| **RQ2 (OCR)** | 2 motores × 4 condições (limpo, resize 0,5, blur 9, JPEG 30) | 48 |

- **Exatidão:** nos trechos com gabarito (2 a 3 min).
- **Tempo:** repita **3 vezes** cada configuração e reporte média e desvio.
- Rode o primeiro trecho, meça quanto demora e só então extrapole para a matriz inteira.
- Só use os vídeos completos para medir tempo em mídia longa e para a demo.

### 5.2 Formato de saída (`results/raw.csv`)

```
video, tarefa, modelo, condicao, repeticao, metrica_erro, valor_erro, tempo_s, duracao_s, rtf
```

Uma linha por execução. O `report.py` só agrega; nada de número digitado à mão.

### 5.3 O que o `report.py` gera para o artigo

| Saída | Conteúdo |
|---|---|
| Tabela 1 | WER médio ± desvio por modelo × condição de ruído |
| Tabela 2 | CER, precisão e recall por motor de OCR × degradação |
| Figura 1 | WER × SNR, uma linha por modelo (**robustez**) |
| Figura 2 | RTF × WER, um ponto por modelo (**trade-off qualidade × custo**) |
| Tabela 3 | Tempo de inferência por minuto de mídia, STT e OCR |

---

## 6. F5: interface de demonstração (etapa 5, ~1 a 2 dias)

Uma página estática basta. Os dados vêm dos arquivos gerados.

```html
<!-- web/index.html -->
<video id="v" controls width="720" src="aula01.mp4">
  <track kind="subtitles" srclang="pt" label="Whisper small" src="aula01.small.vtt" default>
</video>
<input id="q" placeholder="Buscar nos slides…">
<ul id="slides"></ul>
<script>
fetch("aula01.slides.json").then(r => r.json()).then(items => {
  const ul = document.getElementById("slides"), v = document.getElementById("v");
  const draw = f => {
    ul.innerHTML = "";
    items.filter(i => i.texto.toLowerCase().includes(f.toLowerCase())).forEach(i => {
      const li = document.createElement("li");
      li.textContent = `[${Math.floor(i.t/60)}:${String(Math.floor(i.t%60)).padStart(2,"0")}] ${i.texto.slice(0,120)}`;
      li.onclick = () => { v.currentTime = i.t; v.play(); };
      ul.appendChild(li);
    });
  };
  document.getElementById("q").oninput = e => draw(e.target.value);
  draw("");
});
</script>
```

- Copie o MP4, o `.vtt` e o `.slides.json` para `web/` e sirva com `python -m http.server` (o `fetch` não funciona abrindo o arquivo direto).
- A busca por substring já entrega o embrião do F6. Para F6 de verdade, indexe também a transcrição e meça Precision@k e MRR com as consultas do gabarito.

**Roteiro da demo (≈ 3 min):** abrir o vídeo, mostrar a legenda sincronizada, buscar um termo, clicar no slide e saltar ao trecho, e mostrar lado a lado uma saída boa e uma ruim (ruído 0 dB) para ilustrar a robustez.

---

## 7. Ordem de execução e responsabilidades

| Etapa | Duração | Dono sugerido | Pronto quando |
|---|---|---|---|
| 0. Ambiente | 1 dia | todos | testes de fumaça passam em cada máquina |
| 1. Dataset e gabarito | 2 a 3 dias | pessoa 3 (com ajuda) | 6 vídeos, 6 trechos transcritos, slides com gabarito |
| 2. STT | 2 dias | pessoa 1 | WER e RTF por modelo e ruído num trecho |
| 3. OCR | 2 dias | pessoa 2 | CER/precisão/recall por motor num vídeo |
| 4. Experimentos | 2 dias | pessoas 1 e 2 | `raw.csv` completo |
| 5. Interface | 1 a 2 dias | pessoa 3 | demo roda de ponta a ponta |
| 6. Relatório | 1 dia | pessoa 3 | tabelas e figuras geradas por script |

As etapas 2 e 3 podem andar em paralelo assim que o dataset existir. A etapa 1 começa no primeiro dia, porque a transcrição manual leva tempo.

---

## 8. Armadilhas que custam nota

| Armadilha | Consequência | Como evitar |
|---|---|---|
| Gabarito feito corrigindo a saída de um modelo | WER enviesado | Transcrever do zero |
| Medir tempo sem consumir o gerador, ou incluindo o carregamento do modelo | Tempos sem sentido | Aquecimento, `list(segs)`, cronometrar só a inferência |
| Comparar modelos com parâmetros diferentes | Comparação injusta | Fixar `beam_size`, `vad_filter`, idioma |
| Normalização diferente entre gabarito e saída | WER inflado | Uma única função `norm()` |
| Amostra pequena tratada como prova | Conclusão exagerada | Reportar desvio, citar o tamanho da amostra e ser cauteloso |
| Aceitar a saída da IA sem olhar | Perde a "análise técnica dos alunos" | Ler os erros e classificá-los (nomes próprios, siglas, números, sotaque) |
| Hardware não informado | RTF não comparável | Descrever CPU/GPU no artigo |
| Demo só ao vivo | Risco de falha na hora | Gravar um backup em vídeo |

---

## 9. Depois do MVP

1. **F6, busca:** indexar transcrição e OCR, criar 20 a 30 consultas com gabarito e medir Precision@5 e MRR (por texto e, opcionalmente, com embeddings CLIP).
2. **F7, ablação:** variar a taxa de amostragem de keyframes e mostrar o efeito no recall de slides e no tempo.
3. **F8, HLS:** empacotar o vídeo em HLS com FFmpeg e comparar tempo de início e buffering com e sem a IA rodando em paralelo.
4. **F9, resumo:** só se sobrar tempo, porque precisa de avaliação própria.
