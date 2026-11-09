# guia.md — Mapa do projeto Tema 9

**Versão 0.1 — 2026-11-09**

> Complementa o [CLAUDE.md](CLAUDE.md) (regras, decisões e roadmap). Descreve **o que existe hoje** e o contrato planejado de cada módulo, para não precisar reler o código inteiro. Nada foi implementado ainda: o que está marcado com ⏳ é planejado.
> Os detalhes de cada etapa (com trechos de código) estão no [guia de implementação](../Tema09_MVP_Guia_de_Implementacao.md), que é a fonte desta seção.

> ## Regra de manutenção (obrigatória)
>
> **Antes de modificar:** leia a seção do módulo, os contratos (seção 5), os parâmetros (seção 6) e as armadilhas (seção 7).
>
> **Ao modificar:** toda alteração de função, contrato, parâmetro, formato de arquivo ou comando entra neste guia **no mesmo passo**. Se código e guia divergirem, o código é a verdade e o guia deve ser corrigido.
>
> **Versão:** texto sem efeito de comportamento → não muda. Mudança de comportamento, contrato ou comando → *minor* (0.2, 0.3…). Mudança que invalida resultados já gerados (ex.: nova normalização de texto, que altera o WER) → *major* (1.0), e os `results/raw.csv` antigos precisam ser regerados.
>
> Este guia é uma foto do estado atual. Histórico de decisões vai em `docs/decisoes.md`.

---

## 1. Estado em uma olhada

| Etapa | Módulos | Estado |
|---|---|---|
| 0. Ambiente | `pyproject.toml`, `uv.lock`, `.python-version`, `Dockerfile`, `docker-compose.yml`, `.dockerignore`, `README.md`, `.gitignore`, `tests/test_smoke.py` | 🟡 build OK e 5 testes passam no contêiner; falta o NVIDIA Container Toolkit para o teste de GPU (ver README) |
| 1. Dataset e gabarito | `data/raw/`, `data/gt/` | ⏳ depende de licença e de transcrição manual |
| 2. STT (F1, parte de F3 e F4) | `src/extract.py` (áudio), `src/stt.py`, `src/degrade.py` (ruído), `src/metrics.py` (WER, RTF) | ⏳ |
| 3. OCR (F2, parte de F3 e F4) | `src/extract.py` (keyframes), `src/ocr.py`, `src/degrade.py` (imagem), `src/metrics.py` (CER, precisão, recall) | ⏳ |
| 4. Experimentos (F4) | `src/run_experiments.py`, `src/report.py` | ⏳ |
| 5. Interface (F5) | `web/index.html` | ⏳ |
| 6. Artigo | `artigo/` | ⏳ `[PENDENTE]` modelo SBC |
| Desejável | F6 (busca, `metrics.precision_at_k`, `recall_at_k`, `mrr`), F7 (ablação) | ⏳ não iniciado |
| Opcional | F8 (HLS), F9 (resumo por LLM) | ⏳ não iniciado |

### Ambiente verificado (guia de implementação, §1.1)

| Item | Situação |
|---|---|
| Python | 3.14.8 instalado. **Usar 3.12** via `uv` |
| `uv` | instalado |
| FFmpeg | instalado (9.0.2) |
| Tesseract | **não instalado**. Faltam `tesseract` e `tesseract-data-por` |
| GPU | NVIDIA RTX 3060, 12 GB (cabe o Whisper *medium*) |
| CPU e RAM | 24 threads, 62 GB |

Reporte o hardware no artigo: o RTF só é comparável dentro da mesma máquina.

---

## 2. Fluxo de dados

```
data/raw/aulaNN.mp4
  │
  ├─ ffmpeg (áudio mono 16 kHz) ──► data/derived/aulaNN.wav ──► trecho com gabarito (.clip.wav)
  │                                        │
  │                          degrade.add_noise (SNR 20, 10, 0 dB; seed 0)
  │                                        │
  │                          stt.transcribe (modelo × condição)
  │                                        ├─► stt.write_vtt ──► aulaNN.<modelo>.vtt
  │                                        └─► metrics.wer, metrics.rtf ──► results/raw.csv
  │
  └─ extract.keyframes (step_s, thr) ──► data/derived/aulaNN/frames/NNNN.png + frames.json
            │
            ├─ degrade.degrade_image (resize, blur, jpeg)
            ├─ ocr.ocr_tesseract / ocr.ocr_easyocr ──► slides.json (t, file, texto) + tempo
            └─ metrics.cer, metrics.word_pr ──► results/raw.csv

results/raw.csv ──► report.py ──► results/tabelas/, results/figuras/ (Tabelas 1 a 3, Figuras 1 e 2)

aulaNN.mp4 + aulaNN.<modelo>.vtt + slides.json ──► web/ ──► index.html (player, legenda, busca)
```

**Dependências entre módulos** (a direção é sempre de cima para baixo):

- `metrics.py` não importa nada do projeto. Só funções puras.
- `degrade.py` e `extract.py` dependem só de bibliotecas externas (OpenCV, numpy, soundfile).
- `stt.py` e `ocr.py` dependem das bibliotecas de modelo.
- `run_experiments.py` importa os módulos acima e orquestra a matriz.
- `report.py` só lê `results/raw.csv`. Não recalcula métrica.
- `web/` só lê arquivos gerados. Não importa código Python.

---

## 3. Comandos

| Comando | Faz | Observação |
|---|---|---|
| `tesseract --list-langs \| grep por` | Confirma o idioma português do Tesseract | Deve listar `por`. Se não listar, pedir ao usuário a instalação |
| `docker build -t falai:latest .` | Constrói a imagem (CUDA 12.4, FFmpeg, Tesseract `por`, dependências fixadas) | Etapa 0. Repetir só quando o `uv.lock` ou o `Dockerfile` mudar |
| `HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose run --rm pipeline` | Roda `pytest -q` no contêiner, com GPU se o NVIDIA Container Toolkit estiver instalado | Etapa 0. Ver README |
| `uv run pytest -q` | Mesmos testes no ambiente local (exige Tesseract `por` no host) | Alternativa sem contêiner |
| `uv lock` | Atualiza o `uv.lock` depois de mudar uma versão no `pyproject.toml` | Não usar `pip install` avulso |
| `python -c "import faster_whisper, cv2, jiwer, pytesseract; print('ok')"` | Smoke test das bibliotecas | Etapa 0. O EasyOCR entra na etapa 3 (traz o PyTorch) |
| `ffmpeg -i X.mp4 -vn -ac 1 -ar 16000 X.wav` | Extrai o áudio mono a 16 kHz | Guia §3.1 |
| `ffmpeg -i X.wav -ss 300 -t 150 X.clip.wav` | Recorta 2 min 30 s a partir de 5 min | `-ss` e `-t` em segundos |
| `python src/run_experiments.py` ⏳ | Roda a matriz e grava `results/raw.csv` | Flags a definir na etapa 4 |
| `python src/report.py` ⏳ | Gera tabelas e figuras a partir do CSV | Sem argumentos |
| `cd web && python -m http.server` | Serve a demo | Porta 8000. Abrir o HTML direto (`file://`) não funciona (o `fetch` é bloqueado) |

---

## 4. Módulos

### 4.1 `src/extract.py` ⏳ (etapas 2 e 3)

**Áudio.** Feito pelo comando `ffmpeg` da seção 3 (16 kHz, mono). Não há função própria planejada.

**`keyframes(video, out_dir, step_s=1.0, thr=0.04) -> list[dict]`**
- Amostra um quadro a cada `step_s` segundos.
- Reduz o quadro para 320×180 em tons de cinza. Guarda o quadro quando a diferença média para o último keyframe guardado, normalizada em 0–1, passa de `thr`.
- Grava `NNNN.png` (quadro original em cores) e `frames.json` (lista de `{"t": segundos, "file": caminho}`), e devolve a mesma lista.
- Pré-condição: `out_dir` existe. `cv2.imwrite` não cria diretório.
- Ablação F7: variar `step_s` e `thr`.

### 4.2 `src/stt.py` ⏳ (etapa 2)

**`fmt(t: float) -> str`**: segundos para `HH:MM:SS.mmm` (formato WebVTT).

**`transcribe(wav, size, device="cuda", compute="float16") -> (segs, duracao_s, tempo_s)`**
- Carrega `WhisperModel(size, device, compute_type)`, transcreve com `language="pt"`, `beam_size=5`, `vad_filter=True`.
- Consome o gerador com `list(segs)` antes de parar o cronômetro. Só a inferência é cronometrada; o carregamento do modelo fica fora.
- `duracao_s` vem de `info.duration`.

**`write_vtt(segs, path)`**: escreve `WEBVTT`, linha em branco e um bloco por segmento (`início --> fim`, texto sem espaços das pontas).

### 4.3 `src/ocr.py` ⏳ (etapa 3)

**`ocr_tesseract(path) -> (texto, tempo_s)`**: `pytesseract.image_to_string(cv2.imread(path), lang="por", config="--psm 6")`. Testar `--psm 3` e usar o melhor para os dois motores, ou relatar a diferença.

**`ocr_easyocr(path) -> (texto, tempo_s)`**: `" ".join(_reader.readtext(path, detail=0))`.

Ambas seguem a assinatura `(path) -> (texto, tempo_s)`. Um novo motor deve seguir a mesma assinatura.

O exemplo do guia cria o `easyocr.Reader` no topo do módulo. Ver armadilha 7.2.1.

### 4.4 `src/degrade.py` ⏳ (etapas 2 e 3)

**`add_noise(in_wav, out_wav, snr_db, seed=0)`**
- Soma ruído branco gaussiano ao áudio, com potência do ruído = potência média do sinal / 10^(SNR/10).
- Recorta em [-1, 1]. Mesma semente, mesmo resultado.
- Condições do RQ1: limpo, 20, 10 e 0 dB.

**`degrade_image(img, kind, level)`**
- `"resize"`, `level` 0,5 (ou 0,25): reduz e volta ao tamanho original.
- `"blur"`, `level` = tamanho ímpar do kernel (5 ou 9).
- `"jpeg"`, `level` = qualidade (30).
- Condições do RQ2: limpo, resize 0,5, blur 9, JPEG 30.

### 4.5 `src/metrics.py` ⏳ (etapa 2, ampliado na 3)

Funções puras. Nenhuma lê arquivo.

| Função | Faz |
|---|---|
| `norm(t) -> str` | Minúsculas, pontuação trocada por espaço (`[^\w\s]`), acentos mantidos, espaços colapsados. **É a única normalização do projeto.** |
| `wer(ref, hyp)` | `jiwer.wer` sobre os textos normalizados |
| `cer(ref, hyp)` | `jiwer.cer` sobre os textos normalizados |
| `word_pr(ref, hyp) -> (precisao, recall)` | Contagem por multiconjunto de palavras; ignora a ordem |
| `rtf(tempo_proc_s, duracao_s)` | Tempo de processamento / duração da mídia |
| `precision_at_k`, `recall_at_k`, `mrr` ⏳ (F6) | Métricas de busca |

### 4.6 `src/run_experiments.py` ⏳ (etapa 4)

Roda a matriz da seção 4 do CLAUDE.md e grava `results/raw.csv` (contrato na seção 5.5).

- Uma linha por **execução × métrica**. `[PREMISSA]` um STT gera uma linha `wer` e uma linha `rtf`; um OCR gera linhas `cer`, `precisao_palavras`, `recall_palavras`.
- Tempo: 3 repetições por configuração (`repeticao` = 1, 2, 3).
- Para OCR, o tempo deve ser reportado **por minuto de mídia** (Tabela 3 do guia). Para isso, somam-se os tempos das imagens e divide-se pela duração do vídeo. `[PENDENTE]` definir se a leitura da imagem entra nesse tempo.
- Modelo carregado uma vez por (modelo, condição) e reutilizado nas 3 repetições.
- Seed do ruído sempre 0.

### 4.7 `src/report.py` ⏳ (etapa 4)

Só lê `results/raw.csv` e produz:

| Saída | Conteúdo |
|---|---|
| Tabela 1 | WER médio ± desvio, por modelo × condição de ruído |
| Tabela 2 | CER, precisão e recall, por motor de OCR × degradação |
| Figura 1 | WER × SNR, uma linha por modelo (robustez) |
| Figura 2 | RTF × WER, um ponto por modelo (trade-off qualidade × custo) |
| Tabela 3 | Tempo de inferência por minuto de mídia, STT e OCR |

Nenhum número digitado à mão.

### 4.8 `web/index.html` ⏳ (etapa 5)

- Página estática. Espera `aulaNN.mp4`, `aulaNN.<modelo>.vtt` e `aulaNN.slides.json` na mesma pasta.
- Player com `<track kind="subtitles">` apontando para o `.vtt`.
- Campo de busca: substring sem diferença de caixa sobre o texto dos slides. Clicar num item faz `video.currentTime = t`.
- Contrato do JSON: seção 5.3.
- A busca sobre a transcrição (F6) entra depois, sobre o mesmo índice de texto.

---

## 5. Contratos de dados

**5.1 Gabarito de fala** `data/gt/<video>.stt.txt`: texto puro, transcrito à mão, nas regras de normalização do README.

**5.2 Gabarito de slides** `data/gt/<video>.slides.json`: lista `[{"t": 125.0, "text": "..."}]`, como no guia de implementação (§2.3). `t` em segundos.

**5.3 Slides derivados** `frames.json` e `slides.json`: lista `[{"t": 125.0, "file": "...", "texto": "..."}]`. Usada pelo `web/`.

> **Pendência de contrato.** O gabarito usa `text` e a saída usa `texto`. Decidir uma chave antes da etapa 3 (recomendação: `texto` nos dois, `t` em segundos). Depois, ajustar `metrics` e `web/`, e marcar a mudança como *minor* neste guia.

**5.4 Legenda** `aulaNN.<modelo>.vtt`: `WEBVTT`, linha em branco, blocos `HH:MM:SS.mmm --> HH:MM:SS.mmm` com o texto na linha seguinte.

**5.5 Resultados** `results/raw.csv`, um registro por execução × métrica:

```
video,tarefa,modelo,condicao,repeticao,metrica_erro,valor_erro,tempo_s,duracao_s,rtf
```

- `tarefa`: `stt` ou `ocr`.
- `condicao`: `limpo`, `snr20`, `snr10`, `snr0`, `resize0.5`, `blur9`, `jpeg30`.
- `valor_erro`: valor da métrica de erro, vazio quando a linha for só de tempo.
- `[PREMISSA]` separador vírgula, padrão do guia de implementação (lido com `pandas.read_csv` sem opções extras).

**5.6 Nomes**: aulas são `aula01`, `aula02`… Modelos de STT são `tiny`, `base`, `small`, `medium`.

---

## 6. Parâmetros fixos

Todos valem igual para todos os modelos e motores de uma mesma comparação.

| Parâmetro | Valor | Onde | Observação |
|---|---|---|---|
| Idioma do STT | `pt` | `stt.transcribe` | |
| `beam_size` | 5 | `stt.transcribe` | |
| `vad_filter` | `True` | `stt.transcribe` | |
| Dispositivo STT | `cuda` / `float16` | `stt.transcribe` | Fallback `cpu` / `int8`. Usar a mesma escolha em todos os experimentos |
| Modelos STT | `tiny`, `base`, `small` (`medium` se couber) | RQ1 | |
| Tesseract | `lang="por"`, `--psm 6` | `ocr.ocr_tesseract` | Testar `--psm 3` |
| EasyOCR | `["pt"]`, `gpu=True` | `ocr.py` | Ver armadilha 7.2.1 |
| `step_s` (keyframes) | 1,0 s | `extract.keyframes` | Ablação F7 |
| `thr` (keyframes) | 0,04 | `extract.keyframes` | Calibrar entre 0,03 e 0,05 olhando as imagens. Meta: ~1 keyframe por slide |
| Resolução de análise | 320×180, cinza | `extract.keyframes` | Só para comparar quadros |
| SNR | limpo, 20, 10, 0 dB | RQ1 | |
| Seed do ruído | 0 | `degrade.add_noise` | |
| Degradações de imagem | resize 0,5; blur 9; JPEG 30 | RQ2 | |
| Repetições de tempo | 3 | experimentos | Média ± desvio |
| Trecho com gabarito | 2 a 3 min por vídeo | dataset | |
| Vídeos da matriz | 6, de 5 a 15 min | dataset | Vídeos longos (~45 min) ficam fora da matriz |

---

## 7. Armadilhas

### 7.1 Medição (afetam a nota de métricas)

1. **Gerador preguiçoso.** `transcribe()` não transcreve até ser consumido. Sem `list(segs)` dentro do cronômetro, o tempo sai quase zero.
2. **Aquecimento.** A primeira execução inclui a inicialização da GPU. Fazer uma execução de aquecimento e descartá-la. O exemplo do guia não tem isso ainda.
3. **Carregamento do modelo.** Medir à parte. O exemplo recarrega o modelo a cada chamada: nas 3 repetições, carregar uma vez por (modelo, condição).
4. **Parâmetros iguais.** `language`, `beam_size` e `vad_filter` iguais em todos os modelos.
5. **Hardware.** Reportar no artigo. RTF só compara dentro da mesma máquina.
6. **Escopo do tempo de OCR.** Definir se a leitura da imagem entra no tempo e manter igual para os dois motores.

### 7.2 Código de exemplo do guia de implementação (não executado)

1. **`easyocr.Reader` no import.** `_reader = easyocr.Reader(["pt"], gpu=True)` no topo do módulo carrega o modelo e exige GPU só de importar `ocr.py`. Criar sob demanda, numa função com cache.
2. **`device="cuda"` por padrão.** Sem CUDA, falha. Fallback CPU com `compute="int8"`.
3. **Bibliotecas CUDA ausentes** (cuBLAS, cuDNN): instalar as bibliotecas ou usar CPU. Usar a mesma opção em todos os experimentos.
4. **`cap.set(CAP_PROP_POS_FRAMES)` a cada passo** faz seek em cada amostra e pode ser lento em vídeo longo. Medir antes de trocar por leitura sequencial com `cap.grab()`.
5. **`cv2.imwrite` não cria diretório.** Criar `out_dir` antes e conferir o retorno (`True`).
6. **`degrade_image` com `kind` desconhecido** devolve `None` sem erro. Levantar `ValueError`.
7. **`cv2.imread` com caminho inexistente** devolve `None`, e o pytesseract falha com mensagem pouco clara. Conferir antes.
8. **`lang="por"` sem o pacote.** Falha se faltar `tesseract-data-por`. Instalação é de sistema: pedir ao usuário.
9. **Python 3.14.** Pacotes de ML podem não ter binários para essa versão. Usar 3.12.

### 7.3 Dados e métricas

10. **Gabarito corrigido a partir do modelo** enviesa o WER. Transcrever do zero.
11. **Duas normalizações diferentes** (uma no gabarito, outra na saída) inflam o WER. Usar só `norm()`.
12. **Chaves `text` e `texto`** (seção 5.2 e 5.3). Enquanto não for decidida, o `web/` e o `metrics` podem ler campos diferentes.
13. **Referência vazia** em WER ou CER não tem significado. Tratar antes de chamar `jiwer`.
14. **Ruído branco é simplificação.** O recorte em [-1, 1] pode alterar o SNR efetivo. Testar: o SNR medido deve ficar a ±0,5 dB do alvo.
15. **Professor se movendo** gera keyframes falsos. Saídas: recortar a região do slide, ou agrupar quadros consecutivos cujo OCR tenha ≥ 90% de igualdade.
16. **Amostra pequena.** 6 vídeos e 2 a 3 min de gabarito por vídeo. Reportar desvio e tamanho da amostra, sem tratar o resultado como prova.
17. **Saída de IA sem leitura.** Classificar os erros (nomes próprios, siglas, números, sotaque, ruído). O enunciado exige a análise técnica dos alunos (regra 15).

### 7.4 Entrega

18. **`fetch` não funciona em `file://`.** Servir com `python -m http.server`.
19. **Demo ao vivo pode falhar.** Vídeo de backup gravado antes.
20. **Vídeos na pasta `web/`.** Copiar só o necessário para a demo. Não versionar `.mp4` no git.

---

## 8. Testes (planejados)

| Arquivo (⏳) | Cobre |
|---|---|
| `tests/test_smoke.py` | Imports das bibliotecas e idioma `por` no Tesseract (etapa 0) |
| `tests/test_metrics.py` | WER, CER, precisão e recall à mão em frases curtas; `norm` com acentos; RTF |
| `tests/test_stt_vtt.py` | `fmt` (formato `HH:MM:SS.mmm`); `write_vtt` (cabeçalho e blocos) |
| `tests/test_degrade.py` | `add_noise`: mesma semente, mesmo resultado; SNR medido ≈ alvo; `degrade_image` mantém tamanho e recusa `kind` desconhecido |
| `tests/test_extract.py` | `keyframes` em vídeo sintético (cores alternadas, número de quadros esperado); `out_dir` inexistente |

Regras: testes sem rede. Testes que carregam modelo do Whisper ou do EasyOCR ficam marcados como lentos e não rodam por padrão.

---

## 9. Receitas rápidas

| Quero… | Mexa em… |
|---|---|
| Adicionar um modelo de STT | `run_experiments.py` (lista de modelos) e a matriz da RQ1 do CLAUDE.md. Repetir a RQ1 para ele |
| Ajustar a sensibilidade de troca de slide | `extract.keyframes` (`thr`, `step_s`). Registrar em `docs/decisoes.md` |
| Adicionar um motor de OCR | `ocr.py` (função com a assinatura `(path) -> (texto, tempo_s)`), `run_experiments.py` e teste |
| Adicionar uma condição de ruído ou degradação | `degrade.py` e a matriz em `run_experiments.py`. Atualizar a seção 5.5 (`condicao`) |
| Adicionar uma métrica | `metrics.py`, uma linha em `raw.csv` e uma tabela ou figura em `report.py` |
| Mudar a normalização do texto | `metrics.norm`. **Mudança *major*:** invalida o `raw.csv` atual |
| Adicionar uma aula | `data/raw/` (com licença anotada no README), gabarito em `data/gt/` e a extração de `data/derived/` |
| Criar a busca da F6 | `data/gt/consultas.json` ⏳ (consultas e trechos relevantes) e `metrics` (`precision_at_k`, `mrr`) |

---

## 10. O que ainda não existe

- Código do pipeline (etapas 0 a 6): `src/` tem só o `__init__.py`; `web/` não existe.
- `data/raw/`, `data/gt/` e `data/derived/`: vídeos e gabaritos ainda não foram escolhidos nem transcritos.
- `results/raw.csv` e as tabelas e figuras do artigo.
- `artigo/`: falta o modelo SBC (pendência 4 do CLAUDE.md).
- `docs/decisoes.md`.
- F6, F7, F8 e F9.
- Repositório: <https://github.com/luiz2801/FalAI> (público). A pasta local `tema9/` é o clone de trabalho.
