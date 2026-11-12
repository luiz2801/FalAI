# FalAI: análise de vídeos de aula com IA

Pipeline de transcrição (STT), keyframes e OCR de slides, resumo por LLM e avaliação de qualidade e tempo, para a disciplina Redes Multimídia (EN05228, UFPA, Tema 9). Projeto acadêmico. O plano completo está em [CLAUDE.md](CLAUDE.md), o mapa do código em [guia.md](guia.md) e o histórico de decisões em [docs/decisoes.md](docs/decisoes.md).

**Pergunta de investigação:** qual é o trade-off entre qualidade (WER/CER) e tempo de processamento de modelos de STT e OCR em vídeos de aula com ruído? Secundária (F9): o erro do STT piora o resumo gerado por LLM?

**Estado:** STT, degradação, métricas, experimentos (RQ1), relatório, página da demo e F9 prontos e testados. Falta o OCR (etapa 3) e os dados reais (vídeos com licença e gabaritos humanos).

```
Vídeo ─► FFmpeg (16 kHz mono) ─► Whisper (STT) ─► .vtt / .txt ─────────┐
  │              └─ ruído (SNR 20/10/0 dB) ─► WER, RTF ─► results/raw.csv ─► tabelas e figuras
  ├─► slides completos (OpenCV) ─► OCR ⏳ ─► slides.json ───────────────────────┤
  └─► transcrição ─► LLM (Ollama ou Claude) ─► resumo + capítulos ────────┴─► página (web/)
```

## Licença dos vídeos

| Vídeo | Origem | Licença | Situação |
|---|---|---|---|
| Módulo 1 (15 aulas) | a confirmar | a confirmar | **Pendente.** Não usar no projeto até anotar a licença (CLAUDE.md, regra 4) |

Os vídeos não entram no git, na imagem Docker nem no contexto de build (ver `.gitignore` e `.dockerignore`).

## Regras de transcrição do gabarito

`[PENDENTE]` (CLAUDE.md, pendência 8). A equipe define **antes** de transcrever: números (por extenso ou algarismos), hesitações ("é…", "né"), siglas, palavras cortadas. A normalização do código (`metrics.norm`) já ignora caixa e pontuação e mantém os acentos.

## Instalação

### Sem contêiner

Requer FFmpeg. Para o OCR, Tesseract com `por` (`sudo pacman -S tesseract tesseract-data-por` no Arch). Para a F9 local, [Ollama](https://ollama.com).

```bash
uv sync                  # Python 3.12 e dependências fixadas
uv sync --extra llm      # + SDK da Anthropic, só se for usar o Claude na F9
uv run pytest -q
```

### Em contêiner

Pré-requisitos: Docker (com o usuário no grupo `docker`) e, para a GPU, o NVIDIA Container Toolkit no host. Sem ele, o contêiner roda em CPU e o teste de GPU é pulado.

```bash
# Arch: o pacote costuma estar no AUR (confirme o nome antes)
yay -S nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verificação: `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` deve listar a RTX 3060.

```bash
docker build -t falai:latest .
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose run --rm pipeline        # pytest -q
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose run --rm pipeline bash   # shell no contêiner
```

## Uso rápido: um vídeo e a demo

```bash
uv run python src/pipeline.py data/raw/aula01.mp4 --model small
uv run python src/pipeline.py data/raw/aula01.mp4 --model small --summary ollama   # + resumo e tópicos
cd web && python -m http.server     # abrir http://localhost:8000
```

A página tem player com legenda, abas de tópicos, slides, transcrição e resumo, e busca na fala e nos slides. Clicar em qualquer item leva ao instante do vídeo. Sem LLM disponível, o resumo é pulado e o resto segue.

Para transcrever só uma mídia: `uv run python src/stt.py arquivo.mp4 --model small` (gera `.vtt`, `.srt`, `.txt` com a transcrição completa e `.json`).

## Roteiro do experimento

### 1. Dados (etapa 1)

Para cada vídeo `data/raw/aulaNN.mp4` (com licença anotada acima):

1. Escolha um trecho de 2 a 3 min **começando no início de uma frase** e registre em `data/gt/clips.csv` (`video,start_s,dur_s`; ver `clips.example.csv`).
2. Transcreva o trecho **à mão, do zero**, em `data/gt/aulaNN.stt.txt`. Não corrija a saída de um modelo (regra 3).
3. Opcional (I5): os termos técnicos dos trechos em `data/gt/terms.txt` (ver `terms.example.txt`).
4. Só para a F9: os tópicos reais em `data/gt/aulaNN.topics.txt` (ver `aula01.topics.example.txt`).

### 2. STT × ruído (RQ1)

```bash
uv run python src/run_experiments.py                      # tiny, base, small × limpo/20/10/0 dB × 3
uv run python src/run_experiments.py --models medium      # acrescenta um modelo
uv run python src/run_experiments.py --vocab              # + condição limpo_vocab (I5)
uv run python src/report.py                               # results/tables/ e results/figures/
```

As hipóteses ficam em `data/derived/<video>/stt/` para a análise de erros (regra 11). Execuções já registradas no `raw.csv` são puladas.

### 3. Resumo por LLM (F9, opcional)

```bash
ollama pull qwen2.5:7b                                     # uma vez (~4,7 GB)
uv run python src/run_experiments.py --summaries --models tiny base small
# Claude (pago; só com o aval do professor, pendência 3):
# uv sync --extra llm && export ANTHROPIC_API_KEY=...
# uv run python src/run_experiments.py --summaries --backend claude --confirm-paid

uv run python src/eval_summaries.py prepare
```

Envie `results/eval/blind/` aos avaliadores (**não** o `key.csv`). Junte as notas em `results/eval/scores.csv`, preencha `fidelity.csv` (afirmações do resumo e quantas têm respaldo) e revise as estimativas de `topics.csv`. Depois:

```bash
uv run python src/eval_summaries.py consolidate
uv run python src/report.py      # tabela4_resumos e fig4_wer_nota_resumo
```

## Dependências

Versões fixadas em `pyproject.toml` e travadas em `uv.lock`. Para atualizar uma biblioteca, edite a versão no `pyproject.toml` e rode `uv lock`. Não instale pacotes avulsos no contêiner. O SDK da Anthropic é opcional (`--extra llm`). O EasyOCR fica de fora até a etapa 3 (traz o PyTorch e deixa a imagem bem maior).

## Estrutura

```
Dockerfile, docker-compose.yml   imagem CUDA 12.4 + FFmpeg + Tesseract (por) e serviço de desenvolvimento
pyproject.toml, uv.lock          dependências fixadas
src/extract.py                   áudio, recorte, slides completos e keyframes (baseline)
src/stt.py                       faster-whisper, legendas (.vtt/.srt) e texto
src/degrade.py                   ruído e degradação de imagem
src/metrics.py                   WER, CER, precisão/recall, RTF, acerto de termos
src/summarize.py                 F9: resumo por LLM (Ollama ou Claude) e capítulos
src/pipeline.py                  um vídeo de ponta a ponta + pacote da demo
src/run_experiments.py           matriz de experimentos -> results/raw.csv
src/eval_summaries.py            F9: avaliação às cegas dos resumos
src/report.py                    tabelas e figuras a partir do CSV
web/index.html                   página da demo (web/data/ é gerado)
tests/                           testes de fumaça e unitários (sem rede, sem modelos)
data/raw/                        vídeos originais (fora do git)
data/gt/                         gabaritos humanos, trechos, tópicos e vocabulário
data/derived/                    áudio, ruído, frames e saídas intermediárias (fora do git)
results/                         raw.csv, tabelas, figuras e avaliação
artigo/                          esqueleto do artigo (modelo SBC pendente)
docs/decisoes.md                 histórico de decisões
```
