# FalAI: análise de vídeos de aula com IA

Pipeline de transcrição (STT), keyframes e OCR de slides, resumo por LLM e avaliação de qualidade e tempo, para a disciplina Redes Multimídia (EN05228, UFPA, Tema 9). Projeto acadêmico. O plano completo está em [CLAUDE.md](CLAUDE.md), o mapa do código em [guia.md](guia.md) e o histórico de decisões em [docs/decisoes.md](docs/decisoes.md).

**Pergunta de investigação:** qual é o trade-off entre qualidade (WER/CER) e tempo de processamento de modelos de STT e OCR em vídeos de aula com ruído? Secundária (F9): o erro do STT piora o resumo gerado por LLM?

**Estado:** STT, degradação, métricas, experimentos (RQ1), relatório, página da demo, F9 e F10 (o slide como contexto do STT, opcional) prontos e testados, com dados sintéticos e dublês dos modelos. Falta o Tesseract no host (para o OCR de verdade), a matriz de OCR (RQ2) e os dados reais (vídeos com licença e gabaritos humanos).

```
Vídeo ─► FFmpeg (16 kHz mono) ─► Whisper (STT) ─► .vtt / .txt ─────────┐
  │              └─ ruído (SNR 20/10/0 dB) ─► WER, RTF ─► results/raw.csv ─► tabelas e figuras
  ├─► slides completos (OpenCV, com painel) ─► OCR ─► slides.json ─► glossário ─► STT com contexto (F10) ┤
  └─► transcrição ─► LLM (Ollama ou Claude) ─► resumo + capítulos ────────┴─► página (web/)
```

## Licença dos vídeos

| Vídeo | Origem | Licença | Situação |
|---|---|---|---|
| Módulo 1 (15 aulas) | a confirmar | a confirmar | **Pendente.** Não usar no projeto até anotar a licença (CLAUDE.md, regra 4) |
| `economia do dia cb/` (15 vídeos de um curso comercial sobre o Sistema Financeiro Nacional, plataforma Voomp Play) | curso comercial | **sem licença** | **Só banco de testes local** (docs/decisoes.md, D20): fora do git, do Docker, do artigo e dos slides. Citar resultados deles exige autorização |

Os vídeos não entram no git, na imagem Docker nem no contexto de build (ver `.gitignore` e `.dockerignore`).

Gabarito e vídeo com licença são o gargalo da etapa 1; sem eles, a F10 só pode ser exercitada no banco de testes local (nenhum número dele vai para o artigo).

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

# Ou a GUI (FalAI Studio): vídeos, processamento, experimentos, resultados, gabarito, erros e A/B
uv run --extra gui python src/studio.py    # abrir http://127.0.0.1:8000
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

### 4. O slide como contexto (F10, opcional)

Plano e justificativa em [docs/plano_contexto_cruzado.md](docs/plano_contexto_cruzado.md); o passo a passo e a tabela de sintomas estão no [guia.md](guia.md), seção 3.3. **Pré-requisito:** o Tesseract com o português (`sudo pacman -S tesseract tesseract-data-por`, como o usuário).

```bash
uv run python src/context.py glossary data/raw/aula01.mp4          # slides (com painel) + OCR + glossário
uv run python src/context_stt.py data/raw/aula01.mp4 --glossary data/derived/aula01/glossario.json --mode L+F
uv run python src/pipeline.py data/raw/aula01.mp4 --context L+F    # a demo com o texto dos slides e o STT com contexto

# a matriz (RQ6 a RQ8): B0 sem contexto, G global, P placebo, L local, L+F com correção, O oráculo
uv run python src/run_experiments.py --context B0 G P L L+F O --models tiny base small
uv run python src/context_audit.py consolidate    # depois de auditar as correções, ouvindo, na tela Correções do Studio
uv run python src/calibration.py && uv run python src/report.py   # tabelas 6 e 7, figuras 5 a 7
```

Para a matriz, além do trecho e do gabarito da etapa 1, cada vídeo precisa do `glossario.json` e, para o oráculo, do `data/gt/terms.txt`. Com o extrator de slides em dúvida, `uv run python src/extract.py <vídeo> --trace 170,215` mostra a decisão a cada amostra e `src/eval_slides.py` mede contra um gabarito de tempo anotado à mão.

## Dependências

Versões fixadas em `pyproject.toml` e travadas em `uv.lock`. Para atualizar uma biblioteca, edite a versão no `pyproject.toml` e rode `uv lock`. Não instale pacotes avulsos no contêiner. O SDK da Anthropic é opcional (`--extra llm`). O EasyOCR fica de fora até a etapa 3 (traz o PyTorch e deixa a imagem bem maior).

## Estrutura

```
Dockerfile, docker-compose.yml   imagem CUDA 12.4 + FFmpeg + Tesseract (por) e serviço de desenvolvimento
pyproject.toml, uv.lock          dependências fixadas
src/extract.py                   áudio, recorte, slides completos (com painel) e keyframes (baseline)
src/stt.py                       faster-whisper (com hotwords), legendas (.vtt/.srt) e texto
src/ocr.py                       OCR dos slides (Tesseract; EasyOCR opcional)
src/context.py                   F10: glossário por slide e plano de intervalos
src/context_stt.py               F10: STT com o slide como contexto (B0, G, P, L, L+F, O)
src/context_fix.py               F10: correção de siglas e nomes com gate
src/context_audit.py             F10: auditoria das correções -> raw.csv
src/calibration.py               F10: calibração da confiança do Whisper
src/eval_slides.py               F10: extrator de slides × gabarito de tempo
src/degrade.py                   ruído e degradação de imagem
src/metrics.py                   WER, CER, precisão/recall, RTF, acerto de termos (inclusive com gabarito parcial)
src/summarize.py                 F9: resumo por LLM (Ollama ou Claude) e capítulos
src/pipeline.py                  um vídeo de ponta a ponta + pacote da demo
src/run_experiments.py           matriz de experimentos -> results/raw.csv
src/eval_summaries.py            F9: avaliação às cegas dos resumos
src/report.py                    tabelas e figuras a partir do CSV
web/index.html                   página da demo (web/data/ é gerado)
web/studio.html                  FalAI Studio (GUI de trabalho; src/studio.py)
tests/                           testes de fumaça e unitários (sem rede, sem modelos)
data/raw/                        vídeos originais (fora do git)
data/gt/                         gabaritos humanos, trechos, tópicos e vocabulário
data/derived/                    áudio, ruído, frames e saídas intermediárias (fora do git)
results/                         raw.csv, tabelas, figuras e avaliação
artigo/                          esqueleto do artigo (modelo SBC pendente)
docs/decisoes.md                 histórico de decisões
```
