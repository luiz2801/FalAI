# FalAI: análise de vídeos de aula com IA

Pipeline de transcrição (STT), OCR de slides e avaliação de qualidade e tempo, para a disciplina Redes Multimídia (EN05228, UFPA, Tema 9). Projeto acadêmico. O plano completo está em [CLAUDE.md](CLAUDE.md) e o mapa do código em [guia.md](guia.md).

**Estado:** etapa 0 (ambiente). Ainda não há código de pipeline; os testes de fumaça validam as ferramentas.

## Licença dos vídeos

| Vídeo | Origem | Licença | Situação |
|---|---|---|---|
| Módulo 1 (15 aulas) | a confirmar | a confirmar | **Pendente.** Não usar no projeto até anotar a licença (CLAUDE.md, regra 4) |

Os vídeos não entram na imagem Docker nem no contexto de build (ver `.dockerignore`).

## Rodar em contêiner

### Pré-requisitos

- Docker (com o usuário no grupo `docker`).
- Para usar a GPU: NVIDIA Container Toolkit no host. Sem ele, o contêiner roda em CPU e o teste de GPU é pulado.

Instalação do NVIDIA Container Toolkit (comandos de sistema, execute você):

```bash
# Arch: o pacote costuma estar no AUR (confirme o nome antes)
yay -S nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verificação: `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` deve listar a RTX 3060.

### Build e testes

```bash
cd "/home/luiz/ufpa/2026.2/redes multimidia/tema9"

docker build -t falai:latest .

# Testes de fumaça (com docker compose, que já monta o código e a GPU)
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose run --rm pipeline
```

O `docker compose run` roda `pytest -q` por padrão. Para abrir um shell dentro do contêiner:

```bash
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose run --rm pipeline bash
```

Sem compose, para o build e os testes:

```bash
docker run --rm --gpus all falai:latest
```

## Rodar sem contêiner (opcional)

Precisa de Tesseract com `por` no host (`sudo pacman -S tesseract tesseract-data-por` no Arch).

```bash
uv sync
uv run pytest -q
```

## Dependências

Versões fixadas em `pyproject.toml` e travadas em `uv.lock`. Para atualizar uma biblioteca, edite a versão no `pyproject.toml` e rode `uv lock`. Não instale pacotes avulsos no contêiner.

O EasyOCR fica de fora até a etapa 3 (traz o PyTorch e deixa a imagem bem maior).

## Estrutura

```
Dockerfile              imagem: CUDA 12.4 + cuDNN 9, FFmpeg, Tesseract (por), Python 3.12 via uv
docker-compose.yml      serviço de desenvolvimento: código montado, GPU, dados no host
pyproject.toml, uv.lock dependências fixadas
src/                    pipeline (módulos entram nas etapas 2 e 3)
tests/test_smoke.py     testes de fumaça da etapa 0
data/raw/               vídeos originais (fora do git, ainda vazio)
data/gt/                gabaritos humanos
data/derived/           áudio, frames e saídas intermediárias (fora do git)
results/                raw.csv e tabelas geradas
```
