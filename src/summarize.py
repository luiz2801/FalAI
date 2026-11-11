"""F9: relatório estruturado da aula por LLM, a partir da transcrição com tempos.

Dois backends, mesma interface (CLAUDE.md, decisão 11):
  - "ollama" (padrão): LLM local, gratuito e offline. Requer o Ollama rodando
    (ollama serve) e o modelo baixado (ollama pull <modelo>).
  - "claude": API da Anthropic, paga. Só com confirmação (regra 12) e com o custo como métrica.
    Requer `uv sync --extra llm` e ANTHROPIC_API_KEY.

Uso:
    python src/summarize.py data/derived/aula01/aula01.small.segments.srt
    python src/summarize.py X.segments.srt --backend claude --model claude-opus-5-5
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from stt import fmt

DEFAULT_BACKEND = "ollama"
DEFAULT_MODELS = {"ollama": "qwen2.5:7b", "claude": "claude-opus-5-5"}
OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# US$ por milhão de tokens (entrada, saída). Haiku 5.5: preço para prompts <= 100K tokens.
CLAUDE_PRICES = {"claude-opus-5-5": (4.00, 20.00), "claude-sonnet-5-5": (2.00, 10.00),
                 "claude-haiku-5-5": (0.10, 0.50), "claude-opus-5": (5.00, 25.00),
                 "claude-opus-4-8": (5.00, 25.00)}
# Modelos com fallback no servidor (fallbacks: "default"); o Haiku não tem.
CLAUDE_FALLBACK = {"claude-opus-5-5", "claude-sonnet-5-5", "claude-opus-5", "claude-fable-5-1"}

SYSTEM = """Você recebe a transcrição (formato SRT) de uma aula universitária com um único professor.
Gere um relatório em Markdown com exatamente estas seções, nesta ordem:

## 1. Resumo
Um ou dois parágrafos.

## 2. Tópicos abordados
Um tópico por linha, em ordem cronológica, no formato exato:
- [hh:mm:ss] Título do tópico
O timestamp é o início do bloco SRT em que o professor começa o tópico (copie-o do SRT, sem os milissegundos).

## 3. Conceitos-chave e definições
## 4. Exemplos citados
## 5. Perguntas de revisão
## 6. Referências mencionadas

Use apenas informações presentes na transcrição. Se algo não foi mencionado, escreva "não mencionado".
A transcrição foi gerada automaticamente e pode conter erros de reconhecimento em termos técnicos;
corrija-os quando o contexto deixar claro qual era o termo.
Comece a resposta diretamente pelo título "# Relatório da aula" seguido das seções."""

SYSTEM_SLIDES = """

Junto com a transcrição você recebe imagens dos slides exibidos na aula, cada uma precedida do
instante em que apareceu. Use os slides para confirmar termos técnicos, fórmulas e títulos de
tópicos, sem inventar conteúdo que não esteja na transcrição ou nos slides."""


def has_claude_credentials() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
                or os.environ.get("ANTHROPIC_PROFILE") or (Path.home() / ".config" / "anthropic").is_dir())


def ollama_models() -> list[str] | None:
    """Modelos baixados no Ollama, ou None se o servidor não responde."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=2) as r:
            return [m["name"] for m in json.loads(r.read()).get("models", [])]
    except (urllib.error.URLError, OSError, ValueError):
        return None


def backend_status(backend: str, model: str | None = None) -> str | None:
    """None se o backend pode ser usado; senão, o motivo (com o comando para resolver)."""
    if backend == "claude":
        return None if has_claude_credentials() else "sem credencial da API (ANTHROPIC_API_KEY)"
    model = model or DEFAULT_MODELS["ollama"]
    modelos = ollama_models()
    if modelos is None:
        return f"Ollama não responde em {OLLAMA_URL} (rode `ollama serve`)"
    nomes = set(modelos) | {m.removesuffix(":latest") for m in modelos}
    if model not in nomes:
        return f"modelo {model} não está no Ollama (rode `ollama pull {model}`)"
    return None


def backend_available(backend: str, model: str | None = None) -> bool:
    return backend_status(backend, model) is None


def claude_cost(model: str, tokens_in: int, tokens_out: int) -> float | None:
    if model not in CLAUDE_PRICES:
        return None
    pi, po = CLAUDE_PRICES[model]
    return round((tokens_in * pi + tokens_out * po) / 1e6, 5)


# ------------------------------------------------------------------ backends
def _summarize_claude(srt: str, model: str, effort: str, slides) -> tuple[str, dict]:
    import anthropic

    conteudo: list[dict] = []
    for t, caminho in slides or []:
        conteudo.append({"type": "text", "text": f"Slide exibido em [{fmt(t)[:8]}]:"})
        conteudo.append({"type": "image", "source": {
            "type": "base64", "media_type": "image/png",
            "data": base64.standard_b64encode(Path(caminho).read_bytes()).decode()}})
    conteudo.append({"type": "text", "text": f"<transcricao>\n{srt}\n</transcricao>"})
    client = anthropic.Anthropic()
    params = dict(model=model, max_tokens=32000, output_config={"effort": effort},
                  system=SYSTEM + (SYSTEM_SLIDES if slides else ""),
                  messages=[{"role": "user", "content": conteudo}])
    # streaming: entrada longa (~15 mil tokens por hora de aula) sem timeout HTTP
    if model in CLAUDE_FALLBACK:
        with client.beta.messages.stream(betas=["server-side-fallback-2026-07-01"], fallbacks="default",
                                         **params) as stream:
            resp = stream.get_final_message()
    else:
        with client.messages.stream(**params) as stream:
            resp = stream.get_final_message()
    if resp.stop_reason == "refusal":
        cat = getattr(resp.stop_details, "category", None) if resp.stop_details else None
        raise RuntimeError(f"o modelo recusou a requisição (categoria: {cat})")
    texto = "".join(b.text for b in resp.content if b.type == "text")
    u = resp.usage
    tin = (u.input_tokens or 0) + (getattr(u, "cache_read_input_tokens", 0) or 0) \
        + (getattr(u, "cache_creation_input_tokens", 0) or 0)
    return texto, {"model_served": resp.model, "stop_reason": resp.stop_reason, "tokens_in": tin,
                   "tokens_out": u.output_tokens, "cost_usd": claude_cost(resp.model, tin, u.output_tokens)}


def _summarize_ollama(srt: str, model: str, slides) -> tuple[str, dict]:
    if slides:
        raise ValueError("slides (multimodal) só estão implementados no backend claude")
    corpo = json.dumps({
        "model": model, "stream": False,
        "options": {"num_ctx": 32768, "temperature": 0.2, "seed": 0},
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": f"<transcricao>\n{srt}\n</transcricao>"}],
    }).encode()
    req = urllib.request.Request(f"{OLLAMA_URL}/api/chat", data=corpo, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            resp = json.loads(r.read())
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama indisponível em {OLLAMA_URL} ({e}). Rode `ollama serve` e "
                           f"`ollama pull {model}`.") from e
    return resp["message"]["content"], {"model_served": resp.get("model", model),
                                        "stop_reason": resp.get("done_reason"),
                                        "tokens_in": resp.get("prompt_eval_count"),
                                        "tokens_out": resp.get("eval_count"), "cost_usd": 0.0}


def summarize(srt: str, backend: str = DEFAULT_BACKEND, model: str | None = None, effort: str = "medium",
              slides: list[tuple[float, Path]] | None = None) -> tuple[str, dict]:
    """Gera o relatório. Devolve (markdown, meta com tokens, custo e tempo)."""
    if backend not in DEFAULT_MODELS:
        raise ValueError(f"backend desconhecido: {backend!r} (use ollama ou claude)")
    model = model or DEFAULT_MODELS[backend]
    t0 = time.perf_counter()
    if backend == "claude":
        texto, meta = _summarize_claude(srt, model, effort, slides)
    else:
        texto, meta = _summarize_ollama(srt, model, slides)
    meta.update(backend=backend, model=model, time_s=round(time.perf_counter() - t0, 2),
                n_slides=len(slides or []))
    if meta.get("stop_reason") in ("max_tokens", "length"):
        meta["warning"] = "relatório truncado"
    return texto.strip() + "\n", meta


# ------------------------------------------------------------------ tópicos e capítulos
_RE_TOPIC = re.compile(r"^\s*[-*]\s*\[(\d{1,2}:\d{2}(?::\d{2})?)\]\s*(.+?)\s*$")


def to_seconds(ts: str) -> float:
    seg = 0.0
    for p in ts.strip().replace(",", ".").split(":"):
        seg = seg * 60 + float(p)
    return seg


def extract_topics(report: str) -> list[tuple[float, str]]:
    """Lê a seção 'Tópicos abordados' e devolve [(segundos, título)]."""
    topicos, dentro = [], False
    for linha in report.splitlines():
        if linha.startswith("#"):
            dentro = "tópicos" in linha.lower() or "topicos" in linha.lower()
            continue
        if dentro and (m := _RE_TOPIC.match(linha)):
            topicos.append((to_seconds(m.group(1)), m.group(2).strip("*_ ")))
    return sorted(topicos)


def write_chapters_vtt(topics: list[tuple[float, str]], path: str | Path, duration_s: float | None = None) -> Path:
    """Capítulos WebVTT (I6): cada capítulo vai até o início do próximo."""
    blocos = ["WEBVTT"]
    for i, (ini, titulo) in enumerate(topics):
        fim = topics[i + 1][0] if i + 1 < len(topics) else (duration_s if duration_s and duration_s > ini else ini + 60)
        blocos.append(f"{fmt(ini)} --> {fmt(fim)}\n{titulo}")
    Path(path).write_text("\n\n".join(blocos) + "\n", encoding="utf-8")
    return Path(path)


def export_pdf(md: str | Path) -> Path:
    """Relatório em PDF via pandoc (xelatex). Comando de sistema: pacman -S pandoc texlive."""
    import subprocess

    md = Path(md)
    destino = md.with_suffix(".pdf")
    subprocess.run(["pandoc", str(md), "-o", str(destino), "--pdf-engine=xelatex",
                    "-V", "geometry:margin=2.5cm", "-V", "lang=pt-BR"], check=True)
    return destino


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("srt", help="transcrição .segments.srt (um bloco por segmento)")
    ap.add_argument("--backend", default=DEFAULT_BACKEND, choices=["ollama", "claude"])
    ap.add_argument("--model", help=f"padrão: {DEFAULT_MODELS}")
    ap.add_argument("--effort", default="medium", help="só no claude")
    ap.add_argument("--slides", help="pasta com frames.json do extract.keyframes (só claude)")
    ap.add_argument("--pdf", action="store_true")
    a = ap.parse_args()
    slides = None
    if a.slides:
        frames = json.loads((Path(a.slides) / "frames.json").read_text(encoding="utf-8"))
        slides = [(f["t"], Path(a.slides) / f["file"]) for f in frames]
    srt = Path(a.srt)
    texto, meta = summarize(srt.read_text(encoding="utf-8"), a.backend, a.model, a.effort, slides)
    saida = srt.with_name(srt.name.replace(".segments.srt", "") + f".summary.{meta['backend']}.md")
    saida.write_text(texto, encoding="utf-8")
    saida.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    write_chapters_vtt(extract_topics(texto), saida.with_suffix(".chapters.vtt"))
    print(f"{saida} | {meta['model_served']} | {meta['tokens_in']}+{meta['tokens_out']} tokens | "
          f"US$ {meta['cost_usd']} | {meta['time_s']} s")
    if a.pdf:
        print(export_pdf(saida))


if __name__ == "__main__":
    main()
