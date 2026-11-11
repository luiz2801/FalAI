import io
import json

import pytest

import summarize

RELATORIO = """# Relatório da aula

## 1. Resumo
Texto.

## 2. Tópicos abordados
- [00:00:05] Introdução
- [00:01:30] Capital de giro
- [01:05] Fórmula
- não é tópico

## 3. Conceitos-chave e definições
- [00:09:00] isto não está na seção de tópicos
"""


def test_extract_topics():
    assert summarize.extract_topics(RELATORIO) == [(5.0, "Introdução"), (65.0, "Fórmula"), (90.0, "Capital de giro")]


def test_chapters_vtt(tmp_path):
    p = summarize.write_chapters_vtt([(0.0, "A"), (30.0, "B")], tmp_path / "c.vtt", duration_s=50)
    assert p.read_text() == "WEBVTT\n\n00:00:00.000 --> 00:00:30.000\nA\n\n00:00:30.000 --> 00:00:50.000\nB\n"


def test_backend_desconhecido():
    with pytest.raises(ValueError):
        summarize.summarize("x", backend="gpt")


def test_ollama_requisicao(monkeypatch):
    enviado = {}

    class Resp(io.BytesIO):
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        enviado.update(json.loads(req.data))
        return Resp(json.dumps({"model": "m", "message": {"content": RELATORIO}, "done_reason": "stop",
                                "prompt_eval_count": 100, "eval_count": 50}).encode())

    monkeypatch.setattr(summarize.urllib.request, "urlopen", fake_urlopen)
    texto, meta = summarize.summarize("1\n00:00:00,000 --> 00:00:01,000\noi\n", "ollama", "m")
    assert enviado["model"] == "m" and enviado["stream"] is False
    assert enviado["messages"][0]["role"] == "system" and "<transcricao>" in enviado["messages"][1]["content"]
    assert meta["cost_usd"] == 0.0 and meta["tokens_in"] == 100 and texto.startswith("# Relatório")


def test_claude_requisicao_e_fallback(monkeypatch):
    """Requisição real do SDK com transporte HTTP simulado (sem rede, sem custo)."""
    anthropic = pytest.importorskip("anthropic")
    httpx = pytest.importorskip("httpx2")
    capturadas = []

    def sse(model):
        ev = [("message_start", {"type": "message_start", "message": {
                  "id": "m", "type": "message", "role": "assistant", "model": model, "content": [],
                  "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 1000, "output_tokens": 1}}}),
              ("content_block_start", {"type": "content_block_start", "index": 0,
                                       "content_block": {"type": "text", "text": ""}}),
              ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                       "delta": {"type": "text_delta", "text": RELATORIO}}),
              ("content_block_stop", {"type": "content_block_stop", "index": 0}),
              ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn",
                                 "stop_sequence": None}, "usage": {"output_tokens": 500}}),
              ("message_stop", {"type": "message_stop"})]
        return "".join(f"event: {e}\ndata: {json.dumps(d)}\n\n" for e, d in ev).encode()

    def handler(req):
        corpo = json.loads(req.content)
        capturadas.append({"url": str(req.url), "beta": req.headers.get("anthropic-beta"), "corpo": corpo})
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse(corpo["model"]))

    original = anthropic.Anthropic
    monkeypatch.setattr(anthropic, "Anthropic", lambda *a, **k: original(
        api_key="teste", http_client=httpx.Client(transport=httpx.MockTransport(handler))))

    _, meta = summarize.summarize("srt", "claude", "claude-opus-5-5")
    c = capturadas[-1]
    assert c["beta"] == "server-side-fallback-2026-07-01" and c["corpo"]["fallbacks"] == "default"
    assert c["corpo"]["stream"] is True and c["corpo"]["output_config"] == {"effort": "medium"}
    assert meta["cost_usd"] == pytest.approx((1000 * 4 + 500 * 20) / 1e6)

    summarize.summarize("srt", "claude", "claude-haiku-5-5")
    assert capturadas[-1]["beta"] is None and "fallbacks" not in capturadas[-1]["corpo"]
