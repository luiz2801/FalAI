# Decisões e mudanças de plano

Histórico das mudanças no planejamento do [CLAUDE.md](../CLAUDE.md), com o motivo. O estado atual fica no
CLAUDE.md e no [guia.md](../guia.md); aqui fica o porquê.

## 2026-11-11 — Junção com o plano paralelo (transcrição + relatório por LLM)

Havia um segundo plano para o mesmo trabalho, com código já testado: transcrição com faster-whisper,
legendas, relatório da aula por LLM e um experimento sobre o efeito do erro do STT no relatório. Os dois
foram juntados neste repositório. Decisões tomadas pelo usuário:

| # | Decisão | Antes | Agora | Status |
|---|---|---|---|---|
| D1 | Pergunta de investigação | Só STT/OCR × ruído | A mesma pergunta principal; **pergunta secundária na F9**: o erro do STT piora o resumo gerado por LLM? | `[DECIDIDO]` |
| D2 | LLM da F9 | Ollama local | **Dois backends** na mesma interface: Ollama (padrão) e Claude API (pago, só com `--confirm-paid`, custo entra como métrica). Uso da API continua dependendo do professor (pendência 3) | `[DECIDIDO]` + `[VALIDAR]` |
| D3 | Gabarito | Humano (regra 3) | Mantido. O comando que gerava rascunho de gabarito com `large-v3` foi **removido** | `[DECIDIDO]` |
| D4 | Convenções | `data/`, `results/raw.csv`, `.vtt`, identificadores em inglês | Mantidas. O código do outro plano foi reescrito nos módulos do guia; a interface Gradio deu lugar à página estática `web/` | `[DECIDIDO]` |

Escolhas de implementação que mudam números (regra 7) e ficam como `[PREMISSA]` até a equipe revisar:

- **Filtro de alucinações** (`stt.clean_segments`, ligado por padrão): remove segmentos que são frases
  conhecidas de alucinação do Whisper em silêncio ("Legendas pela comunidade Amara.org" etc.) e repetições
  em loop. Muda o WER. Para comparar, `run_experiments.py --no-clean` grava as linhas com o modelo
  rotulado `<modelo>-semfiltro`.
- **`condition_on_previous_text=False`** fixo em todos os modelos: reduz loops de repetição.
- **`word_timestamps=True`** fixo: necessário para quebrar a legenda em blocos legíveis. O alinhamento por
  palavra entra no tempo de inferência, igual para todos os modelos.
- **Condição `limpo_vocab`** (I5, opcional): o mesmo áudio limpo com `initial_prompt` montado a partir de
  `data/gt/terms.txt`. Métrica nova `acerto_termos`.
- **Linhas do `raw.csv`**: cada execução de STT grava `wer`, `cer` e, havendo termos, `acerto_termos`, todas
  com `tempo_s`, `duracao_s` e `rtf` (antes: uma linha `wer` e uma `rtf`). A carga do modelo vai numa linha
  `carga_modelo` à parte.
- **Chave do texto dos slides**: a saída usa `texto` (recomendação do guia, seção 5). O gabarito de slides
  ainda não existe; quando for criado, usar `texto` também.

Escolhas sem efeito nos números:

- O áudio é decodificado pelo FFmpeg (`extract.load_audio`) e entregue ao Whisper como array. O
  faster-whisper 1.2.1 é incompatível com o PyAV 19 (`TypeError: metadata_errors`).
- Keyframes por diferença de quadros com OpenCV, como no guia. O outro plano usava a detecção de cena do
  FFmpeg; ficou de fora.
