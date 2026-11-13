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

## 2026-11-12 — Slides completos: `extract.slides()` ao lado do baseline `keyframes()`

**Problema.** O `keyframes()` grava o primeiro quadro após uma mudança. Quando o professor revela os
bullets aos poucos, isso guarda o slide incompleto, e o OCR desse quadro teria recall baixo por culpa da
extração, não do motor (contaminaria a RQ2). Num teste com dois prints reais de aula (slide com 1 bullet e
depois com a lista completa), o `keyframes` gravou só o quadro parcial e não percebeu a revelação.

**Decisão do usuário.** Vídeos são apresentações de slides; às vezes o professor aparece, às vezes num
canto. Criar `extract.slides()` mantendo `keyframes()` como baseline. PDF "apostila" e vínculo
slide ↔ transcrição ficam para depois, mas são requisito importante.

| # | Decisão | Status |
|---|---|---|
| D5 | Detecção por máscara de tinta com duas medidas (`mantido`, `acrescido`): bullet novo só acrescenta tinta, slide novo remove tinta antiga. Candidato = quadro mais completo do episódio, gravado quando o slide troca ("buffer reverso") | `[PREMISSA]` |
| D6 | `frames.json` ganha `t_complete`, `t_end`, `n_stages`, `visits`. **`t` continua sendo o início do slide**, então a página da F5 não muda | `[DECIDIDO]` (usuário autorizou) |
| D7 | `pipeline.py` usa `slides` por padrão; `--frames-mode diff` volta ao baseline. `step_s` padrão passa a 0,5 s no modo `slides` | `[PREMISSA]` |
| D8 | Limiares (`keep_thr` 0,9, `grow_eps` 0,01, `min_stable` 2, `min_slide_s` 2 s) sem calibração em vídeo real | `[PREMISSA]` calibrar com os vídeos da etapa 1; ablação F7 |

**Avaliação prevista (sem números inventados).** Comparar `slides` × `keyframes` (baseline) contra um
gabarito de tempo anotado à mão (início e "slide completo" de cada slide): recall de slides, duplicatas,
quadros parciais e recall do OCR sobre o quadro final. Esse gabarito de tempo ainda não existe.

**Teste em vídeo real (2026-11-12).** Aula do Coursera/Wharton (960×540, 7,6 min, professor em tela cheia
alternando com slides). A primeira versão gravou o professor como "slide" e dividiu um slide em versão
parcial e completa. Correções: filtro `max_ink` = 0,30 (D9) e fusão parcial/completo com título idêntico a
≥ 95% (D10). Resultado, conferido a olho nas imagens: 6 slides de conteúdo, todos completos (um deles com 7
estágios), mais os 2 cartões de logo; nenhum quadro do professor. Isso é inspeção visual de um único vídeo,
não uma métrica; o gabarito de tempo continua pendente.

| # | Decisão | Status |
|---|---|---|
| D9 | `max_ink` = 0,30: quadros com muita tinta (imagem natural) não são slide. Medido: slides 3–14%, professor 42–59% | `[PREMISSA]` |
| D10 | Título "igual" só com ≥ 95% de coincidência (`TITLE_KEPT`); parcial e completo do mesmo slide viram um arquivo, com o quadro mais completo | `[PREMISSA]` |

**Lote de 12 aulas (2026-11-12).** `extract.slides` rodou nas 12 aulas de "Financial Acumen for Non-Financial
Managers" (1,6 a 3,7 s por vídeo de 6 a 14 min). Conferência visual em 3 vídeos: em 2 deles o professor ainda
passava como slide (camisa azul em fundo escuro: só 29% de tinta). Correção D11.

| # | Decisão | Status |
|---|---|---|
| D11 | `min_flat` = 0,35: exige fundo liso (fração de pixels a ±3 níveis da mediana). Professor 7–18%, slides e cartões ≥ 48% nos 62 quadros gravados | `[PREMISSA]` |

Ressalva: as 12 aulas têm o mesmo visual e o mesmo professor; ajustar limiares nelas não prova que
generalizam. O gabarito de tempo continua pendente.

**FalAI Studio (2026-11-12).** GUI de trabalho (guia.md, 4.10), pedida pelo usuário: FastAPI + HTML/JS, sem build,
reaproveitando o visualizador. Não muda nenhum número: lê o `raw.csv` e roda os mesmos scripts do CLI.

| # | Decisão | Status |
|---|---|---|
| D12 | Categorias da análise de erros (regra 11), gravadas em `results/erros.csv`: `nome_proprio`, `sigla`, `numero`, `termo_tecnico`, `sotaque`, `ruido`, `hesitacao`, `normalizacao`, `alucinacao`, `outro`. As cinco do CLAUDE.md mais termo técnico (liga com a I5), hesitação (fala espontânea), normalização (grafia que o `norm` não iguala, como "u f p a" × "ufpa"), alucinação (texto sem fala correspondente) e outro | `[PREMISSA]` revisar com a equipe antes de classificar |
| D13 | Classifica-se o **trecho** do alinhamento (sequência contígua de substituições, remoções ou inserções), não a palavra: um trecho = uma marcação. A tabela 5 conta trechos, não é uma taxa | `[PREMISSA]` |

## 2026-11-12 — F10: o slide como contexto (docs/plano_contexto_cruzado.md)

Pedido do usuário ("implementa"), depois de o modo ao vivo ser descartado. O plano está em
[plano_contexto_cruzado.md](plano_contexto_cruzado.md); a entrega cobre as fases 0 a 6 em código e testes. **Nada do que
está aqui é resultado do artigo**: não há vídeo com licença nem gabarito humano ainda (etapa 1), e os vídeos do curso
usados nos diagnósticos são só banco de testes local (D20).

| # | Decisão | Status |
|---|---|---|
| D14 | **Slides em layout de painel.** O `SlideTracker` mede a mediana de cinza e o fundo liso **dentro da `roi`** (antes, no quadro inteiro, e um painel de ~50% do quadro fazia a mediana saltar e derrubava o filtro `min_flat`). `extract.slides` ganha `panel` (`"auto"` padrão: `find_panel` acha o painel; `"off"` volta ao quadro inteiro; `roi=` manual tem prioridade) e grava `panel.json`. Sem `roi` nem `ignore` o resultado é idêntico ao antigo (testado). CLI: `--panel`, `--trace t0,t1` | `[PREMISSA]` muda o comportamento do extrator (C3 do plano). Nos 15 vídeos de teste: de 1 para 22 slides, conferidos a olho (não é métrica; o gabarito de tempo falta) |
| D15 | **O contexto entra como `hotwords`.** Pelo código do faster-whisper 1.2.1, com `condition_on_previous_text=False` o `initial_prompt` só chega à 1ª janela de ~30 s (linhas 1143 a 1149, 1187 e 1373 a 1383 de `transcribe.py`), e o `hotwords` vai a toda janela (linha 1542). `stt.transcribe` ganha `hotwords`. **A condição `limpo_vocab` (I5) não foi alterada**: ela provavelmente subestima o efeito do vocabulário. `run_experiments.py --context ... --via initial_prompt` permite medir a diferença | `[PREMISSA]`; **pendente C2** (refazer a I5?). Falta o teste empírico do `initial_prompt` por janela |
| D16 | **Correção com gate** (`context_fix`): R1 (p < `tau` 0,5 e semelhança fonética ≥ `s_min` 0,85 com sigla ou nome do slide visível), R2 (sigla a 1 edição: maiúsculas sem gate; só consoantes com p < `tau`), R3 (letras soletradas). Só `sigla` e `nome` são alvo (palavra comum do slide não); um span de várias palavras só vale se cada palavra faz falta; a nasal final é opcional na semelhança. Cada troca vai para `correcoes.jsonl`, com a frase antes e depois, para auditoria | `[PREMISSA]`; limiares sem calibração (ablação prevista) |
| D17 | `stt.resolve_device("auto")` só escolhe CUDA se `libcublas.so.12` carrega; senão cai para CPU/int8 com um aviso. `hardware.json` ganha `device_auto` (o que o `auto` de fato usa) | `[PREMISSA]` (C9 do plano; reversível: `--device cuda` força a GPU) |
| D18 | **Contratos novos.** `raw.csv`: condição `<ruído>@<modo>` (`limpo@L`, `snr10@L+F`; modos B0, G, P, L, L+F, O), métricas `acerto_termos_slide`, `acerto_termos_fora`, `falsos_positivos`, `taxa_alteracao`, `precisao_correcoes` e a linha `tempo_ctx`. Arquivos: `panel.json`, `ocr.json`, `glossario.json`, `correcoes.jsonl`, `correcoes_auditoria.csv`, `calibracao.csv`, `<v>.slides_t.json`, `<v>.termos.tsv` (guia, 5.12 a 5.17). As linhas antigas continuam válidas | `[PREMISSA]` (C5 e C6) |
| D19 | **Desenho da avaliação.** Controles: placebo (P: termos de outro vídeo, mesmo tamanho e tipos, fora dos termos do trecho), oráculo (O: o `terms.txt` humano), o B0 e o `acerto_termos_fora` (termos que o slide não tem). `acerto_termos` usa só o `terms.txt` **humano**, nunca termos do OCR (circularidade). A precisão das correções vem de auditoria humana ouvindo o áudio (tela Correções do Studio, `context_audit.py consolidate`). Camada A (banco de testes local): auditoria e gabarito parcial de termos; camada B (dataset oficial): a matriz completa | `[PREMISSA]` |
| D20 | Os 15 vídeos de `economia do dia cb/` (curso comercial, sem licença) são **só banco de testes local**: ficam fora do git, do Docker, do artigo e dos slides (regra 4). A pasta entrou no `.gitignore` e no `.dockerignore` | `[DECIDIDO]` pelo usuário ("usa como testes"); `[VALIDAR]` autorização para citar resultados |

**Limites conhecidos do `find_panel` e do `slides` com painel** (conferidos nos 15 vídeos de teste): (a) painel de texto muito denso, ou com a câmera sobrepondo um canto, funciona, mas a miniatura que muda de tamanho no meio do slide (`VP(9)`, aos 66 s) o divide em dois episódios; (b) o painel vazio no fim de um vídeo (`VP(7)`, 3 s) sai como um slide sem texto; (c) cartão de título num cenário virtual (`VP(10)`) é pego pela metade, e os cartões pequenos (`VP(6)`, `VP(11)`) não geram painel; (d) gravação de site com rolagem e legenda queimada (`VP(4)`, `VP(5)`) não gera slides (fora do escopo); (e) o título que fica numa fita sobre a borda do painel entra na caixa do OCR, mas não no rastreio.

### Observações do teste de integração com áudio real (não são resultados)

Um trecho de 3 min (95 a 275 s) do vídeo `VP`, em que o slide "CMN" mostra "BACEN" no texto, com o `medium` em CPU/int8 e o texto do slide digitado à mão (o Tesseract não está instalado). **Uma execução, um trecho escolhido por ter siglas, áudio não ouvido**: serve para ver se a fiação funciona e o que o código faz de errado, não para medir ganho.

- **Os erros mudam com o recorte do áudio.** A primeira sonda (trecho cortado pelo FFmpeg para WAV) leu "Bassim", "cml" e "CMR"; a segunda (o mesmo trecho cortado do array em memória) leu "bassé" e nenhum dos outros dois. Diferenças de poucas palavras entre condições não querem dizer nada sozinhas: daí o placebo e o bootstrap sobre ocorrências de termos.
- No B0, "bassé" foi o único erro de sigla. G e L (o trecho tem um só slide, então os dois recebem os mesmos termos) trocaram por "BACEN". O nome fora do slide ("Pimec"/"Pimek") continuou inconsistente em todos os modos ("PMEC", "Pemec" com contexto): o contexto não ajuda o que não está na tela.
- O **placebo mudou mais texto (8,4%) que o contexto real (6,2%)** e introduziu um "CMR": qualquer prompt mexe na saída, e só o real mexe na direção certa. É o motivo do controle.
- Os tempos (RTF 0,29 sem contexto e 0,58 a 0,59 com ele) **não são comparáveis**: a máquina tinha outras tarefas rodando. O ΔRTF só vale medido com a máquina livre.
- A **correção** aplicada ao B0 real fez uma troca, `Bassé → BACEN` (p = 0,42, semelhança 1,0). Antes de o `context_fix` ser corrigido, ele fazia outra troca, errada, `a composição → Composição` (apagava o artigo), e deixava passar o "bassé" (a nasal sem consoante). Os dois defeitos estão cobertos por testes (D16).
