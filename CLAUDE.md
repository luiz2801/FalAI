# Tema 9 — IA aplicada a sistemas multimídia: análise de vídeos de aula

> Documento-mestre do projeto. Leia inteiro antes de escrever código ou alterar o planejamento.
> Disciplina **Redes Multimídia (EN05228)**, UFPA, prof. Raimundo Viégas Jr. Apresentações em dezembro de 2026. Projeto acadêmico, não é produto.

**Documentos relacionados**

- [guia.md](guia.md): mapa do código, contratos de arquivo, parâmetros e armadilhas. Leia a seção do módulo antes de mexer nele.
- [Planejamento da equipe](../Tema09_IA_Sistemas_Multimidia_Planejamento.md): requisitos, cronograma, divisão de trabalho e riscos.
- [Guia de implementação do MVP](../Tema09_MVP_Guia_de_Implementacao.md): passo a passo das etapas 0 a 6, com trechos de código (não executados ainda).
- [Enunciado da disciplina](../Seminarios_de_Sistemas_e_Redes_Multimidia_2026_Atualizado.pdf): fonte oficial dos requisitos (Tema 9 é o item 09; regras gerais 10 a 17).

---

## 0. Como usar este documento

**Convenções**

- `[DECIDIDO]`: a equipe ou o usuário definiu. Não mude sem perguntar.
- `[PREMISSA]`: padrão do planejamento, adotado porque ainda não foi contestado. Pode ser revisto; registre a mudança em `docs/decisoes.md`.
- `[VALIDAR]`: depende de confirmação do professor ou de fonte externa (licença, versão, URL). Confirme antes de usar.
- `[PENDENTE]`: ainda não há decisão ou material (ex.: modelo SBC, nomes da equipe).
- `[PLACEHOLDER]`: número ilustrativo. Não é resultado de medição.

**Regras para o agente**

1. **Trabalhe por etapa** (seção 10). Cada etapa tem critério de pronto. Não avance com testes falhando.
2. **Nunca invente resultado.** WER, CER, RTF, precisão, recall e MRR vêm só de `results/raw.csv`, gerado por script. Não digite número no artigo nem nos slides.
3. **Gabarito é humano.** Não gere a transcrição de referência com um modelo e não use a saída de um modelo como gabarito para depois corrigi-la: isso enviesa o WER a favor desse modelo. A única exceção é o texto extraído do PDF/PPTX original da aula (`pdftotext`), com a origem registrada.
4. **Dados e direitos.** Não baixe nem use vídeo sem licença anotada (N4). Se gravar pessoas, tenha autorização de imagem e voz. Não copie para o projeto dados pessoais de alunos (a lista de alunos e as notas ficam na pasta da disciplina, fora do projeto).
5. **Medição de tempo** segue a seção 6 (aquecimento, consumir o gerador, cronometrar só a inferência, 3 repetições, mesmos parâmetros em todos os modelos).
6. **Reprodutibilidade.** Seed fixa, versões de bibliotecas fixadas no `pyproject.toml` e a mesma configuração de hardware em todos os experimentos. O hardware vai no artigo.
7. **Pergunte antes** quando a escolha mudar um número, um arquivo de saída, a pergunta de pesquisa ou o escopo (normalização do texto, limiar de keyframe, modelo escolhido, formato de um contrato). Quando a escolha não mudar a saída (organização interna, refatoração com testes idênticos), decida e diga no resumo o que foi feito.
8. **Não instale pacotes de sistema** (`pacman`, `apt`, `sudo`). Diga ao usuário o comando e espere. Pacotes Python entram pelo `uv`.
9. **Idioma.** Texto, documentação e artigo em PT-BR, com acentos. Identificadores de código em inglês, como nos exemplos do guia de implementação. `[PREMISSA]`
10. **Atualize o `guia.md` na mesma entrega** de qualquer mudança de módulo, contrato, parâmetro, formato de arquivo ou comando.
11. **Resultado de IA não substitui análise.** Ao reportar métricas, leia uma amostra dos erros e classifique-os (nomes próprios, siglas, números, sotaque, ruído). O enunciado exige isso (regra 15).
12. **Modelos locais por padrão.** Não chame API paga sem confirmação (seção 3, decisão 11).
13. **Economia de contexto.** Para uma tarefa pontual, leia só a seção do `guia.md` do módulo em foco, não o documento inteiro.

---

## 1. Objetivo e escopo

**Problema.** Vídeos de aula e palestras são difíceis de pesquisar e de acessar: não têm legenda, o conteúdo dos slides existe só como imagem e nada é indexado.

**Solução.** Um pipeline de IA sobre um vídeo de aula:

1. transcreve o áudio em PT-BR e gera legenda `.vtt` (F1);
2. extrai keyframes dos slides e faz OCR (F2);
3. compara modelos e motores e mede erro, tempo e robustez (F3 e F4);
4. entrega uma página com vídeo, legenda e painel de slides clicável (F5).

**Pergunta de investigação** `[PREMISSA]`, rascunho a fechar até ~15/11:

> Qual é o trade-off entre qualidade (WER/CER) e tempo de processamento de modelos de STT e OCR em vídeos de aula com ruído?

Pergunta secundária, só se a F8 entrar: quanto o processamento de IA afeta a entrega em streaming?

**Título provisório** `[PENDENTE]` aprovação do professor: *Análise inteligente de vídeos de aula: transcrição, OCR e indexação multimodal com avaliação de qualidade e custo computacional.*

**Escopo**

- **Obrigatório (MVP):** F1 a F5.
- **Desejável:** F6 (busca com Precision@k e MRR) e F7 (ablação da taxa de keyframes).
- **Opcional, cortar primeiro se faltar tempo:** F8 (HLS com e sem IA em paralelo) e F9 (resumo por LLM).
- **Fora do escopo:** otimização de rede e transporte (é o Tema 5), treino ou ajuste fino de modelos, uso multiusuário ou em produção.

**Fronteira análise × transporte.** O enunciado pede que o trabalho diferencie análise/produção de mídia do transporte pela rede. O projeto separa dois planos (seção 8 e `guia.md`, seção 2): o plano de conteúdo (IA, offline) e o plano de entrega (HLS, medido separadamente). Essa fronteira aparece na introdução do artigo e no diagrama de arquitetura.

---

## 2. Requisitos do enunciado

Pesos: implementação 5,0; artigo técnico 3,0; defesa 2,0.

| ID | Requisito | Como o projeto atende |
|---|---|---|
| I1 | Protótipo, aplicação, experimento ou benchmark funcional | Pipeline e página da F5, com demo ao vivo |
| I2 | Pelo menos uma evidência experimental ou métrica | `results/raw.csv` e tabelas/figuras do `report.py` |
| I3 | Pergunta técnica explícita, no artigo e no slide 1 | Seção 1 deste documento |
| I4 | O que avaliar, como e com quais métricas | Seção 4 deste documento; seção de metodologia do artigo |
| I5 | Relacionar teoria, implementação e resultados | Discussão cita a teoria que explica cada resultado |
| I6 | Arquitetura e decisões técnicas, mesmo com bibliotecas prontas | Diagrama em dois planos; justificativa de cada tecnologia |
| I7 | Viável em tempo e recursos | Escopo MVP; modelos pequenos; Colab se a GPU não der conta |
| Ar1, Ar2 | Artigo de 5 a 8 páginas, SBC, PDF, enviado com 48 h de antecedência | `artigo/` (modelo SBC `[PENDENTE]`) |
| Ar3, Ar4 | Nove seções; limitações e trabalhos futuros | Seção 9 deste documento |
| Ap1 a Ap6 | 20 min de apresentação, 10 min de arguição, qualquer integrante pode ser arguido | Seção 9 |
| Regra 17 | Todos conhecem fundamentação, arquitetura, implementação e resultados | Ensaio em que cada integrante explica todas as partes |

---

## 3. Decisões e premissas

| # | Tema | Decisão | Status |
|---|---|---|---|
| 1 | Tema | Tema 9, IA aplicada a sistemas multimídia | `[VALIDAR]` professor: o tema está livre? |
| 2 | Pergunta | Trade-off qualidade × tempo de STT e OCR com ruído (seção 1) | `[PREMISSA]` fechar até ~15/11 |
| 3 | Escopo | F1 a F5 obrigatórios; F6 e F7 desejáveis; F8 e F9 opcionais | `[PREMISSA]` planejamento, seção 4 |
| 4 | Python | 3.12 via `uv`. O Python 3.14 da máquina é novo demais para pacotes de ML | `[PREMISSA]` guia, §1.2 |
| 5 | STT | `faster-whisper`, comparando `tiny`, `base`, `small` (e `medium` se a GPU permitir). `language="pt"`, `beam_size=5`, `vad_filter=True` fixos | `[PREMISSA]` |
| 6 | OCR | Tesseract (`lang="por"`, testando `--psm 6` e `--psm 3`) e EasyOCR. PaddleOCR fica como alternativa | `[PREMISSA]` |
| 7 | Busca | Substring no F5. Índice sobre transcrição e OCR no F6. CLIP + FAISS só se sobrar tempo | `[PREMISSA]` |
| 8 | Execução | GPU RTX 3060 12 GB. Fallback `device="cpu"`, `compute_type="int8"`. A mesma escolha em todos os experimentos | `[PREMISSA]` guia, §1.3 |
| 9 | Ruído | Ruído branco aditivo, SNR 20, 10 e 0 dB, seed 0. Ruído real de sala como extra, se houver tempo. A simplificação vai como limitação | `[PREMISSA]` |
| 10 | Dataset | 6 vídeos de 5 a 15 min, e trechos de 2 a 3 min com gabarito. Vídeos de ~45 min só para tempo e demo | `[PREMISSA]` |
| 11 | Modelos | Locais. Sem API paga. Se API for usada, o custo entra como métrica | `[VALIDAR]` professor, pergunta 3 |
| 12 | Vídeos | Gravação própria ou licença livre, com licença anotada por vídeo | `[DECIDIDO]` (N4) + `[VALIDAR]` pergunta 2 |
| 13 | Sumarização (F9) | Só se sobrar tempo, com avaliação própria | `[PREMISSA]` |
| 14 | Hardware | RTX 3060 12 GB, 24 threads, 62 GB de RAM, FFmpeg instalado, Tesseract **não** instalado | `[DECIDIDO]` medido no guia, §1.1 |
| 15 | Equipe | Divisão de papéis do planejamento, seção 9 (3 ou 4 integrantes). Nomes não definidos | `[PENDENTE]` |
| 16 | Repositório | O diretório ainda não é um repositório git | `[PENDENTE]` decisão do usuário |

---

## 4. Métricas e experimentos

**Métricas**

- **WER** = (substituições + remoções + inserções) / número de palavras do gabarito. **CER** é o mesmo, em caracteres.
- **Precisão e recall de palavras** (ignoram a ordem; bons para slides): precisão = palavras em comum / palavras da saída; recall = palavras em comum / palavras do gabarito (contagem com multiplicidade).
- **RTF** (real-time factor) = tempo de inferência / duração da mídia. RTF < 1 é mais rápido que o tempo real.
- **Precision@k e Recall@k** (F6): fração dos k primeiros resultados que são relevantes, e fração dos relevantes que aparecem nos k primeiros.
- **MRR** (F6): média, sobre as consultas, de 1 / posição do primeiro resultado relevante.

**Matriz de experimentos**

| RQ | Experimento | Fatores | Combinações (6 vídeos) | Métricas |
|---|---|---|---|---|
| RQ1 | STT | 3 modelos × 4 condições (limpo, 20, 10, 0 dB) | 72 | WER, tempo, RTF |
| RQ2 | OCR | 2 motores × 4 condições (limpo, resize 0,5, blur 9, JPEG 30) | 48 | CER, precisão, recall, tempo |
| RQ3 | Busca (F6) | 20 a 30 consultas com trechos relevantes marcados | — | Precision@5, Recall@5, MRR |
| RQ4 | Entrega HLS (F8) | Com e sem o pipeline de IA em paralelo | — | Tempo de início, buffering, bitrate, tamanho dos metadados |

Ablação (F7): taxa de keyframes (`step_s`) e limiar de troca de slide (`thr`), com efeito no recall de slides e no tempo.

RQ4 é o experimento que responde diretamente à diferença entre análise e transporte pedida pelo tema. Se a F8 for cortada, a fronteira fica só no diagrama e no texto.

**Protocolo**

- Exatidão mede-se nos trechos com gabarito. Vídeos completos servem só para tempo em mídia longa e para a demo.
- Tempo: 3 repetições por configuração, com média e desvio. Relate também o tamanho da amostra e seja cauteloso com conclusões de 6 vídeos.
- Comparação: sempre pelo menos duas alternativas lado a lado. Sem isso o experimento fica fraco.
- Robustez: a degradação aparece nas figuras (WER × SNR) e nas tabelas.

**Medição de tempo** (armadilhas que afetam a nota)

- `transcribe()` devolve um gerador: consuma com `list()` antes de parar o cronômetro.
- A primeira execução inclui inicialização da GPU: faça uma execução de aquecimento e descarte.
- Carregamento do modelo é medido à parte. Só a inferência entra no RTF.
- `language`, `beam_size` e `vad_filter` são iguais entre modelos.

**Normalização de texto.** Uma única função `norm()` para gabarito e saída. As regras de transcrição (números por extenso ou algarismos, hesitações como "é..." e "né", siglas) são escritas no README **antes** de transcrever o gabarito.

---

## 5. Ambiente e stack

| Função | Escolha | Alternativa |
|---|---|---|
| STT | `faster-whisper` (tiny, base, small; medium se couber) | Whisper original |
| OCR | Tesseract (`por`) e EasyOCR | PaddleOCR |
| Imagem | OpenCV (`opencv-python-headless`) | — |
| Indexação multimodal (opcional) | CLIP (`open_clip`) + FAISS | — |
| Métricas | `jiwer`, implementação própria de precisão/recall | — |
| Vídeo | FFmpeg (áudio, keyframes, HLS na F8) | — |
| Backend e front | Página estática com HTML e JavaScript; `python -m http.server` | FastAPI, se a busca precisar de API |
| Sumarização (F9) | LLM local (Ollama) | API, com custo como métrica |

Gerenciador `uv`, Python 3.12. Ferramentas de qualidade: `ruff` e `pytest`. Setup e smoke tests estão em `guia.md`, seção 3, e no guia de implementação, §1.

Os trechos de código do guia de implementação **não foram executados**. A etapa 0 e a etapa 2 incluem testes para validá-los.

---

## 6. Estrutura de pastas

Planejada, a partir do guia de implementação (§1.4) e com pastas adicionadas. Itens marcados ⏳ ainda não existem.

```
tema9/
├── CLAUDE.md                  # este documento
├── guia.md                    # mapa do código e contratos
├── README.md                  # ⏳ comandos para reproduzir tudo; tabela de licença dos vídeos
├── pyproject.toml             # ⏳ versões fixadas
├── .gitignore                 # ⏳ data/raw, data/derived, .venv
├── src/                       # ⏳ código (ver guia.md, seção 4)
│   ├── extract.py
│   ├── stt.py
│   ├── ocr.py
│   ├── degrade.py
│   ├── metrics.py
│   ├── run_experiments.py
│   └── report.py
├── tests/                     # ⏳ testes de fumaça e unitários
├── data/
│   ├── raw/                   # vídeos originais (fora do git)
│   ├── gt/                    # gabaritos humanos: <video>.stt.txt, <video>.slides.json
│   └── derived/               # áudio, ruído, frames, .vtt, .json (regenerável, fora do git)
├── results/                   # raw.csv, tabelas e figuras geradas
├── web/                       # ⏳ index.html; cópias de vídeo, .vtt e .json só para a demo
├── artigo/                    # ⏳ [PENDENTE] modelo SBC e fontes do artigo
└── docs/
    └── decisoes.md            # ⏳ mudanças de plano, com motivo (criar na primeira mudança)
```

Os arquivos de planejamento ficam uma pasta acima, em `redes multimidia/`, e não foram movidos.

---

## 7. Dados e gabaritos

- **Vídeos** (`data/raw/`): 6 aulas de 5 a 15 min, mais vídeos longos só para tempo e demo. Cada um com licença anotada no README.
- **Gabarito de fala** (`data/gt/<video>.stt.txt`): 1 trecho de 2 a 3 min por vídeo, transcrito do zero à mão, seguindo as regras de normalização da seção 4. Custo estimado: 15 a 20 min por trecho.
- **Gabarito de slides** (`data/gt/<video>.slides.json`): se houver o PDF/PPTX original da aula, extraia o texto de cada slide (`pdftotext`) e registre a origem. Senão, digite à mão 10 a 15 slides por vídeo.
- **Consultas da busca** (F6): 20 a 30 consultas com os trechos relevantes marcados, em `data/gt/`. Criar na F6.
- **Formatos de cada arquivo**: `guia.md`, seção 5.

Divergência conhecida: o gabarito de slides e a saída do OCR usam chaves diferentes no guia de implementação (`text` e `texto`). Decida uma antes da etapa 3 (ver `guia.md`, seção 5).

---

## 8. Pipeline e interface (resumo)

| Etapa | Entrada | Saída | Requisito |
|---|---|---|---|
| Extração de áudio | `aulaNN.mp4` | `aulaNN.wav` (16 kHz, mono) e trecho com gabarito | F1 |
| STT | áudio | segmentos, `.vtt` e tempo de inferência | F1, F3 |
| Keyframes | `aulaNN.mp4` | `NNNN.png` e `frames.json` (`t`, `file`) | F2 |
| OCR | keyframes | `slides.json` (`t`, `file`, texto) e tempo | F2, F3 |
| Degradação | áudio e imagens | versões com ruído ou borradas | F4 |
| Métricas | saídas e gabaritos | `results/raw.csv` | F4 |
| Página | mp4, `.vtt`, `slides.json` | player com legenda, busca e clique no slide | F5 |

**Roteiro da demo (≈ 3 min)**: abrir o vídeo, mostrar a legenda sincronizada, buscar um termo, clicar num slide e saltar ao trecho, e mostrar lado a lado uma saída boa e uma ruim (ruído 0 dB) para ilustrar a robustez. Gravar um vídeo de backup.

---

## 9. Artigo, apresentação e defesa

**Artigo (SBC, 5 a 8 páginas, PDF)**

1. Introdução: problema dos vídeos de aula, contribuição e fronteira análise × transporte.
2. Fundamentação teórica: STT, OCR, embeddings multimodais, HLS e QoE.
3. Trabalhos relacionados e tecnologias: artigos do Whisper e do CLIP, legendas automáticas de plataformas.
4. Metodologia: pipeline, dataset, gabarito e protocolo de ruído.
5. Arquitetura e implementação: diagrama em dois planos e justificativa de cada tecnologia.
6. Experimentos e métricas.
7. Resultados e discussão: trade-offs e limitações (sotaque, fonte pequena nos slides, ruído de sala).
8. Conclusão e trabalhos futuros.
9. Referências.

**Apresentação (20 min)**: contexto 3 min; fundamentação 5 min; arquitetura e implementação 5 min; resultados 4 min; conclusão e demo complementar 3 min. Estrutura de slides: título e problema; motivação; conceitos; arquitetura; tecnologias; metodologia; demo; resultados; discussão; conclusão. Vídeo de backup da demo.

**Defesa (10 min)**: arquitetura, protocolos e algoritmos usados; definição de WER, CER, RTF, precisão, recall e MRR; limitações; comparação com alternativas; por que cada tecnologia foi escolhida. Qualquer integrante pode ser arguido sobre qualquer parte.

---

## 10. Etapas e critérios de pronto

| Etapa | Pronto quando | Prazo (planejamento) |
|---|---|---|
| 0. Ambiente | Smoke tests passam em cada máquina; Tesseract com `por` instalado | Até ~15/11 |
| 1. Dataset e gabarito | 6 vídeos com licença anotada, 6 trechos transcritos, slides com gabarito | Começa no dia 1 (é o gargalo) |
| 2. STT (F1, F3) | WER e RTF por modelo e condição de ruído num trecho | Até ~15/11 (prova de conceito) |
| 3. OCR (F2, F3) | CER, precisão e recall por motor num vídeo | Semana seguinte |
| 4. Experimentos (F4) | `raw.csv` completo; `report.py` gera tabelas e figuras | Semana seguinte |
| 5. Interface (F5) | Demo roda de ponta a ponta; backup gravado | Semana seguinte |
| 6. Artigo e apresentação | Artigo revisado, slides, ensaio cronometrado; PDF enviado 48 h antes | Última semana antes da apresentação |

**Critério geral do projeto** (guia de implementação, §0):

- `python src/run_experiments.py` gera `results/raw.csv` sem intervenção manual.
- `python src/report.py` gera tabelas e figuras do artigo a partir do CSV.
- A página abre com legenda sincronizada, e clicar no slide leva ao instante do vídeo.
- Existe um README com os comandos para reproduzir tudo.

---

## 11. Riscos

| Risco | Mitigação |
|---|---|
| Sotaque e ruído pioram o STT | Tratar como resultado: comparar e reportar com honestidade |
| Hardware limitado | Modelos pequenos ou Colab; medir RTF |
| Demo falhar | Vídeo de backup gravado |
| Licença dos vídeos | Gravar material próprio ou usar licença livre |
| Resultado de IA aceito sem análise | Conferir amostras à mão e explicar os erros no artigo |
| Escopo grande demais | Cortar F8 e F9 primeiro |
| Gabarito atrasado | Começar a transcrição no primeiro dia; é o gargalo |

---

## 12. Pendências com o professor e a equipe

1. O Tema 9 está livre? (17 temas para 12 equipes; o número do tema não é o número da equipe.)
2. Posso usar vídeos de terceiros ou preciso gravar o material?
3. Posso usar API paga de IA ou devo usar modelos locais? Se for API, o custo entra como métrica.
4. Onde está o modelo SBC "trabalhado em sala"? Não está na pasta da disciplina.
5. Data exata da apresentação e e-mail de entrega.
6. Aprovação do título refinado.
7. Equipe: nomes e divisão de papéis (planejamento, seção 9).
8. Regras de transcrição (números, hesitações, siglas): a equipe define antes de começar o gabarito.

---

## 13. Checklist de entrega

- [ ] Pergunta técnica escrita em uma frase (slide 1 e artigo)
- [ ] Dataset e gabarito prontos, com licença de cada vídeo
- [ ] Pelo menos 2 modelos comparados por tarefa
- [ ] Tabela de resultados com média, desvio e tamanho da amostra
- [ ] Protótipo rodando e backup da demo gravado
- [ ] Artigo de 5 a 8 páginas, em PDF, no modelo SBC
- [ ] Artigo enviado por e-mail 48 h antes
- [ ] Slides dentro de 20 min, testados com cronômetro
- [ ] Erros do OCR e do STT classificados e explicados
- [ ] Cada integrante consegue explicar todas as partes
- [ ] `guia.md` e `README.md` atualizados com o estado final

---

## 14. Primeiros passos

1. Confirmar as pendências 1 a 3 da seção 12 com o professor, até ~15/11.
2. Etapa 0: criar `pyproject.toml` (Python 3.12), `README.md`, `.gitignore` e os testes de fumaça. Instalar o Tesseract com o idioma `por` (o usuário executa o comando de sistema).
3. Etapa 1, em paralelo: escolher os trechos, transcrever à mão e registrar as licenças.
4. Ao terminar cada etapa, atualizar `guia.md` (seções 1 e 4) e resumir o que ficou pronto, o que foi decidido e o que depende do usuário.
