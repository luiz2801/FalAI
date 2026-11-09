# Tema 9 — Inteligência Artificial Aplicada a Sistemas Multimídia

**Disciplina:** Redes Multimídia (EN05228) — UFPA — Prof. Raimundo Viégas Jr.
**Fonte dos requisitos:** `Seminarios_de_Sistemas_e_Redes_Multimidia_2026_Atualizado.pdf`
**Apresentações:** dezembro de 2026 (data exata a confirmar)

> Documento de planejamento da equipe: requisitos do enunciado, proposta de solução, experimentos e cronograma.

---

## 1. Resumo da proposta

**Problema.** Vídeos de aula e palestras são difíceis de pesquisar e de acessar: não têm legenda, o conteúdo dos slides existe só como imagem e nada é indexado.

**Solução.** Um pipeline de IA que, a partir de um vídeo de aula:

1. transcreve o áudio em PT-BR e gera legendas (`.vtt`);
2. extrai o texto dos slides por OCR;
3. indexa transcrição e slides para busca;
4. (opcional) entrega o resultado por streaming adaptativo (HLS), em um plano separado.

**Pergunta de investigação (rascunho).** Qual é o trade-off entre qualidade (WER/CER) e tempo de processamento de modelos de STT e OCR em vídeos de aula com ruído, e quanto o processamento de IA afeta a entrega em streaming?

**Título provisório.** *Análise inteligente de vídeos de aula: transcrição, OCR e indexação multimodal com avaliação de qualidade e custo computacional.* (O título pode ser refinado com aprovação do professor.)

---

## 2. Requisitos do enunciado

### 2.1 Implementação (peso 5,0)

| ID | Requisito | Como verificar |
|---|---|---|
| I1 | Ter protótipo, aplicação, experimento ou benchmark funcional | Demo ao vivo |
| I2 | Ter **pelo menos uma evidência experimental ou métrica** | Tabela ou gráfico de resultados |
| I3 | Responder a uma **pergunta técnica**. "A ferramenta funciona" não basta | Pergunta explícita no artigo e no slide 1 |
| I4 | Definir o que avaliar, como e com quais métricas | Seção de Metodologia |
| I5 | Relacionar explicitamente teoria, implementação e resultados | Discussão que cita a teoria para explicar cada resultado |
| I6 | Explicar arquitetura e decisões técnicas, mesmo usando bibliotecas prontas | Diagrama e justificativas |
| I7 | Ser viável no tempo e nos recursos disponíveis | Escopo enxuto (seção 4) |

Critérios de nota: funcionamento, aderência ao tema, demonstração, qualidade técnica e métricas.

### 2.2 Artigo técnico (peso 3,0)

| ID | Requisito |
|---|---|
| Ar1 | PDF de 5 a 8 páginas, no modelo SBC trabalhado em sala |
| Ar2 | Enviado por e-mail com pelo menos **48 h** de antecedência |
| Ar3 | Seguir as 9 seções recomendadas (ver seção 6) |
| Ar4 | Registrar **limitações e trabalhos futuros** |
| Ar5 | Critérios de nota: fundamentação, metodologia, referências, implementação, resultados, discussão e clareza |

### 2.3 Apresentação e defesa (peso 2,0)

| ID | Requisito |
|---|---|
| Ap1 | Apresentação de no máximo 20 min, com a estrutura sugerida (seção 7) |
| Ap2 | Arguição de no máximo 10 min |
| Ap3 | **Qualquer integrante** pode ser questionado individualmente (fundamentação, arquitetura, implementação e resultados) |
| Ap4 | Explicar os conceitos sem depender dos slides |
| Ap5 | A demo pode ser interrompida para perguntas técnicas |
| Ap6 | Saber discutir limitações, alternativas e justificar as escolhas |

### 2.4 Regras gerais

- Um único tema por equipe, respeitando a disponibilidade da turma.
- O título pode ser refinado mediante aprovação do professor.
- A equipe define um problema técnico concreto e uma pergunta de investigação.
- Resultados produzidos por ferramentas de IA **não substituem** a análise técnica dos alunos.

---

## 3. Requisitos específicos do Tema 9

| Item do enunciado | Interpretação prática |
|---|---|
| **Objetivo:** IA sobre o conteúdo, "diferenciando análise/produção de mídia do transporte pela rede" | O trabalho trata de IA que **processa a mídia** (o que ela contém), não de otimização de rede (isso é o Tema 5). A fronteira deve aparecer na arquitetura e no texto do artigo. |
| **Conceitos sugeridos:** visão computacional, áudio, vídeo, OCR, legendagem, indexação, IA generativa, multimodalidade | São sugestões, não é preciso cobrir todos. Cobrir a fundamentação de tudo o que for usado. |
| **Implementação mínima:** "analise, transforme, classifique, transcreva, gere **ou** enriqueça" | O "ou" indica que **um verbo basta**. Transcrever e enriquecer ficam acima do mínimo. |
| **Métricas sugeridas:** acurácia, qualidade, tempo de inferência, robustez, comparação | Ver mapeamento abaixo. |

### Mapeamento das métricas sugeridas

| Métrica | O que significa na prática | Experimento |
|---|---|---|
| Acurácia | Erro frente a um gabarito (WER, CER) | RQ1, RQ2 |
| Qualidade | Qualidade da saída (legibilidade da legenda, relevância da busca) | RQ1 (qualitativa), RQ3 |
| Tempo de inferência | Segundos de processamento por minuto de mídia (RTF) | RQ1, RQ2 |
| Robustez | Degradação sob ruído, baixa resolução e desfoque | RQ1, RQ2 |
| Comparação | Pelo menos 2 modelos ou configurações lado a lado | Todos |

> **Atenção:** "comparação" é o ponto mais fácil de esquecer. Sem pelo menos duas alternativas, o experimento fica fraco.

---

## 4. Requisitos da solução e priorização

### Obrigatório (MVP — cobre sozinho o mínimo do tema)

- **F1:** extrair o áudio e gerar transcrição em PT-BR, com legenda `.vtt`.
- **F2:** extrair keyframes e aplicar OCR nos slides.
- **F3:** comparar pelo menos 2 modelos de STT e 2 motores de OCR.
- **F4:** medir WER, CER e tempo de inferência, com teste de ruído em pelo menos 2 níveis.
- **F5:** interface mínima de demonstração (player com legenda e texto dos slides).

### Desejável

- **F6:** índice de busca sobre transcrição e OCR, com Precision@k e MRR (RQ3).
- **F7:** ablação da taxa de amostragem de keyframes (1 fps vs. detecção de troca de slide).

### Opcional (cortar primeiro se faltar tempo)

- **F8:** entrega por HLS com e sem a IA rodando em paralelo (RQ4). Mostra bem a fronteira análise vs. transporte, mas o tema é sobre análise.
- **F9:** resumo por LLM, com avaliação própria (difícil de avaliar bem).

### Requisitos não funcionais

- **N1:** reprodutível, com README e script que rode todos os experimentos.
- **N2:** dataset e gabarito prontos **antes** dos experimentos (é o gargalo).
- **N3:** rodar no hardware disponível, com modelos pequenos ou Colab.
- **N4:** licença dos vídeos clara.

---

## 5. Arquitetura

Dois planos separados, para deixar explícita a diferença entre análise de conteúdo e transporte:

```
PLANO DE CONTEÚDO (IA, offline)
Vídeo MP4 ──► FFmpeg (demux)
               ├─► áudio ──────► STT (Whisper) ──► legendas .vtt + transcrição
               ├─► keyframes ──► OCR (Tesseract/PaddleOCR) ──► texto dos slides
               └─► keyframes ──► embeddings (CLIP) ──► índice vetorial (FAISS)
                                        │
                              metadados (SQLite) ──► API (FastAPI) ──► busca

PLANO DE ENTREGA (medido separadamente, opcional)
Vídeo ──► FFmpeg ──► HLS (360p/720p) ──► player web (hls.js) + legendas WebVTT
```

### Tecnologias sugeridas

| Função | Opções |
|---|---|
| STT | Whisper ou faster-whisper (comparar tiny/base/small) |
| OCR | Tesseract, PaddleOCR |
| Indexação multimodal | CLIP (open_clip) + FAISS |
| Sumarização (opcional) | LLM local (Ollama) ou API (incluir custo como métrica) |
| Vídeo | FFmpeg; HLS para entrega |
| Backend / front | Python + FastAPI; HTML + hls.js |

As escolhas precisam ser justificadas no artigo (regra 14 do enunciado).

---

## 6. Experimentos

| RQ | Experimento | Métricas |
|---|---|---|
| RQ1 | STT: 3 tamanhos de Whisper × ruído (SNR 20, 10 e 0 dB) em áudio de aula PT-BR | WER, tempo de inferência, RTF |
| RQ2 | OCR de keyframes: Tesseract vs. PaddleOCR, slides limpos vs. fotos/borrados | CER, precisão e recall de palavras, tempo |
| RQ3 | Busca multimodal com consultas de teste e gabarito | Precision@5, Recall@5, MRR |
| RQ4 | Entrega HLS com e sem o pipeline de IA em paralelo | Tempo de início, buffering, bitrate, tamanho dos metadados |

**Dataset:** 6 a 10 vídeos de aula (gravados pela equipe ou com licença livre).
**Ground truth:** transcrição manual de 2 a 3 min por vídeo; gabarito dos slides; 20 a 30 consultas com trechos relevantes marcados.
**Ablação:** frequência de keyframes.

> RQ4 é o experimento que responde diretamente à diferenciação análise vs. transporte pedida pelo tema.

---

## 7. Artigo, apresentação e defesa

### Estrutura do artigo (SBC, 5 a 8 páginas)

1. **Introdução:** problema dos vídeos de aula e contribuição.
2. **Fundamentação teórica:** STT, OCR, embeddings multimodais, HLS e QoE.
3. **Trabalhos relacionados e tecnologias:** artigos do Whisper e do CLIP, legendas automáticas de plataformas de vídeo.
4. **Metodologia:** pipeline, dataset, ground truth e protocolo de ruído.
5. **Arquitetura e implementação.**
6. **Experimentos e métricas.**
7. **Resultados e discussão:** trade-offs e limitações (sotaque, fonte pequena nos slides, ruído de sala).
8. **Conclusão e trabalhos futuros.**
9. **Referências.**

### Apresentação (20 min)

| Bloco | Tempo |
|---|---|
| Contexto | 3 min |
| Fundamentação | 5 min |
| Arquitetura e implementação | 5 min |
| Resultados | 4 min |
| Conclusão e demo complementar | 3 min |

Estrutura de slides do enunciado: título/equipe/problema; motivação; conceitos; arquitetura; tecnologias; metodologia; demonstração; resultados; discussão; conclusão. Ter um **vídeo de backup** da demo.

### Defesa (10 min)

Preparar respostas para: arquitetura, protocolos e algoritmos usados, definição das métricas (o que é WER, CER, RTF, MRR), limitações, comparação com alternativas e por que cada tecnologia foi escolhida.

---

## 8. Matriz de rastreabilidade

| Requisito | Onde fica a evidência |
|---|---|
| I2, I3, I4 | Seções 4 e 6 do artigo; slide de metodologia |
| I5 | Seção 7 do artigo (discussão) |
| I6 | Diagrama na seção 5 do artigo; slide de arquitetura |
| Ar4 | Seção 8 do artigo; slide final |
| Tema 9: fronteira análise vs. transporte | Parágrafo na introdução e diagrama em dois planos |
| Tema 9: comparação | Tabelas de resultados lado a lado |
| Ap3, Ap4 | Ensaio em que cada integrante explica todas as partes |

---

## 9. Divisão de trabalho

Equipes da turma têm 3 ou 4 integrantes. Sugestão (todos devem conhecer tudo, regra 17):

| Papel | 4 integrantes | 3 integrantes |
|---|---|---|
| Áudio e STT (RQ1) | A | A |
| OCR e indexação visual (RQ2, RQ3) | B | B |
| API, player e entrega HLS (RQ4) | C | C (junto com dataset) |
| Dataset, ground truth, métricas, coordenação do artigo | D | C |

---

## 10. Cronograma (início em 09/11/2026)

| Período | Entregas |
|---|---|
| Até ~15/11 | Fechar pergunta de pesquisa e dataset; prova de conceito do STT; confirmar pontos da seção 11 com o professor |
| Semana seguinte | OCR, keyframes e índice |
| Semana seguinte | Rodar experimentos RQ1 a RQ4 |
| Semana seguinte | API e player; rascunho do artigo |
| Última semana antes da apresentação | Revisão, slides, ensaio cronometrado; envio do PDF com 48 h de antecedência |

---

## 11. Pontos a confirmar com o professor

1. O Tema 9 está livre? (17 temas para 12 equipes; o número do tema não é o número da equipe.)
2. Posso usar vídeos de terceiros ou preciso gravar o material?
3. Posso usar API paga de IA ou devo usar modelos locais? (Se API, o custo entra como métrica.)
4. Onde está o modelo SBC "trabalhado em sala"? (Não está na pasta da disciplina.)
5. Data exata da apresentação e e-mail de entrega.
6. Aprovação do título refinado.

---

## 12. Riscos

| Risco | Mitigação |
|---|---|
| Sotaque e ruído pioram o STT | Tratar como resultado: comparar e reportar com honestidade |
| Hardware limitado | Modelos pequenos ou Colab; medir RTF |
| Demo falhar | Vídeo de backup gravado |
| Licença dos vídeos | Gravar material próprio ou usar licença livre |
| Resultados de IA aceitos sem análise | Conferir amostras manualmente e explicar os erros no artigo |
| Escopo grande demais | Cortar F8 e F9 primeiro |

---

## 13. Checklist de entrega

- [ ] Pergunta técnica escrita em uma frase
- [ ] Dataset e gabarito prontos
- [ ] Pelo menos 2 modelos comparados por tarefa
- [ ] Tabela de resultados com média e variação
- [ ] Protótipo rodando e backup da demo gravado
- [ ] Artigo de 5 a 8 páginas, em PDF, no modelo SBC
- [ ] Artigo enviado por e-mail 48 h antes
- [ ] Slides dentro de 20 min, testados com cronômetro
- [ ] Cada integrante consegue explicar todas as partes
