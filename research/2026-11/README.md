# Pesquisa de novembro de 2026 — reduzir o RPS do Vitrôncio

Código das trilhas de pesquisa rodadas em paralelo (sem os dados: eles se refazem com `tools/fetch.mjs`, `shared/export.mjs` e os scripts de download de cada trilha). Os caminhos nos scripts apontam para `/home/user/bdv-lab/`; ajuste ao reutilizar. Protocolo comum em `shared/PROTOCOL.md`: DEV 2013–2021, HOLDOUT 2022–2026, odds só como régua, placar em `shared/score.py`, combinação em `final/stack.py`.

RPS do v1 no placar comum: DEV 0,20526 · HOLDOUT 0,20541 (mercado ≈ 0,2003). Com hiperparâmetros ajustados só no passado de cada temporada ("v1 honesto", `core/v1h.py`) o v1 fica em ≈ 0,2060–0,2066: os números do v1 são ≈ 0,001 otimistas.

| trilha | situação | melhor resultado (dRPS contra o v1: DEV / HOLDOUT) |
|---|---|---|
| `context/` — contexto do jogo e outras competições | **concluída e em produção** (`site/js/engine/context.js`) | `ctx_core` −0,00126 / −0,00107 |
| `xg/` — xG por finalização (posição do chute) e ratings de xG | interrompida antes do relatório | `xg_ll` −0,00115 / −0,00080 (IC95 abaixo de zero nos dois); `xg_adj_wf` −0,00155 / −0,00075 |
| `zoo/` — pi-ratings, Elo, GAP, Berrar + árvores | interrompida | só recalibração: −0,00073 / −0,00030 |
| `core/` — melhorias no filtro (variância na virada de temporada, mando, etc.) | interrompida | sem previsão final |
| `players/` — escalações e plus-minus | interrompida | atributos construídos (`build_features.py`), sem previsão final |
| `neural/` — redes sobre sequências | interrompida | sem previsão final |

Próximo passo mais promissor: terminar a trilha `xg/` (conferir `xg_ll` e `xg_adj_wf`, que parecem somar com o contexto) e empilhar com `final/stack.py`.

Revisão de literatura (resumo): os melhores modelos sem odds ficam 0,7–2,1% atrás das casas; estávamos 2,5% atrás. Ideias em ordem de ganho esperado: força por escalação (plus-minus regularizado), variância maior na virada de temporada, mando dependente de contexto, empilhamento entre famílias de modelos, gols ajustados ao estado do jogo.

## v3 (publicado)

O xG por chute de `model/data/xg.json` entrou como um segundo rating de gols (canal `x`: 15% placar, 85% xG, ruído 0,7 — ajustes escolhidos pela trilha só com 2013–2021), combinado no `stack` por um coeficiente estimado a cada temporada só com o passado (≈ 0,64).

| backtest de produção | geral | 2013–2021 | 2022–2026 |
|---|---|---|---|
| v2 | 0,2047 | 0,2048 | 0,2045 |
| v3 | 0,2042 | 0,2042 | 0,2044 |
| trocar o proxy pelo xG num rating só (descartado) | 0,2036 | 0,2031 | 0,2045 |

A troca direta rende mais no geral, mas nada em 2022–2026; a versão com dois ratings é a que não piora em nenhum período. O ganho em 2022–2026 é pequeno demais para ser distinguido de ruído. Empilhar pi-ratings/Elo/árvores (`zoo/`) não ajudou em 2022–2026.

## Testes de aprendizado de máquina (parciais, nenhum em produção)

dRPS contra o v1. "–" = a trilha foi interrompida antes de consultar 2022–2026.

| teste | DEV 2013–2021 | HOLDOUT 2022–2026 |
|---|---|---|
| `zoo` árvores sobre todos os ratings (Elo, pi, GAP…) | −0,00010 | +0,00017 |
| `zoo` só ratings de gols / só Elo | +0,00115 / +0,00135 | +0,00182 / +0,00234 |
| `zoo` v1 + rating de "domínio" (`co_dom`) | −0,00104 | −0,00027 |
| `zoo` mistura recalibrada (`pool_recal_zoo`) | −0,00114 | −0,00032 |
| `zoo` só recalibração | −0,00073 | −0,00030 |
| `neural` DeepSets ajustando o v1 (média de sementes) | −0,00108 | – |
| `neural` GRU ajustando o v1 | −0,00112 | – |
| `neural` MLP / embeddings com Poisson | +0,02 / +0,018 | – |
| `players` atributos de elenco em logit ordenado | −0,0001 a −0,0004 (IC cruza zero) | – |
| `players` atributos de elenco em árvores (3–4 folhas) | −0,00138 a −0,00145 (IC abaixo de zero) | – |

Leitura: em `zoo`, ganhos de ≈ 0,001 no DEV caíram para ≈ 0,0003 em 2022–2026, e esse resto é quase todo recalibração, que o v2 já tem nas constantes do contexto. `neural` e `players` nunca chegaram ao HOLDOUT nem gravaram previsões; os números acima são dos logs. Vale retomar primeiro as árvores com atributos de elenco. Esses modelos rodam em Python: para entrar no site teriam de gerar, no reajuste semanal, o ajuste de cada jogo futuro.


## Fechamento (v3.1 e v3.2, publicados)

- **v3.1** — calibração da cadeia de Markov para reproduzir os gols esperados de cada lado: 0,2042 → 0,2040 (2022–2026: 0,2044 → 0,2040). Sem parâmetro ajustado.
- **v3.2** — escalação anunciada (`players/js/`, portado para `site/js/engine/lineup.js`). No reajuste de produção: 0,2040 → 0,2030 no geral, 0,2041 → 0,2033 em 2022–2026. Na pesquisa, sobre o contexto: DEV −0,00134 [−0,00217; −0,00047], HOLDOUT −0,00095 [−0,00189; +0,00001]. Ressalva: o HOLDOUT desta trilha foi consultado duas vezes (uma versão anterior deu −0,00078); a escolha dos atributos foi só no DEV. Com o "time esperado" no lugar da escalação: zero (+0,00001 no HOLDOUT), por isso o ajuste só entra quando a escalação sai.
- **Árvores com atributos de elenco** (`players/gbm_final.py`): sobre o contexto, somam ≈ o mesmo que o deslocamento linear; ficou o linear, que roda no navegador.
- **Redes neurais** (`neural/`): DeepSets e GRU, sobre contexto + escalação: −0,00005 e −0,00012 no HOLDOUT, IC cruzando zero. Não entraram.
- **Trilha core**: novo canal de gols (EKF com covariância ataque–defesa e surpresa de finalizações) e rating ordinal. Sobre o contexto: +0,00011 no HOLDOUT. Não entraram; detalhes de porte na conversa de 18/11/2026 e em `core/`.

- **Desfalques na véspera** (`absent/`, v3.3): suspensos por cartões (validação 2013+: 2,9% dos suspensos apareceram no jogo) e titulares fora do último jogo. Sobre o contexto, sem escalação: DEV −0,00019 [−0,00062; +0,00023], HOLDOUT −0,00047 [−0,00104; +0,00011], 4 de 5 temporadas melhores; coeficientes negativos em todas as temporadas (−0,12 a −0,22 por suspensos). Abaixo do limiar de ruído no DEV; entrou por ser consistente e plausível. Com a escalação anunciada: nada (+0,00009 / −0,00018). Árbitro: sem efeito, mas a ESPN não tem o árbitro de 2014 a 2021.
- **Mudança de técnico** (`coach/`, Wikipedia): só o "tumulto" (primeiro jogo depois da saída do técnico, ou técnico interino) tem sinal: −0,057 nos gols do time (z −2,6). Sobre o contexto: DEV −0,00018 [−0,00035; −0,00001], HOLDOUT −0,00033 [−0,00076; +0,00009]. Não há "efeito técnico novo" positivo. Não entrou: ganho pequeno, a Wikipedia limitou o acesso (429) e interinos ficam desatualizados nas páginas (ex.: Fluminense "interino" desde agosto de 2026), o que manteria a penalidade ligada por meses.
- **Múltiplas**: sugestão de k pernas por rodada (os k palpites do Best Against com maior probabilidade corrigida × odd), 2014–2026: −6% (1), −11% (2), −50% (3), −31% (4), −79% (5) por aposta. Com a margem média (6,7%), mesmo com o mercado certo: −12% (2), −18% (3), −28% (5).
