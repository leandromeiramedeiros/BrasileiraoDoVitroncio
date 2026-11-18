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
