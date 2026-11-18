# Brasileirão do Vitrôncio — guia para sessões do Claude

Site estático (GitHub Pages) da Série A com simulador próprio. Interface e comentários em **pt-BR**. Sem build, sem dependências: ES modules no navegador, Node 22 nas ferramentas.

## Regras do projeto

- Não é ferramenta de aposta. As odds aparecem no site **apenas como comparação** (página "Odds" e bloco na página do jogo: modelo × odds × resultado) e como régua no backtest. **Nunca** viram entrada, alvo ou critério de seleção do modelo — a comparação só faz sentido se o modelo for independente do mercado. Nada de linguagem de "aposta de valor".
- `site/js/engine/` é JS puro compartilhado entre navegador e `tools/`. Qualquer mudança no motor precisa manter os dois lados funcionando; rode `npm test`.
- O dono autorizou push direto em `main` (cada push publica o site).
- Teste de interface: sirva `site/` e use Playwright com respostas gravadas da ESPN (a CDN deles recusa navegador headless). Não tente contornar esse bloqueio.

## Interface

- O HTML de cada página é gerado por inteiro, mas quem aplica no DOM é `site/js/app/morph.js` (atualização incremental). Listas que crescem usam `data-mk` como chave estável; não troque por `innerHTML`, senão volta a piscar.
- Tom dos textos: direto e neutro ("modelo", "probabilidade"). Sem emoji, sem exclamações, sem personagem. O nome Vitrôncio aparece só na marca e na página do modelo.
- Página inicial curta; o detalhe fica nas páginas de jogo.

## Reajuste semanal (o pedido mais comum)

1. `npm run weekly` (≈1–2 min). Sem rede para a ESPN, pare e avise.
2. Compare o bloco `backtest geral` do log com o `site/data/model.json` anterior (`git show HEAD:site/data/model.json`). RPS deve ficar perto de 0,2056 (referência de mercado ≈ 0,2003, ingênuo ≈ 0,2179). Piora acima de ~0,001 merece investigação antes de publicar.
3. Olhe os hiperparâmetros (`params.g`) e as intensidades (`hazard`): mudanças bruscas de uma semana para outra são sinal de problema nos dados.
4. `npm test`, commit ("Reajuste semanal AAAA-MM-DD" + resumo dos números) e push.
5. Relate ao dono: números do backtest, o que mudou no ranking de força, e qualquer anomalia.

Virada de temporada: `tools/fetch.mjs` já busca o ano seguinte em dezembro; `refit` usa a maior temporada da Série A como atual. Promovidos/rebaixados são tratados em `prepareTeam` (ratings.js).

## Mapa do modelo

- `ratings.js` — EKF Poisson por canal (`g` gols, `sh`, `sot`, `co`, `yc`, `fo`). `applyMatch` atualiza, `expectedMatch` prevê (inclui a camada `stack`).
- `match-model.js` — cadeia de Markov exata: `scoreDistribution(hz, lh, la, estado)`.
- `season.js` — Monte Carlo da temporada. `sim-match.js` — jogo simulado narrado. `players.js` — escalação provável e artilheiros.
- `tools/refit.mjs` — tudo que é estimado: xG proxy, hiperparâmetros (Nelder–Mead), stacking, GLMs de intensidade, backtest, jogadores, exportação.

## Ideias para evoluir o algoritmo (ainda não feitas)

- Outras competições (Copa do Brasil, Libertadores, estaduais) para mais jogos por time e para medir desgaste/rodízio.
- Distância de viagem e dias de descanso como covariáveis.
- Atualização ao vivo dos gols esperados com as finalizações do próprio jogo.
- Árbitro como fator nos cartões; minutos reais de acréscimo.
- Validar a calibração da simulação de temporada (título/Z4) em temporadas passadas.
