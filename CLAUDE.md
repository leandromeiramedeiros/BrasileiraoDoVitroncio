# Brasileirão do Vitrôncio — guia para sessões do Claude

Site estático (GitHub Pages) da Série A com simulador próprio. Interface e comentários em **pt-BR**. Sem build, sem dependências: ES modules no navegador, Node 22 nas ferramentas.

## Regras do projeto

- As odds **nunca** viram entrada, alvo ou critério de seleção da previsão do Vitrôncio — a comparação só faz sentido se o modelo for independente do mercado.
- Aba **Best Against** (`#/contra`, pedido explícito do dono em 18/11/2026): lista de oportunidades onde o Vitrôncio mais discorda do mercado, ordenada pelo ganho esperado com a probabilidade corrigida pelo histórico de discordâncias parecidas (`site/js/engine/against.js`; células e retrospecto calculados em `tools/refit.mjs`). É o único lugar que combina modelo e mercado. Sempre com o aviso no topo e o retrospecto real (negativo na maior parte das temporadas); não esconda nem suavize esses números. A comparação completa fica na sub-aba `#/contra/comparar`; as múltiplas em `#/contra/multiplas`: sugestões de 2 a 6 pernas (resultado), montador com vários mercados por jogo (`site/js/engine/markets.js`: resultado, dupla chance, gols, ambos marcam pela matriz de placares, com a correlação do mesmo jogo; cartões amarelos, escanteios e expulsão como independentes, recalibrados por `against.calMercados`), retrospecto por número de pernas (`against.multiplas`) e a tabela de calibração dos outros mercados (`against.mercados`). Sempre com o aviso de que a margem da casa multiplica. A ESPN só publica odds de resultado e de mais/menos 2,5 gols; nos outros mercados quem usa digita a odd.
- `site/js/engine/` é JS puro compartilhado entre navegador e `tools/`. Qualquer mudança no motor precisa manter os dois lados funcionando; rode `npm test`.
- O dono autorizou push direto em `main` (cada push publica o site).
- Teste de interface: sirva `site/` e use Playwright com respostas gravadas da ESPN (a CDN deles recusa navegador headless). Não tente contornar esse bloqueio.

## Interface

- O HTML de cada página é gerado por inteiro, mas quem aplica no DOM é `site/js/app/morph.js` (atualização incremental). Listas que crescem usam `data-mk` como chave estável; não troque por `innerHTML`, senão volta a piscar.
- Tom dos textos: direto e seco. Sem emoji, sem exclamações.
- **O Vitrôncio é o Victor**, amigo do dono, e é o rosto do site (pedido do dono). O modelo se chama Vitrôncio em toda a interface ("Probabilidades do Vitrôncio", "Vitrôncio × mercado"). Imagens em `site/img/` (recortes sem fundo: `vitroncio-cara` rosto, `vitroncio-joia` sinal de positivo, `vitroncio-olho` olhando por cima dos óculos; favicon). Só os recortes são publicados, nunca as fotos originais.
  - Presença fixa: marca no cabeçalho, rodapé, ao lado de toda barra de probabilidade, página "Odds" e topo da página do modelo.
  - Aparições e easter eggs ficam em `site/js/app/vitroncio.js` (a lista está no cabeçalho do arquivo). São escondidos de propósito: não anuncie na interface nem no README. Falas curtas e secas, na voz dele; nada que zombe de um time específico.
- Página inicial curta; o detalhe fica nas páginas de jogo.
- Menu com 6 itens: Início, Jogos, Campeonato (Tabela | Simulador), Times (Times | Jogadores), Best Against (Oportunidades | Comparação completa), Modelo. Abas internas em `SUBS` e endereços antigos em `ALIAS` (`site/js/app/main.js`). Não acrescente item ao menu sem tirar outro.
- Página do modelo: a tabela "Pesquisa" e as referências vêm de `site/js/app/views/refs.js`. Ao testar algo novo, acrescente a linha com o efeito em 2022–2026; só cite livro ou artigo que foi de fato usado, sem inventar páginas ou DOI.

## Reajuste semanal (o pedido mais comum)

1. `npm run weekly` (≈1–2 min). Sem rede para a ESPN, pare e avise.
2. Compare o bloco `backtest geral` do log com o `site/data/model.json` anterior (`git show HEAD:site/data/model.json`). RPS deve ficar perto de 0,2030 com escalação e 0,2037 sem (`rpsSemEscalacao`); referência de mercado ≈ 0,2003, ingênuo ≈ 0,2179. Piora acima de ~0,001 merece investigação antes de publicar.
3. Olhe os hiperparâmetros (`params.g`) e as intensidades (`hazard`): mudanças bruscas de uma semana para outra são sinal de problema nos dados.
4. `npm test`, commit ("Reajuste semanal AAAA-MM-DD" + resumo dos números) e push.
5. Relate ao dono: números do backtest, o que mudou no ranking de força, e qualquer anomalia.

Virada de temporada: `tools/fetch.mjs` já busca o ano seguinte em dezembro; `refit` usa a maior temporada da Série A como atual. Promovidos/rebaixados são tratados em `prepareTeam` (ratings.js).

## Mapa do modelo

- `ratings.js` — EKF Poisson por canal (`g` gols, `sh`, `sot`, `co`, `yc`, `fo`). `applyMatch` atualiza, `expectedMatch` prevê (inclui a camada `stack`).
- `context.js` — contexto do jogo aplicado sobre os gols esperados: time já rebaixado/campeão (calculado da tabela, também no navegador), jogo continental em seguida, mando fora da sede, portões fechados. Coeficientes em `model.ctx`; os dados vêm de `tools/lib/context.mjs` (calendário das taças em `history.json.gz`, cidades em `model/data/cities.json`). Clube promovido de cidade nova: acrescente a cidade em `cities.json`.
- Canal `x` (v3): segundo rating de gols, mesmo filtro do `g`, observando o xG por finalização de `model/data/xg.json` (id do jogo -> xG de cada lado, escala de gols; Séries A e B 2014–nov/2026). Entra na previsão pelo termo `a4` do `stack`. Jogo sem xG por chute cai na mesma observação do `g`. O arquivo veio da pesquisa (`research/2026-11/xg/`) e **ainda não é atualizado pelo `npm run weekly`** — portar essa etapa para `tools/` é a pendência mais importante.
- `lineup.js` (v3.2) — escalação anunciada: quanto do time habitual começa (`usage`), quanto dos gols+assistências recentes está em campo (`ga`) e titulares habituais no banco (`bench`). Coeficientes em `model.ctx.lineup`, aplicados por `applyContext` só quando a ESPN publica as escalações (o navegador busca o summary dos jogos das próximas 2 h). Sem escalação não há ajuste: o "time esperado" não ajudou. Histórico em `model/data/rosters.json.gz`, atualizado por `tools/rosters.mjs` a partir de `model/cache/sum` (só Série A da temporada atual). O navegador usa `model.lineupSnaps` (foto de cada time no reajuste), que fica até uma semana desatualizada. **Não verificado:** se o summary pré-jogo da ESPN marca `starter` antes do apito; confira num dia de jogo.
- `absent.js` (v3.3) — desfalques conhecidos na véspera: suspensos por cartões (`ev` com id do jogador, regra da CBF) e titulares habituais fora do último jogo. Coeficientes em `model.ctx.absent`; `fixtures[].ab` traz os atributos e os nomes dos suspensos só do próximo jogo de cada time (depois dele, a foto não vale). Aplicado por `applyContext` só quando não há escalação anunciada.
- `match-model.js` — `calibrateLambdas` acerta as taxas-base para a cadeia reproduzir os gols esperados de cada lado; `expectedMatch` devolve `lh/la` (taxas da cadeia) e `xh/xa` (gols esperados). Na interface, mostre `pre.xgH/xgA`, não `e.lh/e.la`.
- `match-model.js` — cadeia de Markov exata: `scoreDistribution(hz, lh, la, estado)`.
- `season.js` — Monte Carlo da temporada. `sim-match.js` — jogo simulado narrado. `players.js` — escalação provável e artilheiros.
- `tools/refit.mjs` — tudo que é estimado: xG proxy, hiperparâmetros (Nelder–Mead), stacking, GLMs de intensidade, backtest, jogadores, exportação.

## Ideias para evoluir o algoritmo (ainda não feitas)

- **Pesquisa interrompida em novembro de 2026:** código e placar parcial em `research/2026-11/` (leia o README de lá antes de recomeçar). O xG por finalização entrou no v3 como segundo rating (ganho pequeno: 0,2047 → 0,2042). A calibração da cadeia para reproduzir o λ de cada lado entrou no v3.1 (0,2042 → 0,2040, ganho nos dois períodos, sem parâmetro ajustado).

- Outras competições como evidência de força (Copa do Brasil, Libertadores): testado em novembro de 2026, ganho ≈ 0,0001 de RPS — não compensou. Estaduais, descanso, viagem, clima, clássicos e "briga contra o rebaixamento" também foram testados e não ajudaram; o que ajudou virou `context.js`.
- Distância de viagem e dias de descanso como covariáveis.
- Atualização ao vivo dos gols esperados com as finalizações do próprio jogo.
- Árbitro como fator nos cartões; minutos reais de acréscimo.
- Validar a calibração da simulação de temporada (título/Z4) em temporadas passadas.
