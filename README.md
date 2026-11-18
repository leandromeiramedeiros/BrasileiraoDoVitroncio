# Brasileirão do Vitrôncio

Série A do Campeonato Brasileiro ao vivo, com as chances de cada jogo e do campeonato calculadas pelo **Vitrôncio**, um simulador estatístico feito por torcedores, para torcedores. Não é ferramenta de aposta.

**Site:** https://leandromeiramedeiros.github.io/BrasileiraoDoVitroncio/

## O que tem no site

| Página | O que mostra |
| --- | --- |
| Início | Jogo ao vivo em destaque (placar gigante, relógio, lances, estatísticas, chance de vitória minuto a minuto). Com mais de um jogo rolando, cartões para trocar. Próximos jogos com previsão, a corrida pelo título/G4/Z4 e os últimos resultados. |
| Jogo | Previsão completa (mapa de placares, gols esperados, quem marca, escanteios, cartões) e o botão **Assistir simulação**: uma versão sorteada do jogo, narrada minuto a minuto. |
| Jogos | Calendário completo da temporada, por mês ou por time. |
| Artilharia | Artilheiros, assistências, finalizações, cartões e minutos. |
| Classificação | Tabela em tempo real + chances de título, Libertadores, Sul-Americana e rebaixamento. |
| Simulador | Confronto livre entre quaisquer dois times e o "E se…?" (trave resultados futuros e veja a corrida mudar). |
| Times | Ranking de força (ataque/defesa), página de cada clube com evolução, jogos e elenco. |
| O modelo | Como o Vitrôncio pensa e o quanto ele acerta (backtest). |

## Como funciona (resumo)

1. **Ratings dinâmicos** — modelo de espaço de estados com gols de Poisson, estimado por filtro de Kalman estendido, partida a partida, sobre ~15 mil jogos das Séries A e B desde 2006. Ataque, defesa, fator casa e a incerteza de cada um; o "resultado" mistura gols e um proxy de gols esperados a partir das finalizações. Ratings paralelos para finalizações, escanteios, cartões e faltas.
2. **Modelo de partida** — cadeia de Markov de 100 passos (minutos + acréscimos). A intensidade de gol depende do tempo de jogo, do placar e de expulsões, com efeitos estimados em ~9 mil partidas com eventos minuto a minuto. A distribuição do placar é calculada de forma exata (programação dinâmica), tanto no pré-jogo quanto ao vivo.
3. **Temporada** — Monte Carlo dos jogos restantes com incerteza de força por time e critérios de desempate da CBF, rodando em Web Worker.
4. **Dinâmico** — o navegador busca os resultados na ESPN (endpoint público, sem chave) e aplica a mesma atualização de ratings a cada jogo encerrado depois da última "foto" do modelo.

Detalhes e números de validação na página **O modelo** do site.

## Estrutura

```
site/                 site estático publicado (GitHub Pages)
  index.html, css/
  js/engine/          motor de simulação (JS puro, sem DOM) — roda no navegador e no Node
  js/data/            parser do formato da ESPN
  js/app/             interface (roteador, store, páginas)
  data/model.json     "foto" semanal do modelo (ratings, intensidades, jogadores, backtest)
tools/                reajuste semanal (Node, sem dependências)
model/data/           base histórica compacta (history.json.gz)
model/cache/          downloads brutos (fora do git)
tests/                testes do motor
```

## Reajuste semanal

```bash
npm run weekly     # baixa dados novos, refaz a base, reajusta o modelo e roda os testes
git add -A && git commit -m "Reajuste semanal" && git push
```

Passo a passo: `fetch` (ESPN, Séries A e B) → `dataset` (base compacta) → `fetch-summaries` (escalações e números por jogador) → `refit` (hiperparâmetros, intensidades, backtest, `site/data/model.json`) → testes. `node tools/refit.mjs --quick` reaproveita os hiperparâmetros atuais.

## Rodar localmente

```bash
npm run serve      # http://localhost:8080
npm test
```

Não há etapa de build nem dependências: são módulos ES servidos direto.

## Publicação

Cada push em `main` roda os testes e publica a pasta `site/` no branch `gh-pages` (`.github/workflows/deploy.yml`). Em **Settings → Pages**, a fonte deve ser *Deploy from a branch* → `gh-pages` / `(root)`.

## Créditos e avisos

Dados de jogo: ESPN (serviço público, não oficial — pode mudar sem aviso). Escudos pertencem aos respectivos clubes. As probabilidades são estimativas de um modelo estatístico, para diversão.
