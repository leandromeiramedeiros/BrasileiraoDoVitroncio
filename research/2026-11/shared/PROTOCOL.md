# Bancada de pesquisa do Vitrôncio — protocolo comum

Objetivo: reduzir o RPS (ranked probability score, 1X2) das previsões PRÉ-JOGO da Série A do Brasileirão.
Hoje (modelo v1): RPS total 0.2056; régua de mercado (odds de fechamento) 0.2003; chute ingênuo 0.2179.

## Dados compartilhados (somente leitura) em /home/user/bdv-lab/shared/
- `matches.json` — 15.205 partidas (Série A 2006–2026, Série B 2008–2026), em ordem cronológica. Campos:
  `id, lg ('A'|'B'), season, date (ISO UTC), status ('post' encerrada | 'pre' futura | 'off' adiada), h, a (ids de time),
  hg, ag, evOk (eventos batem com o placar), statsOk (estatísticas confiáveis), venue, city,
  ev: [{m: minuto, x: acréscimo, t: 'G' gol | 'P' pênalti | 'O' gol contra (s = time beneficiado) | 'Y' amarelo | 'R' vermelho, s: 'h'|'a', p: id jogador, n: nome, pos}],
  stats: {h:{sh finalizações, sot no alvo, co escanteios, fo faltas, po posse}, a:{...}}` (use stats só se `statsOk`).
- `teams.json` — id -> {name, full, abbr}.
- `baseline_features.csv` — uma linha por partida (A e B), com as previsões pré-jogo do v1 calculadas em walk-forward:
  `lh, la` = gols esperados finais do v1; `g_lh, g_la` = só do rating de gols; `g_Sh, g_Sa` = variância (log) do rating;
  `sh_*, sot_*, co_*, yc_*, fo_*` = finalizações/no alvo/escanteios/amarelos/faltas esperados; `nh, na` = jogos do time na temporada; `mu, hfa` = nível e mando da liga.
- `preds_baseline.csv` — id,pH,pD,pA do v1 (Série A, 2013+, 5.279 jogos).
- `odds_benchmark.csv` — probabilidades implícitas do mercado. **SÓ RÉGUA. PROIBIDO usar como entrada, alvo ou critério de seleção de modelo.**
- `score.py` — placar oficial: `/home/user/bdv-lab/venv/bin/python /home/user/bdv-lab/shared/score.py SEU_ARQUIVO.csv`
- Código do v1 (para ler, não editar): /home/user/brasileiraodovitroncio/site/js/engine/{ratings,match-model}.js e tools/refit.mjs.
- Downloads brutos da ESPN: /home/user/brasileiraodovitroncio/model/cache/ (scoreboards anuais; `sum/` = summaries 2026 com escalações).
- Python pronto: /home/user/bdv-lab/venv/bin/python (numpy, pandas, scipy, scikit-learn, lightgbm). Node 22 disponível. Pode instalar pacotes no venv.

## Regras (valem para todos)
1. **Sem vazamento.** A previsão de um jogo só pode usar informação disponível antes do pontapé inicial.
   Qualquer parâmetro AJUSTADO (coeficientes, árvores, hiperparâmetros escolhidos por otimização) usado para prever a
   temporada Y deve ser estimado só com jogos anteriores à temporada Y (janela expansiva por temporada) — ou ser fixo a priori.
2. **DEV = 2013–2021, HOLDOUT = 2022–2026.** Escolha de modelo/atributos/hiperparâmetros olhando só o DEV.
   O HOLDOUT é para o relatório final; não itere em cima dele (no máximo 2–3 consultas).
   Atenção: os hiperparâmetros do v1 foram ajustados com 2012–2026 inteiro, então o baseline é levemente otimista —
   empatar com ele usando procedimento limpo já é ganho real.
3. Ganho real = dRPS negativo no DEV **e** no HOLDOUT, de preferência com IC95 abaixo de zero. Ganhos < 0.0003 são ruído.
4. Odds de mercado não entram no modelo de forma alguma (o site vai comparar o modelo com o mercado; ele precisa ser independente).
5. Trabalhe só dentro do SEU diretório (/home/user/bdv-lab/<sua-trilha>/). Não edite o repositório nem shared/.
   A máquina tem 4 núcleos compartilhados com outras trilhas: use no máximo 2 processos por vez.
6. Rede: a ESPN (site.api.espn.com, site.web.api.espn.com, sports.core.api.espn.com) responde a `curl` simples.
   Não contorne bloqueios de sites nem altere cabeçalhos para fingir ser navegador; se um site negar acesso, registre e siga.

## Entrega (no seu diretório)
- `preds_<nome>.csv` (id,pH,pD,pA para os 5.279 jogos) de cada variante que valha a pena, gerado em walk-forward limpo.
- `features_<nome>.csv` (id + colunas pré-jogo) para qualquer atributo que possa alimentar um empilhamento final.
- `REPORT.md`: o que foi testado, tabela DEV/HOLDOUT (RPS, LL, dRPS com IC) por variante, o que funcionou, o que NÃO funcionou,
  e como rodar o código de novo. Diga também o que seria necessário para gerar os mesmos atributos para jogos FUTUROS.
- Resposta final ao orquestrador: resumo curto com os números e os caminhos dos arquivos.
