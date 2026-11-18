// "Como o Vitrôncio pensa": explicação do modelo + números de validação.
import { S } from '../store.js';
import { esc, pct, num, fmtFull } from '../util.js';
import { RESEARCH, REFS, DATA } from './refs.js';

function calibChart(cal) {
  const W = 360, H = 300, pad = 36;
  const x = (v) => pad + v * (W - pad - 24), y = (v) => H - pad - v * (H - pad - 10);
  const pts = cal.filter(Boolean);
  const mx = Math.max(...pts.map((p) => p.n));
  return `<svg class="chart big" viewBox="0 0 ${W} ${H}" style="max-width:420px" role="img" aria-label="Calibração">
    <line x1="${x(0)}" y1="${y(0)}" x2="${x(1)}" y2="${y(1)}" stroke="rgba(255,255,255,.3)" stroke-dasharray="4 4"/>
    ${[0, 0.25, 0.5, 0.75, 1].map((v) => `<text x="${x(v)}" y="${H - 14}" text-anchor="middle">${v * 100}%</text><text x="${pad - 6}" y="${y(v) + 4}" text-anchor="end">${v * 100}%</text><line x1="${x(v)}" x2="${x(v)}" y1="${y(0)}" y2="${y(1)}" stroke="rgba(255,255,255,.05)"/><line x1="${x(0)}" x2="${x(1)}" y1="${y(v)}" y2="${y(v)}" stroke="rgba(255,255,255,.05)"/>`).join('')}
    <path d="${pts.map((p, i) => `${i ? 'L' : 'M'}${x(p.previsto).toFixed(1)},${y(p.observado).toFixed(1)}`).join('')}" fill="none" stroke="var(--gold)" stroke-width="2"/>
    ${pts.map((p) => `<circle cx="${x(p.previsto)}" cy="${y(p.observado)}" r="${3 + 5 * Math.sqrt(p.n / mx)}" fill="var(--gold)" opacity=".85"><title>previsto ${pct(p.previsto, 1)} · aconteceu ${pct(p.observado, 1)} · ${p.n} casos</title></circle>`).join('')}
    <text x="${W / 2}" y="${H - 1}" text-anchor="middle">probabilidade prevista</text></svg>`;
}
function barChart(vals, labels, color = 'var(--gold)', fmt = (v) => v.toFixed(2)) {
  const W = 520, H = 190, pad = { l: 8, r: 8, t: 18, b: 34 }, n = vals.length, bw = (W - pad.l - pad.r) / n, mx = Math.max(...vals);
  return `<svg class="chart big" viewBox="0 0 ${W} ${H}" style="max-width:560px">${vals.map((v, i) => { const h = (v / mx) * (H - pad.t - pad.b); return `<rect x="${pad.l + i * bw + 5}" y="${H - pad.b - h}" width="${bw - 10}" height="${h}" rx="4" fill="${color}" opacity=".9"/><text x="${pad.l + i * bw + bw / 2}" y="${H - pad.b - h - 5}" text-anchor="middle" style="fill:var(--text)">${fmt(v)}</text><text x="${pad.l + i * bw + bw / 2}" y="${H - 14}" text-anchor="middle">${labels[i]}</text>`; }).join('')}</svg>`;
}

export function modelView() {
  const M = S.model, bt = M.backtest, g = bt.geral, hz = M.hazard;
  const perMin = hz.goal.bin.map((b) => Math.exp(b));
  const binLab = ["1–15'", "16–30'", "31–45'", '45+', "46–60'", "61–75'", "76–90'", '90+'];
  const e = (v) => `${v >= 0 ? '+' : ''}${((Math.exp(v) - 1) * 100).toFixed(0)}%`;
  const seasons = Object.entries(bt.temporadas);
  return `<div class="wrap prose">
    <section data-mk="vith"><div class="vit-hero"><img src="img/vitroncio-olho.webp" width="345" height="680" alt="Victor, o Vitrôncio, olhando por cima dos óculos"><div><h2>Como o Vitrôncio pensa</h2>
      <p>O modelo leva o nome do Victor, o Vitrôncio. Todas as probabilidades do site vêm dele: um modelo estatístico próprio, que não usa odds de casas de aposta. Não é recomendação de aposta.</p></div></div>
      <div class="kv">
        <div><div class="k">Partidas na base</div><div class="v">${M.base.partidas.toLocaleString('pt-BR')}</div></div>
        <div><div class="k">Séries A e B</div><div class="v">desde ${M.base.desde}</div></div>
        <div><div class="k">Gols e cartões com minuto</div><div class="v">${M.base.eventos.toLocaleString('pt-BR')}</div></div>
        <div><div class="k">Último reajuste</div><div class="v s">${esc(fmtFull(M.geradoEm))}</div></div>
      </div>
      <h3>1. Força dos times</h3>
      <p>Cada time tem uma força de ataque, uma de defesa e um fator casa, com a incerteza de cada uma. Depois de cada partida, um filtro de Kalman estendido (modelo de espaço de estados com gols de Poisson) compara o resultado com o esperado e corrige as forças: mais para times pouco conhecidos, menos para times estáveis. O resultado observado combina gols (${pct(M.params.g.w)}) e finalizações (${pct(1 - M.params.g.w)}); gols com um jogador a mais valem ${pct(M.params.g.wRed)}.${M.params.x ? ` Um segundo rating de gols corre em paralelo olhando o xG medido chute a chute (posição e tipo de cada finalização, Séries A e B desde 2014), com peso de ${pct(1 - M.params.x.w)} contra ${pct(M.params.x.w)} do placar; os dois são combinados na previsão final.` : ''}</p>
      <p>A Série B entra para que os promovidos já cheguem avaliados, com desconto de ${e(-M.params.g.dPromoO)} no ataque e ${e(-M.params.g.dPromoD)} na defesa. A cada virada de temporada, as forças recuam ${pct(1 - M.params.g.rho)} em direção à média. Há ratings equivalentes para finalizações, escanteios, cartões e faltas.</p>
      <h3>2. A partida</h3>
      <p>Com os gols esperados de cada lado, a partida é uma cadeia de Markov de ${90 + hz.S1 + hz.S2} minutos. A probabilidade de gol em cada minuto depende do tempo de jogo, do placar e de expulsões, com efeitos estimados em ${hz.nMatches.toLocaleString('pt-BR')} partidas. As taxas de cada lado são ajustadas para que a cadeia reproduza exatamente os gols esperados (sem isso, o azarão de um jogo desigual ganharia uns 6% a mais). A distribuição do placar é calculada de forma exata, e o mesmo cálculo roda ao vivo a partir do minuto e do placar atuais.</p>
      <div class="duo"><div class="panel"><div class="sec-h"><h3>Ritmo de gols por período</h3></div>${barChart(perMin, binLab)}<p class="small muted">Intensidade relativa por minuto (1,00 = média). Os acréscimos duram mais do que o relógio indica.</p></div>
      <div class="panel"><div class="sec-h"><h3>Efeito sobre a chance de gol</h3></div><table>
        <tr><td class="l">Perdendo por 1</td><td class="pts">${e(hz.goal.diff[1])}</td></tr><tr><td class="l">Perdendo por 2 ou mais</td><td class="pts">${e(hz.goal.diff[0])}</td></tr>
        <tr><td class="l">Ganhando por 1</td><td class="pts">${e(hz.goal.diff[3])}</td></tr><tr><td class="l">Ganhando por 2 ou mais</td><td class="pts">${e(hz.goal.diff[4])}</td></tr>
        <tr><td class="l">Com um jogador a mais</td><td class="pts">${e(hz.goal.men[2])}</td></tr><tr><td class="l">Com um jogador a menos</td><td class="pts">${e(hz.goal.men[0])}</td></tr></table></div></div>
      ${M.ctx ? `<h3>3. Contexto do jogo</h3>
      <p>Quatro situações mudam os gols esperados além da força dos times. Os efeitos foram estimados nos jogos da Série A desde 2009 e só ficaram os que melhoraram a previsão em temporadas não usadas na escolha.</p>
      <table style="max-width:560px">
        <tr><td class="l">Time já rebaixado ou já campeão (matematicamente)</td><td class="pts">${e(M.ctx.dec)}</td></tr>
        <tr><td class="l">Jogo de taça continental em até 4 dias depois</td><td class="pts">${e(M.ctx.nc)}</td></tr>
        <tr><td class="l">Mando a mais de 100 km da sede (vantagem de casa)</td><td class="pts">${e(M.ctx.disp)}</td></tr>
        <tr><td class="l">Portões fechados, 2020–21 (vantagem de casa)</td><td class="pts">${e(M.ctx.closed)}</td></tr></table>
      ${M.ctx.lineup ? `<p>Quando a escalação sai (cerca de uma hora antes do jogo), o modelo mede quanto do time habitual de cada lado começa jogando, quanto dos gols e assistências recentes do time está em campo e quantos titulares habituais estão no banco. Cada 10 pontos percentuais a menos do time habitual em campo valem ${e(-0.1 * M.ctx.lineup.usage)} nos gols esperados do time.</p>` : ''}
      ${M.ctx.absent ? `<p>Antes de a escalação sair, entram os desfalques que já se sabem: suspensos por cartões (expulsão ou terceiro amarelo, pela regra da CBF) e titulares habituais que ficaram fora do último jogo, quase sempre por lesão. Cada titular suspenso tira cerca de ${e(-0.09 * M.ctx.absent.susp_w).replace('+', '')} dos gols esperados do time.</p>` : ''}
      <p class="small muted">Efeito nos gols esperados do próprio time; o adversário ganha o equivalente.</p>` : ''}
      <h3>${M.ctx ? 4 : 3}. O campeonato</h3>
      <p>Título, Libertadores e rebaixamento saem de milhares de simulações dos jogos restantes, com um desvio de força sorteado por time em cada simulação e os critérios de desempate da CBF. Com jogos em andamento, a simulação parte do placar e do minuto atuais.</p>
      <h3>${M.ctx ? 5 : 4}. Atualização</h3>
      <p>O modelo é reajustado por completo uma vez por semana. Entre reajustes, o navegador busca os resultados novos e aplica a mesma atualização de forças.</p>
    </section>
    <section><div class="sec-h"><h2>Validação</h2><span class="sub">cada jogo da Série A desde ${bt.desde}, previsto apenas com informação anterior a ele</span></div>
      <div class="kv">
        <div><div class="k">Jogos testados</div><div class="v">${g.n.toLocaleString('pt-BR')}</div></div>
        <div><div class="k">Favorito venceu</div><div class="v">${pct(g.acerto, 1)}</div></div>
        <div><div class="k">Placar exato</div><div class="v">${pct(bt.placarExato.top1, 1)}</div></div>
        <div><div class="k">Placar entre os 3 mais prováveis</div><div class="v">${pct(bt.placarExato.top3, 1)}</div></div>
        <div><div class="k">RPS do modelo</div><div class="v">${g.rps.toFixed(4)}</div></div>
        ${g.rpsSemEscalacao && g.rpsSemEscalacao !== g.rps ? `<div><div class="k">RPS sem a escalação</div><div class="v">${g.rpsSemEscalacao.toFixed(4)}</div></div>` : ''}
        <div><div class="k">RPS de referência simples</div><div class="v">${g.rpsBase.toFixed(4)}</div></div>
        ${g.rpsRef ? `<div><div class="k">RPS do mercado</div><div class="v">${g.rpsRef.toFixed(4)}</div></div>` : ''}
      </div>
      <p class="small muted">RPS mede o erro das probabilidades: quanto menor, melhor.${g.rpsSemEscalacao && g.rpsSemEscalacao !== g.rps ? ' O RPS do modelo usa a escalação anunciada de cada jogo; dias antes do jogo, sem ela, o erro esperado é o da coluna "sem a escalação".' : ''} A referência simples usa sempre as frequências históricas de vitória do mandante, empate e visitante.${g.rpsRef ? ` O modelo cobre ${pct((g.rpsBase - g.rps) / (g.rpsBase - g.rpsRef))} da distância entre essa referência e o mercado.` : ''}</p>
      <div class="duo"><div class="panel"><div class="sec-h"><h3>Calibração</h3></div>${calibChart(bt.calibracao)}<p class="small muted">Probabilidade prevista (horizontal) contra frequência observada (vertical). Sobre a diagonal, o modelo está calibrado.</p></div>
      <div class="panel"><div class="sec-h"><h3>Total de gols por jogo</h3><span class="sub">previsto e real</span></div><table><thead><tr><th class="l">Gols</th><th>Previsto</th><th>Real</th></tr></thead><tbody>${bt.totalGols.previsto.map((v, k) => `<tr><td class="l">${k}${k === 8 ? ' ou mais' : ''}</td><td>${pct(v)}</td><td>${pct(bt.totalGols.observado[k])}</td></tr>`).join('')}</tbody></table></div></div>
      <div class="panel" style="margin-top:12px"><div class="sec-h"><h3>Por temporada</h3><span class="sub">Série A; favorito = time com a maior chance venceu</span></div><div class="tw"><table><thead><tr><th>Ano</th><th>Jogos</th><th>Favorito venceu</th><th>RPS</th><th><span class="nm-full">Referência simples</span><span class="nm-abbr">Ref.</span></th></tr></thead><tbody>${seasons.map(([s, a]) => `<tr><td>${s}</td><td>${a.n}</td><td>${pct(a.acerto)}</td><td>${a.rps.toFixed(4)}</td><td>${a.rpsBase.toFixed(4)}</td></tr>`).join('')}</tbody></table></div></div>
    </section>
    <section><div class="sec-h"><h2>Limitações</h2></div><ul>
      <li>Lesões entram só de forma indireta (titular habitual fora do último jogo), e jogadores poupados só aparecem quando a escalação sai, perto do apito.</li>
      <li>Aprende só com jogos das Séries A e B; das taças continentais usa apenas o calendário.</li>
      <li>Marcadores prováveis e dados de jogadores são atualizados só no reajuste semanal.</li>
      <li>Os dados ao vivo vêm de um serviço público da ESPN e podem atrasar ou mudar sem aviso.</li>
      <li>Não serve para apostar. As odds de fechamento erram menos que o modelo${g.rpsRef ? ` (RPS ${g.rpsRef.toFixed(4)} contra ${g.rps.toFixed(4)})` : ''}, e a casa ainda cobra margem. Entre 2013 e 2026, apostar nos resultados em que o modelo via preço bom teria perdido cerca de 6% do valor apostado.</li></ul></section>
    <section><div class="sec-h"><h2>Pesquisa</h2><span class="sub">novembro de 2026: o que foi testado para reduzir o erro</span></div>
      <p class="small muted">Cada ideia foi escolhida olhando só as temporadas de 2013 a 2021 e conferida depois em 2022–2026. Efeito: mudança no RPS de 2022–2026 (negativo é melhor).</p>
      <div class="tw"><table class="research"><tr><th class="l">Ideia</th><th class="l">Situação</th><th class="l">Efeito</th></tr>${RESEARCH.map(([a, b, c]) => `<tr><td class="l">${esc(a)}</td><td class="l small">${esc(b)}</td><td class="l pts">${esc(c)}</td></tr>`).join('')}</table></div></section>
    <section><div class="sec-h"><h2>Referências</h2><span class="sub">livros, artigos e dados por trás do modelo e dos testes</span></div>
      ${REFS.map(([grp, list]) => `<h3>${esc(grp)}</h3><ul class="refs">${list.map((r) => `<li><a href="https://scholar.google.com/scholar?q=${encodeURIComponent(`"${r.title}"`)}" target="_blank" rel="noopener">${esc(r.authors)} (${r.year}). <i>${esc(r.title)}</i>.</a> ${esc(r.venue)}. <span class="dim">${esc(r.use)}.</span></li>`).join('')}</ul>`).join('')}
      <h3>Dados</h3><ul class="refs">${DATA.map(([a, b]) => `<li><b>${esc(a)}</b>: ${esc(b)}.</li>`).join('')}</ul></section>
  </div>`;
}
