// "Como o Vitrôncio pensa": explicação do modelo + números de validação.
import { S } from '../store.js';
import { esc, pct, num, fmtFull } from '../util.js';

function calibChart(cal) {
  const W = 360, H = 300, pad = 36;
  const x = (v) => pad + v * (W - pad - 10), y = (v) => H - pad - v * (H - pad - 10);
  const pts = cal.filter(Boolean);
  const mx = Math.max(...pts.map((p) => p.n));
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" style="max-width:420px" role="img" aria-label="Calibração">
    <line x1="${x(0)}" y1="${y(0)}" x2="${x(1)}" y2="${y(1)}" stroke="rgba(255,255,255,.3)" stroke-dasharray="4 4"/>
    ${[0, 0.25, 0.5, 0.75, 1].map((v) => `<text x="${x(v)}" y="${H - 14}" text-anchor="middle">${v * 100}%</text><text x="${pad - 6}" y="${y(v) + 4}" text-anchor="end">${v * 100}%</text><line x1="${x(v)}" x2="${x(v)}" y1="${y(0)}" y2="${y(1)}" stroke="rgba(255,255,255,.05)"/><line x1="${x(0)}" x2="${x(1)}" y1="${y(v)}" y2="${y(v)}" stroke="rgba(255,255,255,.05)"/>`).join('')}
    <path d="${pts.map((p, i) => `${i ? 'L' : 'M'}${x(p.previsto).toFixed(1)},${y(p.observado).toFixed(1)}`).join('')}" fill="none" stroke="var(--gold)" stroke-width="2"/>
    ${pts.map((p) => `<circle cx="${x(p.previsto)}" cy="${y(p.observado)}" r="${3 + 5 * Math.sqrt(p.n / mx)}" fill="var(--gold)" opacity=".85"><title>previsto ${pct(p.previsto, 1)} · aconteceu ${pct(p.observado, 1)} · ${p.n} casos</title></circle>`).join('')}
    <text x="${W / 2}" y="${H - 1}" text-anchor="middle">o que o Vitrôncio previu</text></svg>`;
}
function barChart(vals, labels, color = 'var(--gold)', fmt = (v) => v.toFixed(2)) {
  const W = 520, H = 190, pad = { l: 8, r: 8, t: 18, b: 34 }, n = vals.length, bw = (W - pad.l - pad.r) / n, mx = Math.max(...vals);
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" style="max-width:560px">${vals.map((v, i) => { const h = (v / mx) * (H - pad.t - pad.b); return `<rect x="${pad.l + i * bw + 5}" y="${H - pad.b - h}" width="${bw - 10}" height="${h}" rx="4" fill="${color}" opacity=".9"/><text x="${pad.l + i * bw + bw / 2}" y="${H - pad.b - h - 5}" text-anchor="middle" style="fill:var(--text)">${fmt(v)}</text><text x="${pad.l + i * bw + bw / 2}" y="${H - 14}" text-anchor="middle">${labels[i]}</text>`; }).join('')}</svg>`;
}

export function modelView() {
  const M = S.model, bt = M.backtest, g = bt.geral, hz = M.hazard;
  const perMin = hz.goal.bin.map((b) => Math.exp(b));
  const binLab = ["1–15'", "16–30'", "31–45'", '45+', "46–60'", "61–75'", "76–90'", '90+'];
  const e = (v) => `${v >= 0 ? '+' : ''}${((Math.exp(v) - 1) * 100).toFixed(0)}%`;
  const seasons = Object.entries(bt.temporadas);
  return `<div class="wrap prose">
    <section><div class="sec-h"><h2>Como o Vitrôncio pensa</h2></div>
      <p>O Vitrôncio é o motor de simulação deste site. Ele não é ferramenta de aposta: é um brinquedo estatístico sério, feito por e para quem gosta de futebol. Tudo o que aparece como "chance" aqui sai dele.</p>
      <div class="kv">
        <div><div class="k">Partidas estudadas</div><div class="v">${M.base.partidas.toLocaleString('pt-BR')}</div></div>
        <div><div class="k">Série A + Série B</div><div class="v">desde ${M.base.desde}</div></div>
        <div><div class="k">Gols e cartões com minuto</div><div class="v">${M.base.eventos.toLocaleString('pt-BR')}</div></div>
        <div><div class="k">Último reajuste</div><div class="v" style="font-size:16px">${esc(fmtFull(M.geradoEm))}</div></div>
      </div>
      <h3>1. Força de cada time, jogo a jogo</h3>
      <p>Cada time tem uma força de <b>ataque</b> e uma de <b>defesa</b>, mais um fator casa próprio, e o modelo guarda também o quanto ele está <b>incerto</b> sobre cada número. A cada partida, um filtro de Kalman estendido (modelo de espaço de estados com gols de Poisson) compara o que aconteceu com o que era esperado e corrige as forças — muito para times que ele conhece pouco (recém-promovidos, início de temporada), pouco para times estáveis. O "resultado" não é só o placar: ${pct(1 - M.params.g.w)} do peso vem das finalizações (um proxy de gols esperados), e gols marcados com um jogador a mais valem ${pct(M.params.g.wRed)} de um gol normal.</p>
      <p>A Série B entra na conta para o modelo já conhecer quem sobe. Quem é promovido chega com desconto (ataque ${e(-M.params.g.dPromoO)}, defesa ${e(-M.params.g.dPromoD)} em relação ao que mostrava na B), e a cada virada de ano todo mundo é puxado ${pct(1 - M.params.g.rho)} de volta para a média.</p>
      <p>Os mesmos ratings existem para finalizações, chutes no alvo, escanteios, cartões e faltas. Uma camada final combina o rating de gols com os de finalizações para chegar aos <b>gols esperados</b> de cada lado.</p>
      <h3>2. O jogo, minuto a minuto</h3>
      <p>Com os gols esperados em mãos, o jogo vira uma cadeia de Markov de ${90 + hz.S1 + hz.S2} passos (90 minutos mais acréscimos). A cada minuto, a chance de gol depende do tempo de jogo, do placar e de quem tem mais gente em campo — tudo estimado em ${hz.nMatches.toLocaleString('pt-BR')} partidas com eventos minuto a minuto. A distribuição do placar final é calculada de forma <b>exata</b> (programação dinâmica), e é o mesmo cálculo que roda ao vivo a partir do minuto e do placar atuais.</p>
      <div class="duo"><div class="panel"><div class="sec-h"><h3 style="margin:0;color:var(--text)">Ritmo de gols por faixa do jogo</h3></div>${barChart(perMin, binLab)}<p class="small muted">Intensidade relativa por minuto (1,00 = média do jogo). Os acréscimos são mais "quentes" porque duram mais do que o relógio mostra.</p></div>
      <div class="panel"><div class="sec-h"><h3 style="margin:0;color:var(--text)">O que muda a chance de gol</h3></div><table>
        <tr><td class="l">Perdendo por 1</td><td class="pts">${e(hz.goal.diff[1])}</td></tr><tr><td class="l">Perdendo por 2 ou mais</td><td class="pts">${e(hz.goal.diff[0])}</td></tr>
        <tr><td class="l">Ganhando por 1</td><td class="pts">${e(hz.goal.diff[3])}</td></tr><tr><td class="l">Ganhando por 2 ou mais</td><td class="pts">${e(hz.goal.diff[4])}</td></tr>
        <tr><td class="l">Com um jogador a mais</td><td class="pts">${e(hz.goal.men[2])}</td></tr><tr><td class="l">Com um jogador a menos</td><td class="pts">${e(hz.goal.men[0])}</td></tr>
        <tr><td class="l">Visitante: risco de expulsão</td><td class="pts">${e(hz.red.away)}</td></tr></table></div></div>
      <h3>3. O campeonato inteiro</h3>
      <p>Para título, Libertadores e rebaixamento, o resto do campeonato é jogado dezenas de milhares de vezes. Em cada "temporada paralela" cada time ganha um pequeno desvio de forma sorteado (a incerteza do rating), os placares são sorteados das matrizes exatas de cada jogo e a tabela é fechada com os critérios da CBF. Com jogo em andamento, a simulação já parte do placar e do minuto atuais.</p>
      <h3>4. Dinâmico entre reajustes</h3>
      <p>Uma vez por semana o algoritmo é reajustado por completo (hiperparâmetros, intensidades, jogadores). Entre um reajuste e outro, o seu navegador busca os resultados novos e aplica a mesma atualização de ratings — as chances que você vê já consideram o último jogo encerrado.</p>
    </section>
    <section><div class="sec-h"><h2>Ele acerta?</h2><span class="sub">teste honesto: prever cada jogo da Série A desde ${bt.desde} usando só o que se sabia antes dele</span></div>
      <div class="kv">
        <div><div class="k">Jogos testados</div><div class="v">${g.n.toLocaleString('pt-BR')}</div></div>
        <div><div class="k">Favorito confirmou</div><div class="v">${pct(g.acerto, 1)}</div></div>
        <div><div class="k">Placar exato (1º palpite)</div><div class="v">${pct(bt.placarExato.top1, 1)}</div></div>
        <div><div class="k">Placar entre os 3 mais prováveis</div><div class="v">${pct(bt.placarExato.top3, 1)}</div></div>
        <div><div class="k">Erro RPS do Vitrôncio</div><div class="v">${g.rps.toFixed(4)}</div></div>
        <div><div class="k">RPS de um chute "na média"</div><div class="v">${g.rpsBase.toFixed(4)}</div></div>
        ${g.rpsRef ? `<div><div class="k">RPS de referência (mercado)</div><div class="v">${g.rpsRef.toFixed(4)}</div></div>` : ''}
      </div>
      <p class="small muted">RPS mede o erro das probabilidades (quanto menor, melhor). ${g.rpsRef ? `O mercado profissional é a régua mais dura que existe: o Vitrôncio fecha ${pct((g.rpsBase - g.rps) / (g.rpsBase - g.rpsRef))} da distância entre o chute ingênuo e essa régua usando só resultados e estatísticas de jogo.` : ''}</p>
      <div class="duo"><div class="panel"><div class="sec-h"><h3 style="margin:0;color:var(--text)">Calibração</h3></div>${calibChart(bt.calibracao)}<p class="small muted">Quando ele diz 60%, acontece perto de 60%? Pontos sobre a diagonal = sim.</p></div>
      <div class="panel"><div class="sec-h"><h3 style="margin:0;color:var(--text)">Total de gols por jogo</h3></div><table><tr><th class="l">Gols</th>${bt.totalGols.previsto.map((_, k) => `<th>${k}${k === 8 ? '+' : ''}</th>`).join('')}</tr><tr><td class="l">Previsto</td>${bt.totalGols.previsto.map((v) => `<td>${pct(v)}</td>`).join('')}</tr><tr><td class="l">Real</td>${bt.totalGols.observado.map((v) => `<td>${pct(v)}</td>`).join('')}</tr></table>
        <div class="sec-h" style="margin-top:16px"><h3 style="margin:0;color:var(--text)">Por temporada</h3></div><div class="tw" style="max-height:260px;overflow:auto"><table><tr><th>Ano</th><th>Jogos</th><th>Favorito</th><th>RPS</th><th>Ingênuo</th></tr>${seasons.map(([s, a]) => `<tr><td>${s}</td><td>${a.n}</td><td>${pct(a.acerto)}</td><td>${a.rps.toFixed(4)}</td><td>${a.rpsBase.toFixed(4)}</td></tr>`).join('')}</table></div></div></div>
    </section>
    <section><div class="sec-h"><h2>Limites conhecidos</h2></div><ul>
      <li>Ele não sabe de lesões, suspensões, poupados nem de reforços que ainda não jogaram — só vê o que acontece em campo.</li>
      <li>Só enxerga Série A e Série B: jogos de Copa do Brasil, Libertadores e estaduais ainda não entram.</li>
      <li>"Quem marca" usa a escalação provável; os dados de jogadores são da foto semanal.</li>
      <li>Os dados ao vivo vêm de um serviço público da ESPN, que pode atrasar alguns segundos ou mudar sem aviso.</li></ul></section>
  </div>`;
}
