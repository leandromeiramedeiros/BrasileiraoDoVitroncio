// Vitrôncio × Mercado: o que a simulação diz, o que as odds dizem e, nos
// jogos já disputados, o que de fato aconteceu. As odds NÃO entram no modelo —
// por isso a comparação é justa. Não é recomendação de aposta.
import { S, team, upcoming, liveMatches, finished, forecast, market, loadOdds, rpsOf, outcomeProb, teamIds } from '../store.js';
import { esc, pct, num, fmtShort, fmtTime } from '../util.js';
import { logo, probBar, colors, fav } from '../components.js';
import { face } from '../vitroncio.js';

export const cmpState = { team: '', n: 40 };
const dec = (v) => v.toFixed(2).replace('.', ',');
const resultTxt = (m) => (m.hg > m.ag ? `vitória do ${team(m.h).name}` : m.hg < m.ag ? `vitória do ${team(m.a).name}` : 'empate');

/** Maior divergência entre modelo e mercado, em pontos percentuais. */
function gap(m, p, k) {
  const d = [['pH', team(m.h).name], ['pD', 'empate'], ['pA', team(m.a).name]].map(([key, lab]) => ({ lab, v: p[key] - k[key] })).sort((a, b) => Math.abs(b.v) - Math.abs(a.v))[0];
  return d;
}

/** Bloco usado na página do jogo. */
export function versusBlock(m) {
  if (String(m.id).startsWith('sim-')) return '';
  const k = market(m);
  if (!k) { if (m.status !== 'pre') loadOdds([m]); return `<section class="panel" data-mk="versus"><div class="sec-h"><h3>Vitrôncio × mercado</h3></div><p class="muted small">${m.status === 'pre' ? 'As odds deste jogo ainda não foram publicadas (costumam aparecer alguns dias antes).' : 'Buscando as odds de fechamento deste jogo…'}</p></section>`; }
  const f = forecast(m), p = f.pre, played = m.status === 'post';
  const g = gap(m, p, k);
  const row = (lab, q, extra) => `<div class="vs-row"><div class="vs-lab">${lab}</div><div>${probBar(q)}<div class="plabels"><span><b>${pct(q.pH)}</b> ${esc(team(m.h).abbr)}</span><span>empate <b>${pct(q.pD)}</b></span><span>${esc(team(m.a).abbr)} <b>${pct(q.pA)}</b></span></div>${extra || ''}</div></div>`;
  let verdict = '';
  if (played) {
    const rm = rpsOf(p, m), rk = rpsOf(k, m), pm = outcomeProb(p, m), pk = outcomeProb(k, m);
    const who = Math.abs(rm - rk) < 0.002 ? 'Empate.' : rm < rk ? 'Vitrôncio mais preciso.' : 'Mercado mais preciso.';
    verdict = `<div class="kv" style="margin-top:12px"><div><div class="k">Aconteceu</div><div class="v s">${m.hg}–${m.ag}, ${esc(resultTxt(m))}</div></div><div><div class="k">Vitrôncio dava</div><div class="v">${pct(pm)}</div></div><div><div class="k">Mercado dava</div><div class="v">${pct(pk)}</div></div><div><div class="k">Neste jogo</div><div class="v s">${who}</div></div></div>`;
  }
  return `<section class="panel" data-mk="versus" style="${colors(m)}"><div class="sec-h"><h3>Vitrôncio × mercado</h3><span class="sub">${played ? 'antes do jogo' : 'o Vitrôncio não usa as odds'}</span><a class="more" href="#/contra/comparar">todos os jogos</a></div>
    ${row(`${face(18)}Vitrôncio`, p)}
    ${row('Mercado', k, `<div class="small dim" style="margin-top:4px">odds: ${dec(k.dec[0])} · ${dec(k.dec[1])} · ${dec(k.dec[2])} (${esc(k.src)}${k.kind === 'pre' ? ', agora' : k.src.includes('fechamento') ? '' : ', fechamento'}) · margem da casa retirada</div>`)}
    ${played ? verdict : `<p class="small muted" style="margin:12px 0 0">Maior diferença: o Vitrôncio dá <b style="color:var(--text)">${g.v > 0 ? '+' : ''}${(g.v * 100).toFixed(0)} pontos percentuais</b> para ${esc(g.lab)} em relação ao mercado.</p>`}
  </section>`;
}

function cumChart(rows) {
  if (rows.length < 5) return '';
  const W = 720, H = 180, pad = { l: 44, r: 10, t: 12, b: 22 };
  let acc = 0; const pts = rows.map((r, i) => { acc += r.rm - r.rk; return [i + 1, acc]; });
  const mx = Math.max(0.05, ...pts.map((p) => Math.abs(p[1])));
  const x = (i) => pad.l + (i / pts.length) * (W - pad.l - pad.r), y = (v) => pad.t + (1 - (v + mx) / (2 * mx)) * (H - pad.t - pad.b);
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Erro acumulado: Vitrôncio menos mercado">
    <rect x="${pad.l}" y="${pad.t}" width="${W - pad.l - pad.r}" height="${y(0) - pad.t}" fill="rgba(255,61,77,.07)"/><rect x="${pad.l}" y="${y(0)}" width="${W - pad.l - pad.r}" height="${H - pad.b - y(0)}" fill="rgba(31,209,127,.07)"/>
    <line x1="${pad.l}" x2="${W - pad.r}" y1="${y(0)}" y2="${y(0)}" stroke="rgba(255,255,255,.35)"/>
    <text x="${pad.l + 6}" y="${pad.t + 13}">mercado melhor ↑</text><text x="${pad.l + 6}" y="${H - pad.b - 6}">Vitrôncio melhor ↓</text>
    <path d="${pts.map((p, i) => `${i ? 'L' : 'M'}${x(p[0]).toFixed(1)},${y(p[1]).toFixed(1)}`).join('')}" fill="none" stroke="var(--gold)" stroke-width="2.2"/>
    <text x="${pad.l}" y="${H - 6}">1º jogo</text><text x="${W - pad.r}" y="${H - 6}" text-anchor="end">${pts.length}º jogo</text></svg>`;
}

export function compareView() {
  const up = [...liveMatches(), ...upcoming()];
  const fin = finished();
  loadOdds(fin.slice(-30));
  const upRows = up.map((m) => ({ m, p: forecast(m).pre, k: market(m) })).filter((r) => r.k);
  const noOdds = up.length - upRows.length;
  const pastAll = fin.map((m) => { const k = market(m); if (!k) return null; const p = forecast(m).pre; return { m, p, k, rm: rpsOf(p, m), rk: rpsOf(k, m) }; }).filter(Boolean);
  const past = pastAll.filter((r) => !cmpState.team || r.m.h === cmpState.team || r.m.a === cmpState.team);
  const N = past.length, mean = (f) => (N ? past.reduce((s, r) => s + f(r), 0) / N : NaN);
  const closer = past.filter((r) => r.rm < r.rk - 1e-9).length, tie = past.filter((r) => Math.abs(r.rm - r.rk) <= 1e-9).length;
  const favHit = (key) => past.filter((r) => { const q = r[key]; const mx = Math.max(q.pH, q.pD, q.pA); return outcomeProb(q, r.m) === mx; }).length;
  const ids = teamIds().sort((a, b) => team(a).name.localeCompare(team(b).name));
  const three = (q, m, hi) => `<span class="tri">${[['pH', team(m.h).abbr], ['pD', 'E'], ['pA', team(m.a).abbr]].map(([key, lab]) => `<span class="${hi === key ? 'hit' : ''}"><i>${esc(lab)}</i>${pct(q[key])}</span>`).join('')}</span>`;
  const res = (m) => (m.hg > m.ag ? 'pH' : m.hg === m.ag ? 'pD' : 'pA');
  const bt = S.model.backtest;
  return `<div class="wrap">
    <section><div class="sec-h"><h2>Vitrôncio × mercado</h2><span class="sub">probabilidades do Vitrôncio, das odds e o resultado</span></div>
      <p class="muted" style="max-width:860px">As odds são convertidas em probabilidades retirando a margem da casa. O Vitrôncio (o modelo do site) não usa as odds em nenhuma etapa, então as duas estimativas são independentes. Não é recomendação de aposta.</p></section>
    <section><div class="sec-h"><h2>Próximos jogos</h2><span class="sub">${upRows.length ? 'ordenados por data' : ''}</span></div>
      ${upRows.length ? `<p class="small muted only-m" style="margin:0 0 6px">Colunas de probabilidade: mandante · empate · visitante.</p><div class="tw"><table class="cmp"><thead><tr><th class="l">Jogo</th><th class="hide-m">Quando</th><th>${face(16)}Vitrôncio</th><th>Mercado</th><th class="hide-m">Odds</th><th class="l hide-m">Maior diferença</th></tr></thead><tbody>
      ${upRows.map(({ m, p, k }) => { const g = gap(m, p, k); return `<tr class="${fav() && (m.h === fav() || m.a === fav()) ? 'fav' : ''}"><td class="team"><a href="#/jogo/${m.id}">${logo(team(m.h))}<span class="nm-full">${esc(team(m.h).name)} × ${esc(team(m.a).name)}</span><span class="nm-abbr">${esc(team(m.h).abbr)} × ${esc(team(m.a).abbr)}</span>${logo(team(m.a))}</a></td><td class="small hide-m">${m.status === 'in' ? '<span class="livepill">ao vivo</span>' : `${esc(fmtShort(m.date))} ${fmtTime(m.date)}`}</td><td>${three(p, m)}</td><td>${three(k, m)}</td><td class="small dim hide-m">${dec(k.dec[0])} · ${dec(k.dec[1])} · ${dec(k.dec[2])}</td><td class="l small hide-m">${Math.abs(g.v) < 0.03 ? '<span class="dim">quase iguais</span>' : `Vitrôncio <b style="color:${g.v > 0 ? 'var(--green)' : 'var(--orange)'}">${g.v > 0 ? '+' : ''}${(g.v * 100).toFixed(0)} pp</b> em ${esc(g.lab)}`}</td></tr>`; }).join('')}</tbody></table></div>` : '<p class="muted">Ainda não há odds para os próximos jogos.</p>'}
      ${noOdds && upRows.length ? `<p class="small muted" style="margin-top:8px">${noOdds} jogo${noOdds > 1 ? 's' : ''} ainda sem odds.</p>` : ''}</section>
    <section><div class="sec-h"><h2>Desempenho na temporada</h2><span class="sub">Série A ${S.season} · ${N} jogos com odds${cmpState.team ? ` do ${esc(team(cmpState.team).name)}` : ''}</span></div>
      <div class="btnrow" style="margin-bottom:12px"><select data-change="cmpteam" aria-label="Filtrar por time"><option value="">Todos os times</option>${ids.map((id) => `<option value="${id}" ${id === cmpState.team ? 'selected' : ''}>${esc(team(id).name)}</option>`).join('')}</select></div>
      ${N ? `<div class="kv">
        <div><div class="k">Erro médio do Vitrôncio (RPS)</div><div class="v">${mean((r) => r.rm).toFixed(4)}</div></div>
        <div><div class="k">Erro médio do mercado (RPS)</div><div class="v">${mean((r) => r.rk).toFixed(4)}</div></div>
        <div><div class="k">Vitrôncio mais preciso em</div><div class="v">${closer} <small>de ${N - tie} jogos (${pct(closer / Math.max(1, N - tie))})</small></div></div>
        <div><div class="k">Favorito do Vitrôncio venceu</div><div class="v">${pct(favHit('p') / N)}</div></div>
        <div><div class="k">Favorito do mercado venceu</div><div class="v">${pct(favHit('k') / N)}</div></div>
        <div><div class="k">Probabilidade média do resultado</div><div class="v s">Vitrôncio ${pct(mean((r) => outcomeProb(r.p, r.m)))} · mercado ${pct(mean((r) => outcomeProb(r.k, r.m)))}</div></div>
      </div>
      <p class="small muted">RPS mede o erro de uma previsão em probabilidades: 0 é perfeito, quanto menor melhor.</p>
      <div class="panel" style="margin-top:12px"><div class="sec-h"><h3>Erro acumulado</h3><span class="sub">soma, jogo a jogo, do erro do Vitrôncio menos o do mercado</span></div>${cumChart(past)}</div>` : '<p class="muted">Sem jogos com odds para comparar.</p>'}</section>
    ${N ? `<section><div class="sec-h"><h2>Jogo a jogo</h2><span class="sub">em destaque, o resultado</span></div>
      <div class="tw"><table class="cmp"><thead><tr><th class="hide-m">Data</th><th class="l">Jogo</th><th>Placar</th><th>${face(16)}Vitrôncio</th><th>Mercado</th><th class="hide-m">Mais preciso</th></tr></thead><tbody>
      ${past.slice(-cmpState.n).reverse().map((r) => { const m = r.m, d = r.rm - r.rk; return `<tr class="${fav() && (m.h === fav() || m.a === fav()) ? 'fav' : ''}"><td class="small dim hide-m">${esc(fmtShort(m.date))}</td><td class="team"><a href="#/jogo/${m.id}">${logo(team(m.h))}<span class="nm-full">${esc(team(m.h).name)} × ${esc(team(m.a).name)}</span><span class="nm-abbr">${esc(team(m.h).abbr)} × ${esc(team(m.a).abbr)}</span>${logo(team(m.a))}</a></td><td class="pts">${m.hg}–${m.ag}</td><td>${three(r.p, m, res(m))}</td><td>${three(r.k, m, res(m))}</td><td class="hide-m">${Math.abs(d) < 0.002 ? '<span class="dim">empate</span>' : d < 0 ? '<span class="chip win">Vitrôncio</span>' : '<span class="chip">Mercado</span>'}</td></tr>`; }).join('')}</tbody></table></div>
      ${past.length > cmpState.n ? `<div class="btnrow" style="margin-top:10px"><button class="btn sm" data-act="cmpmore">Mostrar mais jogos</button></div>` : ''}</section>` : ''}
    <section><div class="sec-h"><h2>Histórico por temporada</h2><span class="sub">o modelo atual prevendo cada jogo apenas com informação anterior a ele</span></div>
      <div class="tw"><table><thead><tr><th>Temporada</th><th>Jogos</th><th>RPS Vitrôncio</th><th>RPS mercado</th><th>Diferença</th><th>RPS de referência simples</th></tr></thead><tbody>
      ${Object.entries(bt.temporadas).reverse().map(([s, a]) => `<tr><td>${s}</td><td>${a.n}</td><td class="pts">${a.rps.toFixed(4)}</td><td>${a.rpsRef ? a.rpsRef.toFixed(4) : '–'}</td><td class="small">${a.rpsRef ? `${a.rpsModelNoRef - a.rpsRef > 0 ? '+' : ''}${(a.rpsModelNoRef - a.rpsRef).toFixed(4)}` : '–'}</td><td class="dim">${a.rpsBase.toFixed(4)}</td></tr>`).join('')}
      <tr><td><b>Total</b></td><td>${bt.geral.n}</td><td class="pts">${bt.geral.rps.toFixed(4)}</td><td>${bt.geral.rpsRef?.toFixed(4) ?? '–'}</td><td class="small">${bt.geral.rpsRef ? `+${(bt.geral.rpsModelNoRef - bt.geral.rpsRef).toFixed(4)}` : '–'}</td><td class="dim">${bt.geral.rpsBase.toFixed(4)}</td></tr></tbody></table></div>
      <p class="small muted" style="margin-top:8px">Odds históricas: média de fechamento (football-data.co.uk). Próximos jogos: DraftKings, via ESPN. Para jogos passados, a coluna do modelo é o que o modelo atual indicaria antes do jogo.</p></section>
  </div>`;
}
