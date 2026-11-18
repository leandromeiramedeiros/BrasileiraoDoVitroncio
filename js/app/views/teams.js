// Lista de times (ranking de força) e página de cada time.
import { S, team, strengths, table, allMatches, forecast, teamIds } from '../store.js?v=2a005902ef';
import { esc, pct, num, fmtShort, store } from '../util.js?v=2a005902ef';
import { logo, fixtureList, fav, heat } from '../components.js?v=2a005902ef';
import { posDist } from './table.js?v=2a005902ef';

export function teamsView() {
  const str = strengths(), tb = table(), sim = S.seasonSim, f = fav();
  const mxA = Math.max(...str.map((s) => s.att)), mnD = Math.min(...str.map((s) => s.def)), mxD = Math.max(...str.map((s) => s.def));
  return `<div class="wrap"><section><div class="sec-h"><h2>Ranking de força</h2><span class="sub">força estimada pelo modelo, atualizada a cada jogo</span></div>
    <div class="tw"><table><thead><tr><th>#</th><th class="l">Time</th><th title="gols que marcaria contra um time médio, em campo neutro">Ataque</th><th title="gols que sofreria de um time médio, em campo neutro">Defesa</th><th>Força</th><th>Tabela</th><th>Título</th><th>Z4</th><th></th></tr></thead><tbody>
    ${str.map((s) => { const t = team(s.id), r = tb.find((x) => x.id === s.id), sm = sim?.teams[s.id]; return `<tr class="${s.id === f ? 'fav' : ''}"><td class="pos">${s.rank}</td><td class="team"><a href="#/time/${s.id}">${logo(t)}<span>${esc(t.name)}</span></a></td>
      <td><span class="heat" style="background:rgba(31,209,127,${(0.1 + 0.6 * (s.att / mxA) ** 3).toFixed(2)})">${num(s.att, 2)}</span> <span class="dim small">${s.rankAtt}º</span></td>
      <td><span class="heat" style="background:rgba(69,167,255,${(0.1 + 0.6 * ((mxD - s.def) / (mxD - mnD || 1))).toFixed(2)})">${num(s.def, 2)}</span> <span class="dim small">${s.rankDef}º</span></td>
      <td class="pts">${s.rating >= 0 ? '+' : ''}${num(s.rating, 2)}</td><td>${r ? `${r.rank}º · ${r.pts} pts` : '–'}</td><td>${sm ? heat(sm.titulo) : '–'}</td><td>${sm ? heat(sm.rebaixamento, '255,61,77') : '–'}</td>
      <td><button class="btn sm ${s.id === f ? 'on' : ''}" data-act="fav" data-id="${s.id}">${s.id === f ? 'Seguindo' : 'Seguir'}</button></td></tr>`; }).join('')}
    </tbody></table></div>
    <p class="small muted" style="margin-top:10px"><b>Ataque</b>: gols que o time marcaria contra um adversário médio da Série A em campo neutro. <b>Defesa</b>: gols que sofreria (menor é melhor). <b>Força</b>: saldo dos dois, em escala logarítmica — zero é a média da liga.</p></section></div>`;
}

function ratingChart(id) {
  const hist = S.model.ratingHistory?.[id]; if (!hist || hist.length < 3) return '';
  const W = 720, H = 170, pad = { l: 36, r: 10, t: 10, b: 22 };
  const all = Object.values(S.model.ratingHistory).flat().map((p) => p[1] + p[2]);
  const mn = Math.min(...all), mx = Math.max(...all);
  const x = (i, n) => pad.l + (i / Math.max(1, n - 1)) * (W - pad.l - pad.r), y = (v) => pad.t + (1 - (v - mn) / (mx - mn || 1)) * (H - pad.t - pad.b);
  const others = Object.entries(S.model.ratingHistory).filter(([k]) => k !== id).map(([, h]) => `<path d="${h.map((p, i) => `${i ? 'L' : 'M'}${x(i, h.length).toFixed(1)},${y(p[1] + p[2]).toFixed(1)}`).join('')}" fill="none" stroke="rgba(255,255,255,.07)" stroke-width="1"/>`).join('');
  const d = hist.map((p, i) => `${i ? 'L' : 'M'}${x(i, hist.length).toFixed(1)},${y(p[1] + p[2]).toFixed(1)}`).join('');
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Evolução da força ao longo da temporada">${others}
    <line x1="${pad.l}" x2="${W - pad.r}" y1="${y(0)}" y2="${y(0)}" stroke="rgba(255,255,255,.25)" stroke-dasharray="4 4"/><text x="${pad.l - 6}" y="${y(0) + 4}" text-anchor="end">média</text>
    <path d="${d}" fill="none" stroke="var(--gold)" stroke-width="2.5"/><circle cx="${x(hist.length - 1, hist.length)}" cy="${y(hist.at(-1)[1] + hist.at(-1)[2])}" r="4" fill="var(--gold)"/>
    <text x="${pad.l}" y="${H - 6}">1ª rodada</text><text x="${W - pad.r}" y="${H - 6}" text-anchor="end">hoje</text></svg>`;
}

export function teamView(id) {
  const t = S.teams[id];
  if (!t || !teamIds().includes(id)) return `<div class="wrap"><p class="muted">Time não encontrado. <a class="gold" href="#/times">Ver todos</a></p></div>`;
  const s = strengths().find((x) => x.id === id), r = table().find((x) => x.id === id), sm = S.seasonSim?.teams[id], f = fav();
  const games = allMatches().filter((m) => m.h === id || m.a === id);
  const past = games.filter((m) => m.status === 'post').slice(-6).reverse(), next = games.filter((m) => m.status === 'pre' || m.status === 'in').slice(0, 6);
  const squad = (S.model.players?.[id] || []).filter((p) => p.min >= 90).sort((a, b) => b.g - a.g || b.a - a.a || b.min - a.min);
  const P = { G: 'GOL', D: 'DEF', M: 'MEI', F: 'ATA' };
  return `<div class="wrap">
    <section class="panel"><div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap">${logo(t, 'big')}<div><h1>${esc(t.full || t.name)}</h1><div class="muted">${r ? `${r.rank}º lugar · ${r.pts} pontos · ${r.v}V ${r.e}E ${r.d}D · saldo ${r.gp - r.gc > 0 ? '+' : ''}${r.gp - r.gc}` : ''}</div></div>
      <button class="btn ${id === f ? 'on' : ''}" style="margin-left:auto" data-act="fav" data-id="${id}">${id === f ? 'Seguindo' : 'Seguir este time'}</button></div>
      <div class="kv" style="margin-top:14px">
        ${s ? `<div><div class="k">Força (ranking)</div><div class="v">${s.rank}º <small>de ${strengths().length}</small></div></div><div><div class="k">Ataque</div><div class="v">${num(s.att, 2)} <small>${s.rankAtt}º</small></div></div><div><div class="k">Defesa</div><div class="v">${num(s.def, 2)} <small>${s.rankDef}º</small></div></div>` : ''}
        ${sm ? `<div><div class="k">Título</div><div class="v">${pct(sm.titulo)}</div></div><div><div class="k">Libertadores (G4)</div><div class="v">${pct(sm.libertadores)}</div></div><div><div class="k">Sul-Americana</div><div class="v">${pct(sm.sulAmericana)}</div></div><div><div class="k">Rebaixamento</div><div class="v">${pct(sm.rebaixamento)}</div></div><div><div class="k">Pontos finais</div><div class="v">${sm.pts.toFixed(0)} <small>${sm.ptsLo}–${sm.ptsHi}</small></div></div>` : ''}
      </div>
      ${sm ? `<p class="small muted" style="margin:12px 0 0">Posição final: ${sm.pos.map((p, i) => (p >= 0.03 ? `<span class="chip">${i + 1}º <b>${pct(p)}</b></span>` : '')).join(' ')}</p>` : ''}</section>
    ${ratingChart(id) ? `<section class="panel"><div class="sec-h"><h3>Força ao longo da temporada</h3><span class="sub">em destaque: ${esc(t.name)}; em cinza, os demais times</span></div>${ratingChart(id)}</section>` : ''}
    ${next.length ? `<section><div class="sec-h"><h2>Próximos jogos</h2><span class="sub">C = em casa, F = fora</span><a class="more" href="#/jogos/${id}">todos os jogos</a></div>${fixtureList(next, { focus: id })}</section>` : ''}
    ${past.length ? `<section><div class="sec-h"><h2>Últimos jogos</h2><a class="more" href="#/jogos/${id}">todos os jogos</a></div>${fixtureList(past, { focus: id })}</section>` : ''}
    ${squad.length ? `<section><div class="sec-h"><h2>Elenco</h2><span class="sub">Série A ${S.model.temporada}, até o último reajuste do modelo</span></div><div class="tw"><table><thead><tr><th>Nº</th><th class="l">Jogador</th><th>Pos</th><th>Jogos</th><th>Tit.</th><th>Min</th><th>Gols</th><th>Assist.</th><th>Final.</th><th>No alvo</th><th>CA</th><th>CV</th></tr></thead><tbody>
      ${squad.map((p) => `<tr><td class="dim">${esc(p.j)}</td><td class="l" style="font-weight:600">${esc(p.n)}</td><td class="dim">${P[p.pos]}</td><td>${p.apps}</td><td>${p.st}</td><td>${p.min}</td><td class="pts">${p.g || ''}</td><td>${p.a || ''}</td><td>${p.sh || ''}</td><td>${p.sot || ''}</td><td>${p.y || ''}</td><td>${p.r || ''}</td></tr>`).join('')}</tbody></table></div></section>` : ''}
  </div>`;
}
