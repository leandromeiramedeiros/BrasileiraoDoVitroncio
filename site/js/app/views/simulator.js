// Simulador: "e se?" da temporada + confronto livre entre dois times.
import { S, team, upcoming, liveMatches, teamIds, runSeasonSim, table } from '../store.js';
import { esc, pct, fmtShort, store } from '../util.js';
import { logo, fav } from '../components.js';
import { pairBlock, pairMatch, sim } from './match.js';

export const simState = { fixed: {}, result: null, running: false, h: null, a: null, n: 20000, all: false };
const SCORE = { h: [2, 1], d: [1, 1], a: [1, 2] };

export function runWhatIf(render) {
  const fixed = {}; for (const [id, r] of Object.entries(simState.fixed)) fixed[id] = SCORE[r];
  simState.running = true; render();
  runSeasonSim({ n: simState.n, fixed, onDone: (res) => { simState.result = res; simState.running = false; render(); } });
}

export function simulatorView() {
  const ids = teamIds().sort((x, y) => team(x).name.localeCompare(team(y).name));
  if (!simState.h) { simState.h = fav() || ids[0]; simState.a = ids.find((i) => i !== simState.h); }
  const opt = (sel) => ids.map((id) => `<option value="${id}" ${id === sel ? 'selected' : ''}>${esc(team(id).name)}</option>`).join('');
  const rest = [...liveMatches(), ...upcoming()];
  const base = S.seasonSim, res = simState.result;
  const nFixed = Object.keys(simState.fixed).length;
  const shownN = simState.all ? rest.length : 10;
  const whatif = rest.slice(0, shownN).map((m) => {
    const cur = simState.fixed[m.id];
    const b = (k, l) => `<button class="${cur === k ? 'on' : ''}" data-act="whatif" data-id="${m.id}" data-r="${k}">${l}</button>`;
    return `<div class="whatif"><div class="t r"><span>${esc(team(m.h).name)}</span>${logo(team(m.h))}</div><div class="seg">${b('h', 'Casa')}${b('d', 'Emp.')}${b('a', 'Fora')}</div><div class="t">${logo(team(m.a))}<span>${esc(team(m.a).name)}</span><span class="dim small" style="margin-left:auto;white-space:nowrap">${esc(fmtShort(m.date))}</span></div></div>`;
  }).join('');
  let cmp = '';
  if (res && base) {
    const rows = table().map((r) => ({ r, b: base.teams[r.id], w: res.teams[r.id] })).filter((x) => x.b && x.w);
    const d = (a, b) => { const v = (a - b) * 100; return Math.abs(v) < 0.5 ? '<span class="dim">=</span>' : `<span style="color:${v > 0 ? 'var(--green)' : 'var(--red)'}">${v > 0 ? '+' : ''}${v.toFixed(0)} pp</span>`; };
    const dz = (a, b) => { const v = (a - b) * 100; return Math.abs(v) < 0.5 ? '<span class="dim">=</span>' : `<span style="color:${v < 0 ? 'var(--green)' : 'var(--red)'}">${v > 0 ? '+' : ''}${v.toFixed(0)} pp</span>`; };
    cmp = `<div class="tw" style="margin-top:14px"><table><thead><tr><th>#</th><th class="l">Time</th><th>Pts finais</th><th>Título</th><th></th><th>G4</th><th></th><th>Z4</th><th></th></tr></thead><tbody>${rows.sort((x, y) => y.w.pts - x.w.pts).map((x, i) => `<tr class="${x.r.id === fav() ? 'fav' : ''}"><td class="pos">${i + 1}</td><td class="team"><a href="#/time/${x.r.id}">${logo(team(x.r.id))}<span>${esc(team(x.r.id).name)}</span></a></td><td class="pts">${x.w.pts.toFixed(0)}</td><td>${pct(x.w.titulo)}</td><td class="small">${d(x.w.titulo, x.b.titulo)}</td><td>${pct(x.w.libertadores)}</td><td class="small">${d(x.w.libertadores, x.b.libertadores)}</td><td>${pct(x.w.rebaixamento)}</td><td class="small">${dz(x.w.rebaixamento, x.b.rebaixamento)}</td></tr>`).join('')}</tbody></table></div><p class="small muted">pp = pontos percentuais de diferença em relação à simulação sem resultados fixados.</p>`;
  }
  return `<div class="wrap">
    <section><div class="sec-h"><h2>Confronto livre</h2><span class="sub">qualquer mandante contra qualquer visitante</span></div>
      <div class="btnrow" style="margin-bottom:12px"><select data-change="pair" data-side="h" aria-label="Mandante">${opt(simState.h)}</select><span class="muted">×</span><select data-change="pair" data-side="a" aria-label="Visitante">${opt(simState.a)}</select>
        <button class="btn" data-act="swap">Inverter mando</button></div>
      ${simState.h !== simState.a ? pairBlock(simState.h, simState.a) : '<p class="muted">Escolha dois times diferentes.</p>'}</section>
    <section><div class="sec-h"><h2>Cenários</h2><span class="sub">fixe o resultado de jogos futuros e veja o efeito na tabela final</span></div>
      ${rest.length ? `<div class="grid g2">${whatif}</div>
      <div class="btnrow" style="margin-top:12px"><button class="btn gold" data-act="runwhatif">${simState.running ? 'Simulando…' : `Simular o campeonato${nFixed ? ` com ${nFixed} resultado${nFixed > 1 ? 's' : ''} fixado${nFixed > 1 ? 's' : ''}` : ''}`}</button>${rest.length > 10 ? `<button class="btn" data-act="whatifall">${simState.all ? 'Mostrar menos jogos' : `Mostrar os ${rest.length} jogos restantes`}</button>` : ''}${nFixed ? '<button class="btn" data-act="clearwhatif">Limpar</button>' : ''}<span class="muted small">${simState.n.toLocaleString('pt-BR')} simulações</span></div>${cmp}` : '<p class="muted">Não há jogos restantes nesta temporada.</p>'}
    </section></div>`;
}
