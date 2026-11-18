// Componentes de interface (funções que devolvem HTML).
import { S, team, forecast, table, liveState } from './store.js';
import { esc, pct, fmtTime, fmtShort, teamColor, store } from './util.js';

export const fav = () => store.get('fav', null);
export const failedLogos = new Set();
export function logo(t, cls = '') {
  if (!t?.logo || failedLogos.has(t.logo)) return `<span class="logo fallback ${cls}">${esc((t?.abbr || '?').slice(0, 3))}</span>`;
  return `<img class="logo ${cls}" src="${esc(t.logo)}" alt="" loading="lazy" onerror="bdvLogoFail(this)">`;
}
export const colors = (m) => `--tc-h:${teamColor(team(m.h))};--tc-a:${distinct(teamColor(team(m.h)), teamColor(team(m.a)))}`;
function distinct(a, b) { // evita duas cores quase iguais lado a lado
  const rgb = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
  const [x, y] = [rgb(a), rgb(b)];
  const d = Math.abs(x[0] - y[0]) + Math.abs(x[1] - y[1]) + Math.abs(x[2] - y[2]);
  return d < 90 ? '#e8eef5' : b;
}

export function probBar(p, { sm = false } = {}) {
  const w = [p.pH, p.pD, p.pA].map((v) => Math.max(0, v * 100));
  const lab = (v) => (v >= 9 ? pct(v / 100) : '');
  return `<div class="pbar ${sm ? 'sm' : ''}"><div class="h" style="width:${w[0]}%">${lab(w[0])}</div><div class="d" style="width:${w[1]}%">${lab(w[1])}</div><div class="a" style="width:${w[2]}%">${lab(w[2])}</div></div>`;
}
export function probLabels(m, p) {
  return `<div class="plabels"><span><b>${pct(p.pH)}</b> ${esc(team(m.h).name)}</span><span>empate <b>${pct(p.pD)}</b></span><span>${esc(team(m.a).name)} <b>${pct(p.pA)}</b></span></div>`;
}
export function formDots(f) { return `<span class="form">${f.map((x) => `<i class="${x.r}" title="${esc(team(x.m.h).name)} ${x.m.hg}–${x.m.ag} ${esc(team(x.m.a).name)}">${x.r}</i>`).join('')}</span>`; }

export function statusLabel(m) {
  if (m.status === 'in') return /HALFTIME/.test(m.statusName) ? 'Intervalo' : m.clock || 'Ao vivo';
  if (m.status === 'post') return 'Encerrado';
  if (m.status === 'off') return 'Adiado';
  return fmtTime(m.date);
}

/** Cartão compacto de jogo (listas). */
export function matchCard(m, { selected = false, act = null, day = false } = {}) {
  const H = team(m.h), A = team(m.a), f = forecast(m), p = m.status === 'pre' || m.status === 'off' ? f.pre : f.now;
  const played = m.status === 'in' || m.status === 'post';
  const top = f.pre.sum.top[0];
  const isFav = fav() && (m.h === fav() || m.a === fav());
  const attrs = act ? `href="#/" data-act="${act}" data-id="${m.id}"` : `href="#/jogo/${m.id}"`;
  const cls = (side) => (m.status !== 'post' || m.hg === m.ag ? '' : (side === 'h' ? m.hg > m.ag : m.ag > m.hg) ? '' : 'lose');
  const when = m.status === 'in' ? `<span class="livepill">${esc(statusLabel(m))}</span>` : `${day || m.status === 'post' ? `${esc(fmtShort(m.date))}${m.status === 'pre' ? ' · ' : ''}` : ''}${m.status === 'post' ? '' : esc(statusLabel(m))}`;
  const foot = m.status === 'post' ? `<span>probabilidade pré-jogo deste resultado</span><b>${pct(m.hg > m.ag ? f.pre.pH : m.hg === m.ag ? f.pre.pD : f.pre.pA)}</b>`
    : m.status === 'in' ? `<span>agora</span><span><b>${pct(p.pH)}</b> · ${pct(p.pD)} · <b>${pct(p.pA)}</b></span>`
    : `<span>empate ${pct(p.pD)}</span><span>mais provável <b>${top.h}–${top.a}</b></span>`;
  return `<a class="mcard ${selected ? 'sel' : ''} ${isFav ? 'fav' : ''}" ${attrs} style="${colors(m)}">
    <div class="meta"><span>${when}</span><span>${esc(m.venue || '')}</span></div>
    <div class="row ${cls('h')}">${logo(H)}<span>${esc(H.name)}</span>${played ? `<span class="sc">${m.hg}</span>` : `<span class="pp">${pct(p.pH)}</span>`}</div>
    <div class="row ${cls('a')}">${logo(A)}<span>${esc(A.name)}</span>${played ? `<span class="sc">${m.ag}</span>` : `<span class="pp">${pct(p.pA)}</span>`}</div>
    ${m.status === 'post' ? '' : `<div style="margin-top:7px">${probBar(p, { sm: true })}</div>`}
    <div class="foot">${foot}</div>
  </a>`;
}

/** Matriz de placares (mapa de calor). `now` destaca o placar atual. */
export function scoreMatrix(dist, m, { max = 5, now = null } = {}) {
  const mx = dist.matrix; let peak = 0;
  for (let i = 0; i <= max; i++) for (let j = 0; j <= max; j++) peak = Math.max(peak, mx[i][j]);
  let h = `<table class="mx"><tr><th></th>${Array.from({ length: max + 1 }, (_, j) => `<th>${j}</th>`).join('')}</tr>`;
  for (let i = 0; i <= max; i++) {
    h += `<tr><th>${i}</th>`;
    for (let j = 0; j <= max; j++) {
      const p = mx[i][j], a = Math.pow(p / (peak || 1), 0.75);
      const col = i > j ? '79,157,232' : i < j ? '240,149,74' : '150,163,180';
      const dead = now && (i < now[0] || j < now[1]);
      h += `<td class="${now && now[0] === i && now[1] === j ? 'now' : ''} ${dead ? 'dead' : ''}" style="background:rgba(${col},${(0.08 + a * 0.82).toFixed(2)})" title="${i}–${j}: ${pct(p, 1)}">${p >= 0.005 ? pct(p) : ''}</td>`;
    }
    h += '</tr>';
  }
  return `${h}</table><div class="legend" style="justify-content:center"><span>linhas: gols do ${esc(team(m.h).name)}</span><span>colunas: gols do ${esc(team(m.a).name)}</span></div>`;
}

/** Gráfico da chance de vitória ao longo do jogo (áreas empilhadas). */
export function wpChart(tl, m, { total = null } = {}) {
  const W = 720, Hh = 190, pad = { l: 34, r: 10, t: 10, b: 22 };
  const n = total ?? tl.total, pts = tl.points;
  if (!pts.length) return '';
  const x = (t) => pad.l + (t / n) * (W - pad.l - pad.r), y = (v) => pad.t + (1 - v) * (Hh - pad.t - pad.b);
  const line = (f) => pts.map((p, i) => `${i ? 'L' : 'M'}${x(p.t).toFixed(1)},${y(f(p)).toFixed(1)}`).join('');
  const last = pts[pts.length - 1];
  const homeArea = `${line((p) => p.pH)}L${x(last.t)},${y(0)}L${x(0)},${y(0)}Z`;
  const awayArea = `${line((p) => 1 - p.pA)}L${x(last.t)},${y(1)}L${x(0)},${y(1)}Z`;
  const hz = S.model.hazard, half = 45 + hz.S1;
  const marks = tl.events.filter((e) => e.step <= last.t).map((e) => {
    const cx = x(e.step + 1), isG = e.t !== 'R';
    return `<g><line x1="${cx}" x2="${cx}" y1="${pad.t}" y2="${Hh - pad.b}" stroke="${isG ? 'rgba(242,194,48,.5)' : 'rgba(239,68,82,.5)'}" stroke-dasharray="3 3"/><circle cx="${cx}" cy="${e.s === 'h' ? Hh - pad.b - 8 : pad.t + 8}" r="4.5" fill="${isG ? '#f2c230' : '#ef4452'}"><title>${esc(e.n || '')} ${e.m}'${e.x ? '+' + e.x : ''}</title></circle></g>`;
  }).join('');
  const grid = [0.25, 0.5, 0.75].map((v) => `<line x1="${pad.l}" x2="${W - pad.r}" y1="${y(v)}" y2="${y(v)}" stroke="rgba(255,255,255,.07)"/><text x="${pad.l - 6}" y="${y(v) + 4}" text-anchor="end">${v * 100}%</text>`).join('');
  const ticks = [[0, "0'"], [half, 'int.'], [n, 'fim']].map(([t, l]) => `<text x="${x(t)}" y="${Hh - 6}" text-anchor="${t === 0 ? 'start' : t === n ? 'end' : 'middle'}">${l}</text>`).join('');
  return `<svg class="chart" viewBox="0 0 ${W} ${Hh}" role="img" aria-label="Probabilidade de vitória ao longo do jogo">
    <rect x="${pad.l}" y="${pad.t}" width="${W - pad.l - pad.r}" height="${Hh - pad.t - pad.b}" fill="rgba(125,143,163,.28)" rx="4"/>
    <path d="${homeArea}" fill="var(--tc-h)" opacity=".85"/><path d="${awayArea}" fill="var(--tc-a)" opacity=".85"/>
    ${grid}<line x1="${x(half)}" x2="${x(half)}" y1="${pad.t}" y2="${Hh - pad.b}" stroke="rgba(255,255,255,.25)"/>${marks}
    <circle cx="${x(last.t)}" cy="${y(last.pH)}" r="4" fill="#fff"/>${ticks}
  </svg><div class="legend"><span><i style="background:var(--tc-h)"></i>${esc(team(m.h).name)} vence</span><span><i style="background:rgba(125,143,163,.6)"></i>empate</span><span><i style="background:var(--tc-a)"></i>${esc(team(m.a).name)} vence</span><span><i style="background:var(--gold);border-radius:50%"></i>gol</span><span><i style="background:var(--red);border-radius:50%"></i>expulsão</span></div>`;
}

/** Linha de lance. `key` estável = o mesmo elemento entre atualizações (não pisca). */
export function eventRow(e, key = null) {
  return `<div class="ev ${e.type || ''} ${e.side || ''}"${key ? ` data-mk="${key}" data-enter` : ''}><span class="t">${esc(e.label || '')}</span><span class="ic"></span><span>${esc(e.text)}</span></div>`;
}
export function statBars(st) {
  const row = (lab, a, b, suffix = '') => {
    if (a == null || b == null) return '';
    const tot = a + b || 1;
    return `<div class="stat"><div class="lab">${lab}</div><div class="v">${a}${suffix}</div><div class="bar"><span><i class="l" style="width:${(a / tot) * 100}%"></i></span><span><i class="r" style="width:${(b / tot) * 100}%"></i></span></div><div class="v">${b}${suffix}</div></div>`;
  };
  return row('Posse de bola', st.h.po != null ? Math.round(st.h.po) : null, st.a.po != null ? Math.round(st.a.po) : null, '%') + row('Finalizações', st.h.sh, st.a.sh) + row('No alvo', st.h.sot, st.a.sot) + row('Escanteios', st.h.co, st.a.co) + row('Faltas', st.h.fo, st.a.fo) + row('Cartões amarelos', st.h.yc, st.a.yc) + row('Cartões vermelhos', st.h.rc, st.a.rc) + row('Impedimentos', st.h.off, st.a.off) + row('Defesas', st.h.sv, st.a.sv) + row('Passes certos', st.h.passOk, st.a.passOk);
}
export function scorerList(list, n = 6) {
  const mx = Math.max(...list.slice(0, n).map((x) => x.p), 0.01);
  return `<div class="plist">${list.slice(0, n).map((x) => `<div class="p"><span class="j">${esc(x.j || '')}</span><span>${esc(x.n)}${x.gols ? ` <span class="dim small">${x.gols} gol${x.gols > 1 ? 's' : ''}</span>` : ''}</span><span class="pc">${pct(x.p)}</span><span class="pb"><i style="width:${(x.p / mx) * 100}%"></i></span></div>`).join('')}</div>`;
}
export function heat(p, rgb = '255,210,31') { return `<span class="heat" style="background:rgba(${rgb},${(Math.pow(p, 0.6) * 0.75).toFixed(2)})">${pct(p)}</span>`; }

/** Lista de probabilidades da simulação de temporada (título, G4, Z4). */
export function raceBlock(key, n, rgb) {
  const sim = S.seasonSim; if (!sim) return '<p class="muted small">Simulando…</p>';
  const rows = Object.values(sim.teams).sort((a, b) => b[key] - a[key]).slice(0, n).filter((r) => r[key] > 0.0005);
  if (!rows.length) return '<p class="muted small">Já definido.</p>';
  return `<div class="race">${rows.map((r) => { const t = team(r.id); return `<a class="r" href="#/time/${r.id}">${logo(t)}<span>${esc(t.name)}</span><span class="b"><i style="width:${r[key] * 100}%;background:rgb(${rgb})"></i></span><span class="v">${pct(r[key])}</span></a>`; }).join('')}</div>`;
}
export { liveState, table };
