// Calendário completo e rankings de jogadores.
import { S, team, allMatches, teamIds } from '../store.js?v=2a005902ef';
import { esc, dayKey, relDay, fmtFull } from '../util.js?v=2a005902ef';
import { fixtureList, logo, fav } from '../components.js?v=2a005902ef';

export const calState = { month: null };
const MES = ['Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho', 'Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Dezembro'];
const monthKey = (iso) => dayKey(iso).slice(0, 7);
const byDay = (list) => { const g = new Map(); for (const m of list) { const k = dayKey(m.date); if (!g.has(k)) g.set(k, []); g.get(k).push(m); } return [...g.values()]; };
export const calMonths = () => [...new Set(allMatches().map((m) => monthKey(m.date)))].sort();

/** Jogos: o mês escolhido, agrupado por dia; ou, com um time (#/jogos/<id>), a campanha inteira dele. */
export function calendarView(teamId) {
  const all = allMatches();
  const months = calMonths();
  const nowK = monthKey(new Date().toISOString());
  if (!calState.month || !months.includes(calState.month)) calState.month = months.includes(nowK) ? nowK : months.filter((k) => k <= nowK).at(-1) || months[0];
  const ids = teamIds().sort((a, b) => team(a).name.localeCompare(team(b).name));
  const tid = ids.includes(teamId) ? teamId : '';
  const done = all.filter((m) => m.status === 'post').length;
  const sel = `<select data-change="calteam" aria-label="Filtrar por time"><option value="">Todos os times</option>${ids.map((id) => `<option value="${id}" ${id === tid ? 'selected' : ''}>${esc(team(id).name)}</option>`).join('')}</select>`;
  let body;
  if (tid) {
    const games = all.filter((m) => m.h === tid || m.a === tid);
    const next = games.filter((m) => m.status !== 'post'), past = games.filter((m) => m.status === 'post').reverse();
    const r = { V: 0, E: 0, D: 0, gp: 0, gc: 0 };
    for (const m of past) { const [a, b] = m.h === tid ? [m.hg, m.ag] : [m.ag, m.hg]; r.gp += a; r.gc += b; r[a > b ? 'V' : a === b ? 'E' : 'D']++; }
    const t = team(tid);
    body = `<div class="panel favline" style="margin-bottom:14px"><a class="who" href="#/time/${tid}">${logo(t)}${esc(t.name)}</a>
        <span><b>${3 * r.V + r.E}</b> pts em ${past.length} jogos</span><span><b>${r.V}</b>V <b>${r.E}</b>E <b>${r.D}</b>D</span><span>gols <b>${r.gp}</b>–<b>${r.gc}</b></span>
        <span>${next.length ? `faltam <b>${next.length}</b> jogo${next.length > 1 ? 's' : ''}` : 'campanha encerrada'}</span><a class="more" href="#/time/${tid}">página do time</a></div>
      ${next.length ? `<section><div class="sec-h"><h3>Próximos</h3><span class="sub">C = em casa, F = fora</span></div>${fixtureList(next, { focus: tid })}</section>` : ''}
      ${past.length ? `<section><div class="sec-h"><h3>Disputados</h3><span class="sub">do mais recente ao primeiro</span></div>${fixtureList(past, { focus: tid })}</section>` : ''}`;
  } else {
    const i = months.indexOf(calState.month);
    const list = all.filter((m) => monthKey(m.date) === calState.month);
    const nav = `<span class="monthnav"><button class="btn sm" data-act="calmonth" data-id="${months[i - 1] || ''}" ${i > 0 ? '' : 'disabled'} aria-label="Mês anterior">‹</button>
      <select data-change="calmonth" aria-label="Mês">${months.map((k) => `<option value="${k}" ${k === calState.month ? 'selected' : ''}>${MES[+k.slice(5) - 1]}</option>`).join('')}</select>
      <button class="btn sm" data-act="calmonth" data-id="${months[i + 1] || ''}" ${i < months.length - 1 ? '' : 'disabled'} aria-label="Próximo mês">›</button></span>`;
    body = `<div class="btnrow" style="margin-bottom:14px">${sel}${nav}<span class="muted small">${list.length} jogos em ${MES[+calState.month.slice(5) - 1].toLowerCase()}</span></div>
      ${byDay(list).map((g) => `<div class="day">${esc(relDay(g[0].date))}</div>${fixtureList(g, { date: false })}`).join('') || '<p class="muted">Nenhum jogo neste mês.</p>'}`;
    return `<div class="wrap"><section><div class="sec-h"><h2>Jogos</h2><span class="sub">${done} de ${all.length} partidas disputadas na Série A ${S.season}</span></div>${body}</section></div>`;
  }
  return `<div class="wrap"><section><div class="sec-h"><h2>Jogos</h2><span class="sub">Série A ${S.season}</span></div><div class="btnrow" style="margin-bottom:14px">${sel}</div>${body}</section></div>`;
}

export function playersView() {
  const rows = [];
  for (const [tid, list] of Object.entries(S.model.players || {})) for (const p of list) rows.push({ ...p, tid });
  const tbl = (title, sub, sorted, cols) => `<div class="panel"><div class="sec-h"><h3>${title}</h3><span class="sub">${sub}</span></div><div class="tw"><table><thead><tr><th>#</th><th class="l">Jogador</th>${cols.map((c) => `<th>${c[0]}</th>`).join('')}</tr></thead><tbody>
    ${sorted.slice(0, 15).map((p, i) => `<tr class="${p.tid === fav() ? 'fav' : ''}"><td class="pos">${i + 1}</td><td class="team"><a href="#/time/${p.tid}">${logo(team(p.tid))}<span>${esc(p.n)}</span></a></td>${cols.map((c, k) => `<td class="${k === 0 ? 'pts' : ''}">${c[1](p)}</td>`).join('')}</tr>`).join('')}</tbody></table></div></div>`;
  const per90 = (v, p) => (p.min >= 450 ? ((v / p.min) * 90).toFixed(2).replace('.', ',') : '–');
  return `<div class="wrap"><section><div class="sec-h"><h2>Jogadores</h2><span class="sub">Série A ${S.model.temporada} · dados até ${esc(fmtFull(S.model.dadosAte))}</span></div>
    <div class="grid g2">
      ${tbl('Artilheiros', '', [...rows].sort((a, b) => b.g - a.g || a.min - b.min), [['Gols', (p) => p.g], ['Pên.', (p) => p.pk || ''], ['Jogos', (p) => p.apps], ['Gols/90', (p) => per90(p.g, p)]])}
      ${tbl('Assistências', '', [...rows].sort((a, b) => b.a - a.a || a.min - b.min), [['Assist.', (p) => p.a], ['Jogos', (p) => p.apps], ['Assist./90', (p) => per90(p.a, p)]])}
      ${tbl('Finalizações', '', [...rows].sort((a, b) => b.sh - a.sh), [['Final.', (p) => p.sh], ['No alvo', (p) => p.sot], ['Gols', (p) => p.g], ['Final./90', (p) => per90(p.sh, p)]])}
      ${tbl('Cartões', '', [...rows].sort((a, b) => b.y + 3 * b.r - (a.y + 3 * a.r)), [['Amarelos', (p) => p.y], ['Vermelhos', (p) => p.r || ''], ['Jogos', (p) => p.apps]])}
      ${tbl('Participações em gol', 'gols + assistências', [...rows].sort((a, b) => b.g + b.a - (a.g + a.a)), [['G+A', (p) => p.g + p.a], ['Gols', (p) => p.g], ['Assist.', (p) => p.a], ['Min', (p) => p.min]])}
      ${tbl('Minutos em campo', '', [...rows].sort((a, b) => b.min - a.min), [['Min', (p) => p.min], ['Jogos', (p) => p.apps], ['Titular', (p) => p.st]])}
    </div><p class="small muted" style="margin-top:10px">Atualizado a cada reajuste semanal do modelo.</p></section></div>`;
}
