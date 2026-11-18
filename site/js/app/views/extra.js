// Calendário completo e rankings de jogadores.
import { S, team, allMatches, teamIds } from '../store.js';
import { esc, dayKey, relDay, fmtFull } from '../util.js';
import { matchCard, logo, fav } from '../components.js';

export const calState = { month: null, team: '' };
const MES = ['Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho', 'Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Dezembro'];
const monthKey = (iso) => dayKey(iso).slice(0, 7);

export function calendarView() {
  const all = allMatches();
  const months = [...new Set(all.map((m) => monthKey(m.date)))].sort();
  const nowK = monthKey(new Date().toISOString());
  if (!calState.month || !months.includes(calState.month)) calState.month = months.includes(nowK) ? nowK : months.filter((k) => k <= nowK).at(-1) || months[0];
  const ids = teamIds().sort((a, b) => team(a).name.localeCompare(team(b).name));
  const list = all.filter((m) => (calState.team ? m.h === calState.team || m.a === calState.team : monthKey(m.date) === calState.month));
  const groups = new Map();
  for (const m of list) { const k = dayKey(m.date); if (!groups.has(k)) groups.set(k, []); groups.get(k).push(m); }
  const done = all.filter((m) => m.status === 'post').length;
  return `<div class="wrap"><section><div class="sec-h"><h2>Jogos</h2><span class="sub">${done} de ${all.length} partidas disputadas na Série A ${S.season}</span></div>
    <div class="btnrow" style="margin-bottom:14px">
      <select data-change="calteam" aria-label="Filtrar por time"><option value="">Todos os times</option>${ids.map((id) => `<option value="${id}" ${id === calState.team ? 'selected' : ''}>${esc(team(id).name)}</option>`).join('')}</select>
      ${calState.team ? '' : months.map((k) => `<button class="btn sm ${k === calState.month ? 'on' : ''}" data-act="calmonth" data-id="${k}">${MES[+k.slice(5) - 1].slice(0, 3)}</button>`).join('')}
    </div>
    ${[...groups.values()].map((g) => `<div style="margin-bottom:16px"><div class="sec-h"><h3 style="font-size:16px;color:var(--gold)">${esc(relDay(g[0].date))}</h3></div><div class="grid g3">${g.map((m) => matchCard(m, { day: !!calState.team })).join('')}</div></div>`).join('') || '<p class="muted">Nenhum jogo neste período.</p>'}
  </section></div>`;
}

export function playersView() {
  const rows = [];
  for (const [tid, list] of Object.entries(S.model.players || {})) for (const p of list) rows.push({ ...p, tid });
  const tbl = (title, sub, sorted, cols) => `<div class="panel"><div class="sec-h"><h3>${title}</h3><span class="sub">${sub}</span></div><div class="tw"><table><thead><tr><th>#</th><th class="l">Jogador</th>${cols.map((c) => `<th>${c[0]}</th>`).join('')}</tr></thead><tbody>
    ${sorted.slice(0, 15).map((p, i) => `<tr class="${p.tid === fav() ? 'fav' : ''}"><td class="pos">${i + 1}</td><td class="team"><a href="#/time/${p.tid}">${logo(team(p.tid))}<span>${esc(p.n)}</span></a></td>${cols.map((c, k) => `<td class="${k === 0 ? 'pts' : ''}">${c[1](p)}</td>`).join('')}</tr>`).join('')}</tbody></table></div></div>`;
  const per90 = (v, p) => (p.min >= 450 ? ((v / p.min) * 90).toFixed(2).replace('.', ',') : '–');
  return `<div class="wrap"><section><div class="sec-h"><h2>Artilharia e números</h2><span class="sub">Série A ${S.model.temporada} · dados até ${esc(fmtFull(S.model.dadosAte))}</span></div>
    <div class="grid g2">
      ${tbl('Artilheiros', 'gols', [...rows].sort((a, b) => b.g - a.g || a.min - b.min), [['Gols', (p) => p.g], ['Pên.', (p) => p.pk || ''], ['Jogos', (p) => p.apps], ['Gols/90', (p) => per90(p.g, p)]])}
      ${tbl('Garçons', 'assistências', [...rows].sort((a, b) => b.a - a.a || a.min - b.min), [['Assist.', (p) => p.a], ['Jogos', (p) => p.apps], ['Assist./90', (p) => per90(p.a, p)]])}
      ${tbl('Quem mais finaliza', 'finalizações', [...rows].sort((a, b) => b.sh - a.sh), [['Final.', (p) => p.sh], ['No alvo', (p) => p.sot], ['Gols', (p) => p.g], ['Final./90', (p) => per90(p.sh, p)]])}
      ${tbl('Os mais advertidos', 'cartões', [...rows].sort((a, b) => b.y + 3 * b.r - (a.y + 3 * a.r)), [['Amarelos', (p) => p.y], ['Vermelhos', (p) => p.r || ''], ['Jogos', (p) => p.apps]])}
      ${tbl('Participações em gol', 'gols + assistências', [...rows].sort((a, b) => b.g + b.a - (a.g + a.a)), [['G+A', (p) => p.g + p.a], ['Gols', (p) => p.g], ['Assist.', (p) => p.a], ['Min', (p) => p.min]])}
      ${tbl('Homens de ferro', 'minutos em campo', [...rows].sort((a, b) => b.min - a.min), [['Min', (p) => p.min], ['Jogos', (p) => p.apps], ['Titular', (p) => p.st]])}
    </div><p class="small muted" style="margin-top:10px">Os números de jogadores são atualizados no reajuste semanal do modelo.</p></section></div>`;
}
