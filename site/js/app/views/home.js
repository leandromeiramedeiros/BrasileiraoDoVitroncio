// Página inicial: jogo ao vivo em destaque (ou o próximo), a rodada, a
// projeção do campeonato e os últimos resultados. Curta de propósito — o
// detalhe fica nas páginas de cada jogo.
import { S, team, liveMatches, upcoming, finished, table } from '../store.js';
import { esc, pct, dayKey, relDay, countdown, fmtTime } from '../util.js';
import { matchCard, raceBlock, logo, fav } from '../components.js';
import { matchBlock, sim } from './match.js';

export const homeState = { selected: null, simHere: false };

export function homeView() {
  const live = liveMatches(), up = upcoming(), fin = finished();
  const today = dayKey(new Date().toISOString());
  const simM = sim.res && sim.m && homeState.simHere ? S.matches.get(sim.m.id) : null;
  let hero = '', heroId = null;
  if (simM) { hero = `<section data-mk="hero">${matchBlock(simM)}</section>`; heroId = simM.id; }
  else if (live.length) {
    const sel = live.find((m) => m.id === homeState.selected) || live[0];
    const others = [...live, ...up.filter((m) => dayKey(m.date) === today), ...fin.filter((m) => dayKey(m.date) === today)];
    heroId = sel.id;
    hero = `<section data-mk="hero"><div class="sec-h"><h2>Ao vivo</h2>${live.length > 1 ? `<span class="sub">${live.length} jogos em andamento</span>` : ''}</div>
      ${matchBlock(sel)}
      ${others.length > 1 ? `<div class="strip">${others.map((m) => matchCard(m, { selected: m.id === sel.id, act: m.status === 'in' ? 'selectlive' : null })).join('')}</div>` : ''}</section>`;
  } else if (up.length) {
    const nx = up[0]; heroId = nx.id;
    hero = `<section data-mk="hero"><div class="sec-h"><h2>Próximo jogo</h2><span class="sub">${esc(relDay(nx.date))}, ${fmtTime(nx.date)} · ${countdown(nx.date)}</span></div>${matchBlock(nx, { compact: true })}</section>`;
  } else {
    hero = '<section class="panel" data-mk="hero"><h2>Temporada encerrada</h2><p class="muted" style="margin:6px 0 0">Não há jogos marcados. A tabela final e o histórico continuam disponíveis.</p></section>';
  }

  // a rodada: os próximos dois dias com jogo (sem repetir o que já está em destaque)
  const groups = new Map();
  for (const m of up) { if (live.length && dayKey(m.date) === today) continue; const k = dayKey(m.date); if (!groups.has(k)) groups.set(k, []); groups.get(k).push(m); }
  const days = [...groups.values()].slice(0, 2);
  const upcomingHtml = days.map((list) => `<div class="day" data-mk="d${dayKey(list[0].date)}">${esc(relDay(list[0].date))}</div><div class="grid g3" data-mk="g${dayKey(list[0].date)}">${list.filter((m) => m.id !== heroId || live.length).map((m) => matchCard(m)).join('')}</div>`).join('');

  const f = fav(), favTeam = f && S.teams[f], favRow = favTeam && table().find((r) => r.id === f), favSim = favTeam && S.seasonSim?.teams[f];
  const favNext = favTeam && up.find((m) => m.h === f || m.a === f);
  const favHtml = favTeam && favRow ? `<section class="panel" data-mk="fav"><div class="favline"><a class="who" href="#/time/${f}">${logo(favTeam)}${esc(favTeam.name)}</a><span><b>${favRow.rank}º</b> · ${favRow.pts} pts</span>
      ${favSim ? `<span>título <b>${pct(favSim.titulo)}</b></span><span>G4 <b>${pct(favSim.libertadores)}</b></span><span>rebaixamento <b>${pct(favSim.rebaixamento)}</b></span>` : ''}
      ${favNext ? `<a href="#/jogo/${favNext.id}">próximo: <b>${esc(team(favNext.h).abbr)} × ${esc(team(favNext.a).abbr)}</b>, ${esc(relDay(favNext.date).toLowerCase())}</a>` : ''}</div></section>` : '';

  const last = fin.slice(-6).reverse();
  return `<div class="wrap">
    ${S.error ? `<div class="banner" data-mk="banner">${esc(S.error)}</div>` : ''}
    ${hero}
    ${favHtml}
    ${upcomingHtml ? `<section data-mk="next"><div class="sec-h"><h2>Próximos jogos</h2><span class="sub">probabilidade de vitória de cada lado</span><a class="more" href="#/jogos">calendário completo</a></div>${upcomingHtml}</section>` : ''}
    <section data-mk="proj"><div class="sec-h"><h2>Projeção do campeonato</h2><span class="sub">${S.seasonSim ? `${S.seasonSim.n.toLocaleString('pt-BR')} simulações do restante da temporada${live.length ? ', com os placares em andamento' : ''}` : ''}</span><a class="more" href="#/tabela">tabela completa</a></div>
      <div class="grid g3">
        <div class="panel"><div class="sec-h"><h3>Título</h3></div>${raceBlock('titulo', 4, '242,194,48')}</div>
        <div class="panel"><div class="sec-h"><h3>Libertadores (G4)</h3></div>${raceBlock('libertadores', 6, '79,157,232')}</div>
        <div class="panel"><div class="sec-h"><h3>Rebaixamento</h3></div>${raceBlock('rebaixamento', 6, '239,68,82')}</div>
      </div></section>
    ${last.length ? `<section data-mk="last"><div class="sec-h"><h2>Últimos resultados</h2><a class="more" href="#/jogos">todos</a></div><div class="grid g3">${last.map((m) => matchCard(m, { day: true })).join('')}</div></section>` : ''}
  </div>`;
}
