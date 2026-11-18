// Página inicial: jogo ao vivo em destaque (ou o próximo), próximos jogos,
// a corrida pelo título/contra o rebaixamento e os últimos resultados.
import { S, team, liveMatches, upcoming, finished, forecast, table } from '../store.js';
import { esc, pct, dayKey, relDay, countdown, fmtTime, store } from '../util.js';
import { matchCard, raceBlock, logo, fav } from '../components.js';
import { matchBlock, sim } from './match.js';

export const homeState = { selected: null };

export function homeView() {
  const live = liveMatches(), up = upcoming(), fin = finished();
  const today = dayKey(new Date().toISOString());
  let hero = '';
  // se há simulação aberta a partir da home, ela ocupa o palco
  const simM = sim.res && sim.m && S.matches.get(sim.m.id) && homeState.simHere ? S.matches.get(sim.m.id) : null;
  if (simM) hero = `<section>${matchBlock(simM)}</section>`;
  else if (live.length) {
    const sel = live.find((m) => m.id === homeState.selected) || live[0];
    const others = [...live, ...up.filter((m) => dayKey(m.date) === today), ...fin.filter((m) => dayKey(m.date) === today)];
    hero = `<section><div class="sec-h"><h2>Ao vivo agora</h2><span class="sub">${live.length} jogo${live.length > 1 ? 's' : ''} em andamento · atualiza sozinho</span></div>
      ${matchBlock(sel)}
      ${others.length > 1 ? `<div class="strip">${others.map((m) => matchCard(m, { selected: m.id === sel.id, act: m.status === 'in' ? 'selectlive' : null })).join('')}</div>` : ''}</section>`;
  } else if (up.length) {
    const nx = up[0], f = forecast(nx), H = team(nx.h), A = team(nx.a);
    hero = `<section><div class="sec-h"><h2>Próximo jogo</h2><span class="sub">${esc(relDay(nx.date))} às ${fmtTime(nx.date)} · ${countdown(nx.date)}</span></div>
      ${matchBlock(nx)}
      <div class="btnrow" style="margin-top:12px"><button class="btn gold" data-act="simstart" data-id="${nx.id}" data-home="1">▶ Assistir uma simulação</button><a class="btn" href="#/jogo/${nx.id}">Abrir página do jogo</a></div></section>`;
  } else {
    hero = `<section class="panel center"><h2>Temporada encerrada</h2><p class="muted">Sem jogos marcados no momento. A tabela final e o histórico continuam disponíveis.</p></section>`;
  }

  // próximos jogos agrupados por dia
  const rest = up.filter((m) => !(live.length === 0 && m === up[0] && !simM) || true).slice(0, 20);
  const groups = new Map();
  for (const m of rest) { const k = dayKey(m.date); if (!groups.has(k)) groups.set(k, []); groups.get(k).push(m); }
  const upcomingHtml = [...groups.entries()].slice(0, 5).map(([k, list]) => `<div style="margin-bottom:16px"><div class="sec-h"><h3 style="font-size:16px;color:var(--gold)">${esc(relDay(list[0].date))}</h3><span class="sub">${list.length} jogo${list.length > 1 ? 's' : ''}</span></div><div class="grid g3">${list.map((m) => matchCard(m)).join('')}</div></div>`).join('');

  const f = fav();
  const favTeam = f && S.teams[f];
  const favRow = favTeam && table().find((r) => r.id === f);
  const favSim = favTeam && S.seasonSim?.teams[f];
  const favNext = favTeam && up.find((m) => m.h === f || m.a === f);
  const favHtml = favTeam && favRow ? `<section class="panel" style="border-color:rgba(31,209,127,.5)"><div class="sec-h"><h2 style="display:flex;align-items:center;gap:10px">${logo(favTeam)} ${esc(favTeam.name)}</h2><span class="sub">seu time</span><a class="more" href="#/time/${f}">ver tudo →</a></div>
      <div class="kv"><div><div class="k">Posição</div><div class="v">${favRow.rank}º <small>${favRow.pts} pts</small></div></div>
      ${favSim ? `<div><div class="k">Título</div><div class="v">${pct(favSim.titulo)}</div></div><div><div class="k">Libertadores (G4)</div><div class="v">${pct(favSim.libertadores)}</div></div><div><div class="k">Rebaixamento</div><div class="v">${pct(favSim.rebaixamento)}</div></div><div><div class="k">Pontos projetados</div><div class="v">${favSim.pts.toFixed(0)} <small>${favSim.ptsLo}–${favSim.ptsHi}</small></div></div>` : ''}
      ${favNext ? `<div><div class="k">Próximo jogo</div><div class="v" style="font-size:16px"><a href="#/jogo/${favNext.id}">${esc(team(favNext.h).abbr)} × ${esc(team(favNext.a).abbr)} · ${esc(relDay(favNext.date))}</a></div></div>` : ''}</div></section>` : '';

  const last = fin.slice(-9).reverse();
  return `<div class="wrap">
    ${S.error ? `<div class="banner">${esc(S.error)}</div>` : ''}
    ${hero}
    ${favHtml}
    <section><div class="sec-h"><h2>A corrida</h2><span class="sub">${S.seasonSim ? `${S.seasonSim.n.toLocaleString('pt-BR')} temporadas simuladas${liveMatches().length ? ' · já contando os placares de agora' : ''}` : 'simulando o resto do campeonato…'}</span><a class="more" href="#/tabela">tabela completa →</a></div>
      <div class="grid g3">
        <div class="panel"><div class="sec-h"><h3>🏆 Título</h3></div>${raceBlock('titulo', 5, '255,210,31')}</div>
        <div class="panel"><div class="sec-h"><h3>Libertadores (G4)</h3></div>${raceBlock('libertadores', 7, '69,167,255')}</div>
        <div class="panel"><div class="sec-h"><h3>Rebaixamento (Z4)</h3></div>${raceBlock('rebaixamento', 7, '255,61,77')}</div>
      </div></section>
    ${upcomingHtml ? `<section><div class="sec-h"><h2>Próximos jogos</h2><span class="sub">chance de vitória de cada lado, segundo o Vitrôncio</span></div>${upcomingHtml}</section>` : ''}
    ${last.length ? `<section><div class="sec-h"><h2>Últimos resultados</h2></div><div class="grid g3">${last.map((m) => matchCard(m, { day: true })).join('')}</div></section>` : ''}
  </div>`;
}
