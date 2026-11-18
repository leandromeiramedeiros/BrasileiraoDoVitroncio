// Bloco de jogo: placar, relógio, probabilidades, gráfico, lances,
// estatísticas e escalações. Serve para jogo ao vivo, encerrado, pré-jogo e
// para o jogo simulado.
//
// O HTML é sempre gerado por inteiro, mas quem aplica no DOM é o morph.js, que
// só mexe no que mudou. Por isso as listas usam data-mk (cada lance mantém o
// mesmo elemento entre atualizações) e nada pisca.
import { team } from './store.js';
import { esc, pct, num } from './util.js';
import { logo, probBar, eventRow, statBars, scoreMatrix, wpChart } from './components.js';

const tabState = new Map();
const feedState = new Map();
export function setTab(key, tab) { tabState.set(key, tab); }
export function setFeed(key, mode) { feedState.set(key, mode); }

/**
 * vm = { key, m, mode: 'live'|'post'|'pre'|'sim', compact, score:[h,a]|null, clock, reds:[h,a],
 *        scorers:{h:[..],a:[..]}, probs, pre, dist, next, tl, events:[], keyEvents:[],
 *        stats, lineups, controls (html), preBody (html), strip (html), info, oracleRight }
 * `events` e `keyEvents` vêm do mais recente para o mais antigo.
 */
export function renderPlayer(vm) {
  const m = vm.m, H = team(m.h), A = team(m.a);
  const tab = tabState.get(vm.key) || (vm.mode === 'pre' ? 'previsao' : 'lances');
  const tag = vm.mode === 'live' ? '<span class="livepill">Ao vivo</span>' : vm.mode === 'sim' ? '<span class="tag sim">Simulação</span>' : vm.mode === 'post' ? '<span class="tag">Encerrado</span>' : '<span class="tag">Pré-jogo</span>';
  const redsHtml = (n) => (n ? `<span class="rc">${'<i></i>'.repeat(n)}</span>` : '');
  const shown = vm.mode === 'post' ? vm.pre : vm.probs;
  const label = vm.mode === 'post' ? 'Probabilidades antes do jogo' : vm.mode === 'pre' ? 'Probabilidades' : 'Probabilidades agora';
  const sc = (side) => (vm.scorers?.[side] || []).map((s) => `<b>${esc(s.n)}</b> ${esc(s.t)}`).join(' · ');

  const stage = `<div class="stage">
    <div class="stage-top"><span>${tag}</span><span>${esc(vm.info || '')}</span></div>
    <div class="board">
      <a class="side" href="#/time/${m.h}">${logo(H)}<span class="nm">${esc(H.name)}</span>${redsHtml(vm.reds?.[0])}</a>
      ${vm.score ? `<div class="digits"><span class="n" data-flash>${vm.score[0]}</span><span class="x">×</span><span class="n" data-flash>${vm.score[1]}</span></div>` : '<div class="digits vs">×</div>'}
      <a class="side" href="#/time/${m.a}">${logo(A)}<span class="nm">${esc(A.name)}</span>${redsHtml(vm.reds?.[1])}</a>
    </div>
    <div class="clock"><span class="${vm.mode === 'live' || vm.mode === 'sim' ? 'on' : ''}">${esc(vm.clock || '')}</span></div>
    ${vm.scorers && (vm.scorers.h.length || vm.scorers.a.length) ? `<div class="scorers" data-mk="scorers"><div>${sc('h')}</div><div>${sc('a')}</div></div>` : ''}
    ${vm.controls ? `<div data-mk="controls">${vm.controls}</div>` : ''}
    <div class="oracle" data-mk="oracle"><div class="oracle-h"><span>${label}</span><span>${esc(vm.oracleRight || '')}</span></div>
      ${probBar(shown)}
      <div class="plabels"><span><b>${pct(shown.pH)}</b> ${esc(H.name)}</span><span>empate <b>${pct(shown.pD)}</b></span><span>${esc(A.name)} <b>${pct(shown.pA)}</b></span></div>
    </div>
    ${vm.strip ? `<div data-mk="strip">${vm.strip}</div>` : ''}
  </div>`;
  if (vm.compact) return `<div style="${vm.colors}">${stage}</div>`;

  const tabs = [['lances', 'Lances'], ['previsao', vm.mode === 'pre' ? 'Previsão' : 'Probabilidades'], ['stats', 'Estatísticas'], ['escalacoes', 'Escalações']].filter(([k]) => !(vm.mode === 'pre' && (k === 'lances' || k === 'stats')));
  let body = '';
  if (tab === 'lances') {
    const all = vm.events || [], key = vm.keyEvents || [];
    // "Principais" é o padrão sempre que existem lances principais; o seletor só aparece quando há diferença
    const canFilter = key.length > 0 && all.length > key.length;
    const mode = key.length ? feedState.get(vm.key) || 'principais' : 'todos';
    const list = mode === 'principais' ? key : all;
    const pre = mode === 'principais' ? 'k' : 'e';
    body = `${canFilter ? `<div class="feed-h" data-mk="feed-h"><div class="seg">${[['principais', 'Principais'], ['todos', 'Todos os lances']].map(([k, l]) => `<button class="${k === mode ? 'on' : ''}" data-act="feed" data-key="${esc(vm.key)}" data-mode="${k}">${l}</button>`).join('')}</div></div>` : ''}
      ${list.length ? `<div class="feed" data-mk="feed-${mode}">${list.map((e, i) => eventRow(e, `${pre}${list.length - i}`)).join('')}</div>` : `<p class="muted">${vm.mode === 'live' ? 'Aguardando os primeiros lances.' : 'Sem lances registrados para este jogo.'}</p>`}`;
  } else if (tab === 'stats') {
    body = vm.stats ? `<div class="panel">${statBars(vm.stats)}</div>` : '<p class="muted">Estatísticas ainda não disponíveis.</p>';
  } else if (tab === 'escalacoes') {
    const col = (side, T) => {
      const L = vm.lineups?.[side];
      if (!L) return `<div class="panel"><h3>${esc(T.name)}</h3><p class="muted small">Escalação ainda não divulgada.</p></div>`;
      return `<div class="panel"><h3 style="margin-bottom:6px">${esc(T.name)} <span class="dim small">${esc(L.formation || '')}${L.provavel ? ' provável' : ''}</span></h3><div class="lineup">${L.xi.map((p) => `<div class="${p.out ? 'off' : ''}"><span class="j">${esc(p.j || '')}</span><span>${esc(p.n)}${p.out ? ' ↓' : ''}</span><span class="po">${esc(p.pos || '')}</span></div>`).join('')}</div>${L.bench?.some((p) => p.in) ? `<p class="small muted" style="margin:8px 0 2px">Entraram</p><div class="lineup">${L.bench.filter((p) => p.in).map((p) => `<div><span class="j">${esc(p.j || '')}</span><span>${esc(p.n)} ↑</span></div>`).join('')}</div>` : ''}</div>`;
    };
    body = `<div class="duo">${col('h', H)}${col('a', A)}</div>`;
  } else if (vm.mode === 'pre') {
    body = vm.preBody || '';
  } else {
    const d = vm.dist;
    body = `${vm.tl ? `<div class="panel" style="margin-bottom:12px"><div class="sec-h"><h3>Probabilidade de vitória ao longo do jogo</h3></div>${wpChart(vm.tl, m, { total: vm.tlTotal })}</div>` : ''}
      <div class="duo"><div class="panel"><div class="sec-h"><h3>${vm.mode === 'post' ? 'Placar final' : 'Placar final provável'}</h3></div>${scoreMatrix(d, m, { now: vm.score })}</div>
      <div><div class="kv">
        <div><div class="k">Antes do jogo</div><div class="v s">${pct(vm.pre.pH)} · ${pct(vm.pre.pD)} · ${pct(vm.pre.pA)}</div></div>
        <div><div class="k">Gols esperados (pré-jogo)</div><div class="v s">${num(vm.e.lh, 2)} × ${num(vm.e.la, 2)}</div></div>
        ${vm.next && vm.mode !== 'post' ? `<div><div class="k">Próximo gol</div><div class="v s">${esc(H.abbr)} ${pct(vm.next.pH)} · ${esc(A.abbr)} ${pct(vm.next.pA)}</div></div><div><div class="k">Sem mais gols</div><div class="v">${pct(vm.next.pNone)}</div></div><div><div class="k">Gols até o fim</div><div class="v">${num(d.xgH + d.xgA - (vm.score[0] + vm.score[1]))}</div></div>` : ''}
      </div></div></div>`;
  }
  return `<div style="${vm.colors}">${stage}
  <div class="tabs" data-mk="tabs">${tabs.map(([k, l]) => `<button class="${k === tab ? 'on' : ''}" data-act="tab" data-key="${esc(vm.key)}" data-tab="${k}">${l}</button>`).join('')}</div>
  <div data-mk="tab-${tab}">${body}</div></div>`;
}
