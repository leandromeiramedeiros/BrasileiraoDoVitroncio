// "Player" de jogo: placar gigante, relógio, chances do Vitrôncio, gráfico,
// lances, estatísticas e escalações. Serve para jogo ao vivo, encerrado,
// pré-jogo e para o jogo simulado.
import { team } from './store.js';
import { esc, pct, num } from './util.js';
import { logo, probBar, eventRow, statBars, scoreMatrix, wpChart } from './components.js';

const tabState = new Map();
export function setTab(key, tab) { tabState.set(key, tab); }
let lastScore = new Map();

/**
 * vm = { key, m, mode: 'live'|'post'|'pre'|'sim', score:[h,a]|null, clock, statusText, reds:[h,a],
 *        scorers:{h:[..],a:[..]}, probs:{pH,pD,pA}, pre:{...}, dist, next, tl, events:[], latest:[],
 *        stats, lineups, extraTop, controls (html), preBody (html) }
 */
export function renderPlayer(vm) {
  const m = vm.m, H = team(m.h), A = team(m.a);
  const tab = tabState.get(vm.key) || (vm.mode === 'pre' ? 'previsao' : 'lances');
  const prev = lastScore.get(vm.key);
  const bump = (i) => (prev && vm.score && vm.score[i] > prev[i] ? 'bump' : '');
  if (vm.score) lastScore.set(vm.key, vm.score.slice());
  const tag = vm.mode === 'live' ? '<span class="livepill">Ao vivo</span>' : vm.mode === 'sim' ? '<span class="tag sim">Simulação do Vitrôncio</span>' : vm.mode === 'post' ? '<span class="tag end">Encerrado</span>' : '<span class="tag pre">Pré-jogo</span>';
  const redsHtml = (n) => (n ? `<span class="rc">${'<i></i>'.repeat(n)}</span>` : '');
  const tabs = [['lances', 'Lances'], ['previsao', vm.mode === 'pre' ? 'Previsão' : 'Chances'], ['stats', 'Estatísticas'], ['escalacoes', 'Escalações']].filter(([k]) => !(vm.mode === 'pre' && k === 'lances') && !(vm.mode === 'pre' && k === 'stats'));
  let body = '';
  if (tab === 'lances') {
    body = vm.events?.length ? `<div class="feed">${vm.events.map((e, i) => eventRow(e, vm.mode !== 'post' && i === 0)).join('')}</div>` : `<p class="muted">${vm.mode === 'live' ? 'Aguardando os primeiros lances…' : 'Sem lances registrados para este jogo.'}</p>`;
  } else if (tab === 'stats') {
    body = vm.stats ? statBars(vm.stats) : '<p class="muted">Estatísticas ainda não disponíveis.</p>';
  } else if (tab === 'escalacoes') {
    const col = (side, T) => { const L = vm.lineups?.[side]; if (!L) return `<div><h3>${esc(T.name)}</h3><p class="muted small">Escalação ainda não divulgada.</p></div>`; return `<div><h3 style="margin-bottom:6px">${esc(T.name)} <span class="dim small">${esc(L.formation || '')}${L.provavel ? ' · provável' : ''}</span></h3><div class="lineup">${L.xi.map((p) => `<div class="${p.out ? 'off' : ''}"><span class="j">${esc(p.j || '')}</span><span>${esc(p.n)}${p.out ? ' ↓' : ''}</span><span class="po">${esc(p.pos || '')}</span></div>`).join('')}</div>${L.bench?.some((p) => p.in) ? `<p class="small muted" style="margin:8px 0 2px">Entraram</p><div class="lineup">${L.bench.filter((p) => p.in).map((p) => `<div><span class="j">${esc(p.j || '')}</span><span>${esc(p.n)} ↑</span></div>`).join('')}</div>` : ''}</div>`; };
    body = `<div class="duo">${col('h', H)}${col('a', A)}</div>`;
  } else {
    body = vm.preBody || '';
    if (vm.mode !== 'pre') {
      const d = vm.dist;
      body = `${vm.tl ? `<div class="panel" style="margin-bottom:12px"><div class="sec-h"><h3>Chance de vitória minuto a minuto</h3></div>${wpChart(vm.tl, m, { total: vm.tlTotal })}</div>` : ''}
      <div class="duo"><div class="panel"><div class="sec-h"><h3>${vm.mode === 'post' ? 'Placar final' : 'Placar final provável'}</h3></div>${scoreMatrix(d, m, { now: vm.score })}</div>
      <div><div class="kv">
        <div><div class="k">Antes do jogo</div><div class="v" style="font-size:17px">${pct(vm.pre.pH)} · ${pct(vm.pre.pD)} · ${pct(vm.pre.pA)}</div></div>
        ${vm.next && vm.mode !== 'post' ? `<div><div class="k">Próximo gol</div><div class="v" style="font-size:17px">${esc(H.abbr)} ${pct(vm.next.pH)} · ${esc(A.abbr)} ${pct(vm.next.pA)}</div></div><div><div class="k">Não sai mais gol</div><div class="v">${pct(vm.next.pNone)}</div></div><div><div class="k">Gols até o fim</div><div class="v">${num(d.xgH + d.xgA - (vm.score[0] + vm.score[1]))}</div></div>` : ''}
        <div><div class="k">Gols esperados (pré)</div><div class="v" style="font-size:17px">${num(vm.e.lh, 2)} × ${num(vm.e.la, 2)}</div></div>
      </div></div></div>`;
    }
  }
  const sc = (side) => (vm.scorers?.[side] || []).map((s) => `<b>${esc(s.n)}</b> ${esc(s.t)}`).join(' · ');
  return `<div class="stage" style="${vm.colors}">
    <div class="stage-top"><span>${tag}</span><span>${esc(vm.info || '')}</span></div>
    <div class="board">
      <a class="side" href="#/time/${m.h}">${logo(H)}<span class="nm">${esc(H.name)}</span>${redsHtml(vm.reds?.[0])}</a>
      ${vm.score ? `<div class="digits"><span class="n ${bump(0)}">${vm.score[0]}</span><span class="x">×</span><span class="n ${bump(1)}">${vm.score[1]}</span></div>` : '<div class="digits vs">×</div>'}
      <a class="side" href="#/time/${m.a}">${logo(A)}<span class="nm">${esc(A.name)}</span>${redsHtml(vm.reds?.[1])}</a>
    </div>
    <div class="clock"><span class="${vm.mode === 'live' || vm.mode === 'sim' ? 'on' : ''}">${esc(vm.clock || '')}</span></div>
    ${vm.scorers && (vm.scorers.h.length || vm.scorers.a.length) ? `<div class="scorers"><div>${sc('h')}</div><div>${sc('a')}</div></div>` : ''}
    ${vm.controls || ''}
    <div class="oracle"><div class="oracle-h"><span>${vm.mode === 'post' ? 'O que o Vitrôncio dizia antes do jogo' : vm.mode === 'pre' ? 'Vitrôncio prevê' : 'Vitrôncio agora'}</span><span>${vm.oracleRight || ''}</span></div>
      ${probBar(vm.mode === 'post' ? vm.pre : vm.probs)}
      <div class="plabels"><span><b>${pct((vm.mode === 'post' ? vm.pre : vm.probs).pH)}</b> ${esc(H.name)}</span><span>empate <b>${pct((vm.mode === 'post' ? vm.pre : vm.probs).pD)}</b></span><span>${esc(A.name)} <b>${pct((vm.mode === 'post' ? vm.pre : vm.probs).pA)}</b></span></div>
    </div>
  </div>
  ${vm.latest?.length && tab !== 'lances' ? `<div class="latest" style="margin-top:12px">${vm.latest.map((e) => eventRow(e)).join('')}</div>` : ''}
  <div class="tabs">${tabs.map(([k, l]) => `<button class="${k === tab ? 'on' : ''}" data-act="tab" data-key="${esc(vm.key)}" data-tab="${k}">${l}</button>`).join('')}</div>
  <div>${body}</div>`;
}
