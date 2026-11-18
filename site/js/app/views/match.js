// Página/bloco de jogo: monta o "view model" do player para jogo real
// (pré, ao vivo, encerrado) e controla o jogo simulado.
import { S, team, forecast, forecastPair, wpTimeline, liveState, loadDetails, lineup, scorers, form, expected, expectedPair, reds } from '../store.js';
import { esc, pct, num, fmtFull, fmtTime, relDay, countdown } from '../util.js';
import { renderPlayer } from '../player.js';
import { colors, scoreMatrix, scorerList, formDots, logo } from '../components.js';
import { simulateMatch } from '../../engine/sim-match.js';
import { scoreDistribution, nextGoal, summarize, compile, timeline } from '../../engine/match-model.js';
import { makeRng } from '../../engine/rng.js';

// ── previsão pré-jogo (corpo da aba "Previsão") ──────────────────────────────
export function preBody(m, f) {
  const H = team(m.h), A = team(m.a), s = f.pre.sum, e = f.e, pr = e.pr;
  const disp = S.model.dispersion || {};
  const sH = scorers(m.h, e.lh), sA = scorers(m.a, e.la);
  const isReal = !String(m.id).startsWith('sim-');
  return `<div class="kv" style="margin-bottom:12px">
      <div><div class="k">Gols esperados</div><div class="v">${num(e.lh, 2)} <small>×</small> ${num(e.la, 2)}</div></div>
      <div><div class="k">Placar mais provável</div><div class="v">${s.top[0].h}–${s.top[0].a} <small>${pct(s.top[0].p)}</small></div></div>
      <div><div class="k">Mais de 2,5 gols</div><div class="v">${pct(s.over25)}</div></div>
      <div><div class="k">Ambos marcam</div><div class="v">${pct(s.btts)}</div></div>
      <div><div class="k">${esc(H.abbr)} não sofre gol</div><div class="v">${pct(s.cleanSheetH)}</div></div>
      <div><div class="k">${esc(A.abbr)} não sofre gol</div><div class="v">${pct(s.cleanSheetA)}</div></div>
      <div><div class="k">Primeiro gol</div><div class="v" style="font-size:17px">${esc(H.abbr)} ${pct(f.pre.next.pH)} · ${esc(A.abbr)} ${pct(f.pre.next.pA)}</div></div>
      <div><div class="k">Alguém expulso</div><div class="v">${pct(1 - (1 - f.pre.pRedH) * (1 - f.pre.pRedA))}</div></div>
    </div>
    <div class="duo">
      <div class="panel"><div class="sec-h"><h3>Mapa de placares</h3></div>${scoreMatrix(f.pre, m)}
        <p class="small muted center" style="margin:10px 0 0">Mais prováveis: ${s.top.slice(0, 5).map((x) => `<span class="chip"><b>${x.h}–${x.a}</b> ${pct(x.p)}</span>`).join(' ')}</p></div>
      <div class="panel"><div class="sec-h"><h3>Números esperados</h3></div>
        <table><tr><th class="l"></th><th>${esc(H.abbr)}</th><th>${esc(A.abbr)}</th></tr>
        <tr><td class="l">Finalizações</td><td>${num(pr.sh.lh)}</td><td>${num(pr.sh.la)}</td></tr>
        <tr><td class="l">No alvo</td><td>${num(pr.sot.lh)}</td><td>${num(pr.sot.la)}</td></tr>
        <tr><td class="l">Escanteios</td><td>${num(pr.co.lh)}</td><td>${num(pr.co.la)}</td></tr>
        <tr><td class="l">Cartões amarelos</td><td>${num(pr.yc.lh)}</td><td>${num(pr.yc.la)}</td></tr>
        <tr><td class="l">Faltas</td><td>${num(pr.fo.lh)}</td><td>${num(pr.fo.la)}</td></tr></table>
        <p class="small muted" style="margin:10px 0 0">Total de gols: ${s.totals.slice(0, 6).map((p, k) => `<span class="chip">${k}${k === 5 ? '+' : ''}: <b>${pct(k === 5 ? s.totals.slice(5).reduce((a, b) => a + b, 0) : p)}</b></span>`).join(' ')}</p>
      </div>
    </div>
    <div class="duo" style="margin-top:14px">
      <div class="panel"><div class="sec-h"><h3>Quem marca · ${esc(H.name)}</h3>${isReal ? `<span class="sub">${formDots(form(m.h))}</span>` : ''}</div>${scorerList(sH)}</div>
      <div class="panel"><div class="sec-h"><h3>Quem marca · ${esc(A.name)}</h3>${isReal ? `<span class="sub">${formDots(form(m.a))}</span>` : ''}</div>${scorerList(sA)}</div>
    </div>
    <p class="small dim" style="margin-top:10px">"Quem marca" usa a escalação provável (titulares recentes). Com a escalação oficial, os números podem mudar.</p>`;
}

const probableLineups = (m) => { const f = (id) => { const L = lineup(id); return { formation: '', provavel: true, xi: L.xi.map((p) => ({ id: p.id, n: p.n, j: p.j, pos: { G: 'GOL', D: 'DEF', M: 'MEI', F: 'ATA' }[p.pos] })), bench: [] }; }; return { h: f(m.h), a: f(m.a) }; };
const goalList = (m) => { const out = { h: [], a: [] }; for (const e of m.ev) if ('GPO'.includes(e.t)) out[e.s].push({ n: e.n || 'Gol', t: `${e.m}'${e.x ? '+' + e.x : ''}${e.t === 'P' ? ' (pên.)' : e.t === 'O' ? ' (contra)' : ''}` }); return out; };
const evText = (m, e) => { const T = team(e.s === 'h' ? m.h : m.a).name; return e.t === 'R' ? `Cartão vermelho: ${e.n || ''} (${T})` : e.t === 'Y' ? `Cartão amarelo: ${e.n || ''} (${T})` : e.t === 'O' ? `Gol contra! ${e.n || ''} marca para o ${T}` : `Gol do ${T}! ${e.n || ''}${e.t === 'P' ? ' (pênalti)' : ''}`; };

/** View model de um jogo real. */
export function realVM(m) {
  const f = forecast(m);
  const mode = m.status === 'in' ? 'live' : m.status === 'post' ? 'post' : 'pre';
  if (mode !== 'pre' || Math.abs(new Date(m.date) - Date.now()) < 3 * 3600000) loadDetails(m);
  const det = S.details.get(m.id);
  const base = { key: `m${m.id}`, m, mode, colors: colors(m), e: f.e, pre: f.pre, probs: f.now, dist: f.now, next: f.now.next, info: `${fmtFull(m.date)}${m.venue ? ' · ' + m.venue : ''}` };
  if (mode === 'pre') {
    return { ...base, score: null, clock: m.status === 'off' ? 'Adiado' : `${relDay(m.date)} · ${fmtTime(m.date)}`, oracleRight: m.status === 'off' ? '' : `começa ${countdown(m.date)}`, preBody: preBody(m, f), lineups: det?.lineups || probableLineups(m), events: [] };
  }
  const derived = m.ev.filter((e) => e.t !== 'Y' || mode === 'live').map((e) => ({ label: `${e.m}'${e.x ? '+' + e.x : ''}`, type: e.t === 'R' ? 'red' : e.t === 'Y' ? 'yellow' : e.t === 'O' ? 'own' : 'goal', side: e.s, text: evText(m, e) })).reverse();
  const events = det?.commentary?.length ? det.commentary : det?.key?.length ? det.key : derived;
  const st = det?.stats || (m.stats && m.stats.h.sh != null ? { h: { ...m.stats.h, yc: m.ev.filter((e) => e.t === 'Y' && e.s === 'h').length, rc: reds(m, 'h') }, a: { ...m.stats.a, yc: m.ev.filter((e) => e.t === 'Y' && e.s === 'a').length, rc: reds(m, 'a') } } : null);
  return {
    ...base, score: [m.hg, m.ag], reds: [reds(m, 'h'), reds(m, 'a')], scorers: goalList(m),
    clock: mode === 'post' ? 'Fim de jogo' : /HALFTIME/.test(m.statusName) ? 'Intervalo' : m.clock || 'Ao vivo',
    oracleRight: mode === 'live' ? `antes do jogo: ${pct(f.pre.pH)} · ${pct(f.pre.pD)} · ${pct(f.pre.pA)}` : '',
    tl: wpTimeline(m), events, latest: (det?.key?.length ? det.key : derived).slice(0, 3), stats: st, lineups: det?.lineups || null,
  };
}

// ── jogo simulado ────────────────────────────────────────────────────────────
export const sim = { key: null, m: null, res: null, pos: 0, playing: false, dur: 150, timer: null, wp: [], cfg: null, onTick: null };
const TICK = 200;
export function stopSim() { clearInterval(sim.timer); sim.timer = null; sim.playing = false; }
export function closeSim() { stopSim(); sim.key = null; sim.res = null; }
export function startSim(m, onTick) {
  stopSim();
  const real = !String(m.id).startsWith('sim-');
  const e = real ? expected(m) : expectedPair(m.h, m.a);
  const cfg = { home: { name: team(m.h).name, lineup: lineup(m.h) }, away: { name: team(m.a).name, lineup: lineup(m.a) }, lh: e.lh, la: e.la, rates: { sh: [e.pr.sh.lh, e.pr.sh.la], sot: [e.pr.sot.lh, e.pr.sot.la], co: [e.pr.co.lh, e.pr.co.la], yc: [e.pr.yc.lh, e.pr.yc.la], fo: [e.pr.fo.lh, e.pr.fo.la] } };
  Object.assign(sim, { key: `s${m.id}`, m, res: simulateMatch(S.model, cfg, makeRng((Math.random() * 2 ** 32) >>> 0)), pos: 0, wp: [], cfg, e, onTick });
  playSim();
}
export function playSim() {
  if (!sim.res) return;
  if (sim.pos >= sim.res.steps) { sim.pos = 0; sim.wp = []; }
  sim.playing = true; clearInterval(sim.timer);
  sim.timer = setInterval(() => {
    sim.pos = Math.min(sim.res.steps, sim.pos + (sim.res.steps / sim.dur) * (TICK / 1000));
    if (sim.pos >= sim.res.steps) stopSim();
    sim.onTick?.();
  }, TICK);
}
export function simAction(act, val) {
  if (act === 'pause') stopSim();
  else if (act === 'play') playSim();
  else if (act === 'end') { sim.pos = sim.res.steps; stopSim(); }
  else if (act === 'speed') { sim.dur = +val; if (sim.playing) playSim(); }
  else if (act === 'again') startSim(sim.m, sim.onTick);
  else if (act === 'close') closeSim();
  sim.onTick?.();
}
function simState(step) {
  let hg = 0, ag = 0, rh = 0, ra = 0;
  for (const e of sim.res.events) { if (e.t >= step) break; if (e.type === 'goal') (e.side === 'h' ? hg++ : ag++); else if (e.type === 'red') (e.side === 'h' ? rh++ : ra++); }
  return { step, hg, ag, rh, ra };
}
export function simVM() {
  const m = sim.m, hz = S.model.hazard, n = sim.res.steps, step = Math.min(n, Math.floor(sim.pos));
  const done = step >= n;
  // curva de probabilidade incremental
  for (let t = sim.wp.length; t <= step; t++) { const d = scoreDistribution(hz, sim.e.lh, sim.e.la, simState(t)); sim.wp.push({ t, pH: d.pH, pD: d.pD, pA: d.pA }); }
  const st = simState(step);
  const d = scoreDistribution(hz, sim.e.lh, sim.e.la, st);
  const pre = scoreDistribution(hz, sim.e.lh, sim.e.la);
  const shown = sim.res.events.filter((e) => e.t < step || (done && e.t <= n) || (e.t === 0 && e.type === 'kickoff'));
  const stats = { h: { sh: 0, sot: 0, co: 0, yc: 0, rc: 0 }, a: { sh: 0, sot: 0, co: 0, yc: 0, rc: 0 } };
  const scorersL = { h: [], a: [] };
  for (const e of shown) {
    if (!e.side) continue; const s = stats[e.side];
    if (e.type === 'goal') { if (e.kind !== 'own') { s.sh++; s.sot++; } scorersL[e.side].push({ n: e.player, t: `${e.label}${e.kind === 'pen' ? ' (pên.)' : e.kind === 'own' ? ' (contra)' : ''}` }); }
    else if (e.type === 'save') { s.sh++; s.sot++; } else if (e.type === 'miss') s.sh++; else if (e.type === 'corner') s.co++; else if (e.type === 'yellow') s.yc++; else if (e.type === 'red') s.rc++;
  }
  if (done) { stats.h.po = sim.res.stats.h.po; stats.a.po = sim.res.stats.a.po; stats.h.fo = sim.res.stats.h.fo; stats.a.fo = sim.res.stats.a.fo; }
  const tlSteps = timeline(hz);
  const half = step === 45 + hz.S1;
  const evs = shown.filter((e) => e.type === 'goal' || e.type === 'red').map((e) => ({ step: e.t, t: e.type === 'red' ? 'R' : 'G', s: e.side, n: e.player, m: parseInt(e.label) || 0, x: 0 }));
  const speeds = [[45, 'Rápido'], [150, 'Normal'], [420, 'Lento']];
  const controls = `<div class="simctl">
      ${done ? '<button class="btn sm gold" data-act="sim" data-do="again">↻ Simular de novo</button>' : sim.playing ? '<button class="btn sm" data-act="sim" data-do="pause">❚❚ Pausar</button>' : '<button class="btn sm gold" data-act="sim" data-do="play">▶ Continuar</button>'}
      ${done ? '' : '<button class="btn sm" data-act="sim" data-do="end">⏭ Ir para o fim</button>'}
      ${speeds.map(([v, l]) => `<button class="btn sm ${sim.dur === v ? 'on' : ''}" data-act="sim" data-do="speed" data-val="${v}">${l}</button>`).join('')}
      <span class="progress"><i style="width:${(sim.pos / n) * 100}%"></i></span>
      <button class="btn sm" data-act="sim" data-do="close">✕ Fechar</button>
    </div>`;
  const lu = (id) => { const L = lineup(id); return { formation: '', provavel: true, xi: L.xi.map((p) => ({ n: p.n, j: p.j, pos: { G: 'GOL', D: 'DEF', M: 'MEI', F: 'ATA' }[p.pos] })), bench: [] }; };
  return {
    key: sim.key, m, mode: 'sim', colors: colors(m), e: sim.e, pre, probs: d, dist: d, next: done ? null : nextGoal(hz, sim.e.lh, sim.e.la, st),
    score: [st.hg, st.ag], reds: [st.rh, st.ra], scorers: scorersL, clock: done ? 'Fim de jogo' : half ? 'Intervalo' : tlSteps[Math.min(n - 1, step)].label,
    info: 'Uma das infinitas versões possíveis deste jogo', oracleRight: `antes do jogo: ${pct(pre.pH)} · ${pct(pre.pD)} · ${pct(pre.pA)}`,
    tl: { points: sim.wp.slice(0, step + 1), events: evs, total: n }, events: shown.filter((e) => e.type !== 'corner' || true).map((e) => ({ label: e.label, type: e.type === 'goal' && e.kind === 'own' ? 'own' : e.type, side: e.side, text: e.text })).reverse(),
    latest: shown.filter((e) => ['goal', 'red', 'yellow', 'sub'].includes(e.type)).slice(-3).reverse().map((e) => ({ label: e.label, type: e.type, side: e.side, text: e.text })), stats, lineups: { h: lu(m.h), a: lu(m.a) }, controls,
  };
}

/** Bloco completo (player) de um jogo; se houver simulação ativa para ele, mostra a simulação. */
export function matchBlock(m) {
  if (sim.res && sim.key === `s${m.id}`) return renderPlayer(simVM());
  return renderPlayer(realVM(m));
}

export function matchView(id) {
  const m = S.matches.get(id);
  if (!m) return `<div class="wrap"><p class="muted">Jogo não encontrado. <a class="gold" href="#/">Voltar ao início</a></p></div>`;
  const simOn = sim.res && sim.key === `s${m.id}`;
  return `<div class="wrap"><section>${matchBlock(m)}</section>
    ${m.status === 'pre' && !simOn ? `<section class="btnrow"><button class="btn gold" data-act="simstart" data-id="${m.id}">▶ Assistir uma simulação deste jogo</button><span class="muted small">Cada simulação é diferente — o placar segue as probabilidades acima.</span></section>` : ''}
  </div>`;
}

/** Simulador livre: qualquer mandante × qualquer visitante. */
export function pairMatch(h, a) { return { id: `sim-${h}-${a}`, h, a, status: 'pre', date: new Date().toISOString(), ev: [], season: S.season, venue: null }; }
export function pairBlock(h, a) {
  const m = pairMatch(h, a);
  if (sim.res && sim.key === `s${m.id}`) return renderPlayer(simVM());
  const f = forecastPair(h, a);
  const L = (id) => { const x = lineup(id); return { formation: '', provavel: true, xi: x.xi.map((p) => ({ n: p.n, j: p.j, pos: { G: 'GOL', D: 'DEF', M: 'MEI', F: 'ATA' }[p.pos] })), bench: [] }; };
  return renderPlayer({ key: `m${m.id}`, m, mode: 'pre', colors: colors(m), e: f.e, pre: f.pre, probs: f.pre, dist: f.pre, score: null, clock: 'Confronto hipotético', info: `${team(h).name} em casa`, preBody: preBody(m, f), lineups: { h: L(h), a: L(a) }, events: [] });
}
