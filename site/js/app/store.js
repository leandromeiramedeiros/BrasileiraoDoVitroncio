// Estado central do site: modelo (foto semanal) + dados dinâmicos da ESPN.
// Os ratings são atualizados aqui mesmo, no navegador, a cada jogo encerrado
// depois da foto — por isso as previsões nunca ficam "paradas" na semana.
import * as api from './api.js';
import { parseScoreboard, parseClock, parseOdds, impliedProbs } from '../data/espn-parse.js';
import { store as ls } from './util.js';
import * as R from '../engine/ratings.js';
import { scoreDistribution, nextGoal, summarize, stepOf, compile } from '../engine/match-model.js';
import { buildTable, sortTable } from '../engine/season.js';
import { probableLineup, scorerProbabilities, genericLineup } from '../engine/players.js';

export const S = {
  model: null, state: null, teams: {}, season: null,
  matches: new Map(), applied: new Set(), lam: new Map(), details: new Map(),
  odds: new Map(),
  seasonSim: null, simKey: '', simRunning: false,
  online: null, lastUpdate: null, version: 0, error: null,
};
const listeners = new Set();
export const subscribe = (fn) => { listeners.add(fn); return () => listeners.delete(fn); };
let emitQueued = false;
function emit() { if (emitQueued) return; emitQueued = true; queueMicrotask(() => { emitQueued = false; for (const fn of listeners) { try { fn(); } catch (e) { console.error(e); } } }); }

// ── carga inicial ────────────────────────────────────────────────────────────
export async function boot() {
  const model = await (await fetch('data/model.json', { cache: 'no-cache' })).json();
  S.model = model; S.state = structuredClone(model.state); S.teams = { ...model.teams }; S.season = model.temporada;
  S.applied = new Set(model.applied);
  for (const f of model.fixtures) {
    S.matches.set(f.id, { ...f, lg: 'A', season: model.temporada, ev: (f.ev || []).map((e) => ({ m: e[0], x: e[1], t: e[2], s: e[3], n: e[4] })), stats: null, clock: '', period: 0, statusName: '', snapshot: true });
    if (f.lam) S.lam.set(f.id, { lh: f.lam[0], la: f.lam[1] });
    if (f.mkt) S.odds.set(f.id, { dec: f.mkt, src: 'média de fechamento do mercado', kind: 'fechamento' });
  }
  // odds vistas neste navegador antes de o jogo começar (ficam guardadas para a comparação depois)
  for (const [id, o] of Object.entries(ls.get('odds', {}))) if (!S.odds.has(id) && S.matches.has(id)) S.odds.set(id, o);
  S.version++; emit();
  refreshSeason().then(() => poll());
  document.addEventListener('visibilitychange', () => { if (!document.hidden) poll(true); });
}

let oddsDirty = false;
function merge(json) {
  const { matches, teams, seasonYear } = parseScoreboard(json, 'A');
  let changed = false;
  for (const [id, t] of Object.entries(teams)) if (!S.teams[id] || S.teams[id].name !== t.name) { S.teams[id] = { ...S.teams[id], ...t }; changed = true; }
  if (seasonYear && seasonYear > S.season && matches.some((m) => m.season === seasonYear)) {
    // virou a temporada: o calendário novo substitui o antigo
    S.season = seasonYear;
    for (const [id, m] of S.matches) if (m.season < seasonYear) S.matches.delete(id);
    changed = true;
  }
  for (const m of matches) {
    if (m.season !== S.season) continue;
    const prev = S.matches.get(m.id);
    const sig = (x) => x && `${x.status}|${x.hg}|${x.ag}|${x.clock}|${x.date}|${x.ev.length}|${x.statsOk}`;
    if (sig(prev) !== sig(m) || prev?.snapshot) { S.matches.set(m.id, m); changed = true; }
    // odds: só as vistas ANTES do jogo valem como palpite pré-jogo do mercado
    if (m.odds && m.status === 'pre') { const cur = S.odds.get(m.id); if (!cur || cur.kind !== 'fechamento') { if (!cur || cur.dec.join() !== m.odds.dec.join()) { changed = true; oddsDirty = true; } S.odds.set(m.id, { ...m.odds, kind: 'pre', at: Date.now() }); } }
  }
  // remarcações: remove "adiado" se o mesmo confronto já existe com outro id
  const live = new Set([...S.matches.values()].filter((m) => m.status !== 'off').map((m) => `${m.h}|${m.a}`));
  for (const [id, m] of S.matches) if (m.status === 'off' && live.has(`${m.h}|${m.a}`)) S.matches.delete(id);
  // aplica aos ratings os jogos encerrados que ainda não entraram (em ordem cronológica)
  const pending = [...S.matches.values()].filter((m) => m.status === 'post' && !S.applied.has(m.id) && !m.snapshot).sort((a, b) => (a.date < b.date ? -1 : 1));
  for (const m of pending) {
    const e = R.expectedMatch(S.model, S.state, m.h, m.a, 'A', m.season);
    S.lam.set(m.id, { lh: e.lh, la: e.la });
    let hg = 0, ag = 0; for (const x of m.ev) if ('GPO'.includes(x.t)) (x.s === 'h' ? hg++ : ag++);
    m.evOk = hg === m.hg && ag === m.ag;
    R.applyMatch(S.state, S.model.params, m, S.model.xg);
    S.applied.add(m.id);
    changed = true;
  }
  if (oddsDirty) { oddsDirty = false; const keep = {}; for (const [id, o] of S.odds) if (o.kind !== 'fechamento') keep[id] = o; ls.set('odds', keep); }
  if (changed) { S.version++; cache.clear(); }
  return changed;
}

async function refreshSeason() {
  try {
    merge(await api.scoreboard(String(S.season)));
    S.online = true; S.lastUpdate = Date.now(); S.error = null;
  } catch (e) { S.online = false; S.error = 'Sem conexão com os dados ao vivo — mostrando a última foto do modelo.'; }
  emit();
}

let pollTimer = null, lastSeasonRefresh = Date.now();
export async function poll(force = false) {
  clearTimeout(pollTimer);
  if (document.hidden && !force) return;
  try {
    const json = await api.scoreboard();
    merge(json);
    S.online = true; S.lastUpdate = Date.now(); S.error = null;
    // se a ESPN já virou de temporada, busca o calendário novo
    const y = json?.leagues?.[0]?.season?.year;
    if ((y && y > S.season) || Date.now() - lastSeasonRefresh > 30 * 60000) { lastSeasonRefresh = Date.now(); if (y && y > S.season) { S.season = y; } await refreshSeason(); }
  } catch (e) { S.online = false; }
  emit();
  const live = liveMatches().length > 0;
  const soon = [...S.matches.values()].some((m) => m.status === 'pre' && Math.abs(new Date(m.date) - Date.now()) < 20 * 60000);
  pollTimer = setTimeout(poll, live ? 12000 : soon ? 30000 : 120000);
}

// ── seletores ────────────────────────────────────────────────────────────────
export const team = (id) => S.teams[id] || { id, name: `Time ${id}`, abbr: '???', logo: '' };
export const allMatches = () => [...S.matches.values()].sort((a, b) => (a.date < b.date ? -1 : a.date > b.date ? 1 : 0));
export const liveMatches = () => allMatches().filter((m) => m.status === 'in');
export const upcoming = () => allMatches().filter((m) => m.status === 'pre');
export const finished = () => allMatches().filter((m) => m.status === 'post');
export const teamIds = () => { const s = new Set(); for (const m of S.matches.values()) { s.add(m.h); s.add(m.a); } return [...s]; };
export const reds = (m, side) => m.ev.filter((e) => e.t === 'R' && e.s === side).length;

export function liveState(m) {
  const hz = S.model.hazard, n = compile(hz).n;
  const base = { hg: m.hg || 0, ag: m.ag || 0, rh: reds(m, 'h'), ra: reds(m, 'a') };
  if (m.status === 'post') return { ...base, step: n };
  if (m.status !== 'in') return { step: 0, hg: 0, ag: 0, rh: 0, ra: 0 };
  if (/HALFTIME/.test(m.statusName)) return { ...base, step: 45 + hz.S1 };
  const { min, add } = parseClock(m.clock);
  const half = m.period >= 2 || min > 45 ? 2 : 1;
  return { ...base, step: Math.min(n - 1, stepOf(hz, min, add, half)) };
}

const cache = new Map();
const memo = (key, fn) => { if (!cache.has(key)) cache.set(key, fn()); return cache.get(key); };

/** Gols esperados pré-jogo (e demais canais) de uma partida. */
export function expected(m) {
  return memo(`e${m.id}`, () => {
    const e = R.expectedMatch(S.model, S.state, m.h, m.a, 'A', m.season);
    const lam = S.lam.get(m.id);
    return lam ? { ...e, lh: lam.lh, la: lam.la } : e;
  });
}
export function expectedPair(h, a, neutral = false) {
  return memo(`p${h}|${a}|${neutral}`, () => R.expectedMatch(S.model, S.state, h, a, 'A', S.season, neutral));
}
/** Previsão completa: pré-jogo e, se a bola já rolou, a partir do estado atual. */
export function forecast(m) {
  const st = liveState(m);
  return memo(`f${m.id}|${st.step}|${st.hg}|${st.ag}|${st.rh}|${st.ra}`, () => {
    const e = expected(m), hz = S.model.hazard;
    const pre = memo(`pre${m.id}`, () => { const d = scoreDistribution(hz, e.lh, e.la); return { ...d, sum: summarize(d), next: nextGoal(hz, e.lh, e.la) }; });
    if (m.status === 'pre' || m.status === 'off') return { e, pre, now: pre, st };
    const d = scoreDistribution(hz, e.lh, e.la, st);
    return { e, pre, now: { ...d, sum: summarize(d), next: nextGoal(hz, e.lh, e.la, st) }, st };
  });
}
export function forecastPair(h, a) {
  return memo(`fp${h}|${a}`, () => { const e = expectedPair(h, a), hz = S.model.hazard; const d = scoreDistribution(hz, e.lh, e.la); return { e, pre: { ...d, sum: summarize(d), next: nextGoal(hz, e.lh, e.la) } }; });
}

/** Curva de probabilidade (mandante/empate/visitante) ao longo do jogo. */
export function wpTimeline(m) {
  const cur = liveState(m);
  return memo(`wp${m.id}|${cur.step}|${m.ev.length}|${cur.hg}|${cur.ag}`, () => {
    const hz = S.model.hazard, e = expected(m), out = [];
    const evs = m.ev.filter((x) => 'GPOR'.includes(x.t)).map((x) => ({ ...x, step: stepOf(hz, x.m, x.x) })).sort((a, b) => a.step - b.step);
    let hg = 0, ag = 0, rh = 0, ra = 0, k = 0;
    const end = Math.min(cur.step, compile(hz).n);
    for (let t = 0; t <= end; t++) {
      while (k < evs.length && evs[k].step < t) { const x = evs[k++]; if (x.t === 'R') (x.s === 'h' ? rh++ : ra++); else (x.s === 'h' ? hg++ : ag++); }
      const last = t === end;
      const d = last ? scoreDistribution(hz, e.lh, e.la, cur) : scoreDistribution(hz, e.lh, e.la, { step: t, hg, ag, rh, ra });
      out.push({ t, pH: d.pH, pD: d.pD, pA: d.pA });
    }
    return { points: out, events: evs, total: compile(hz).n };
  });
}

/** Tabela (inclui placares em andamento, marcados como "ao vivo"). */
export function table() {
  return memo('table', () => {
    const ids = teamIds();
    const T = buildTable(ids, allMatches().filter((m) => m.status === 'post' || m.status === 'in'));
    const liveIds = new Set(); for (const m of liveMatches()) { liveIds.add(m.h); liveIds.add(m.a); }
    const rows = sortTable(Object.values(T).map((r) => ({ ...r, name: team(r.id).name, live: liveIds.has(r.id) })));
    rows.forEach((r, i) => { r.rank = i + 1; r.form = form(r.id, 5); });
    return rows;
  });
}
export function form(id, n = 5) {
  return finished().filter((m) => m.h === id || m.a === id).slice(-n).map((m) => { const gf = m.h === id ? m.hg : m.ag, ga = m.h === id ? m.ag : m.hg; return { r: gf > ga ? 'V' : gf === ga ? 'E' : 'D', m }; });
}

/** Elenco/escalação provável e artilheiros prováveis de um time num jogo. */
export function lineup(id) { return memo(`lu${id}`, () => { const sq = S.model.players?.[id]; return sq?.length ? probableLineup(sq) : genericLineup(team(id).name); }); }
export function scorers(id, lambda) { return scorerProbabilities(lineup(id), lambda); }

/** Ranking de forças (ataque/defesa) dos times da Série A. */
export function strengths() {
  return memo('str', () => {
    const rows = teamIds().map((id) => {
      const tmp = { teams: { [id]: structuredClone(S.state.teams[id] || null) }, lg: S.state.lg };
      if (!tmp.teams[id]) return null;
      R.prepareTeam(tmp, S.model.params, id, 'A', S.season);
      return { id, ...R.teamStrength(tmp, id, 'A') };
    }).filter(Boolean).sort((a, b) => b.rating - a.rating);
    rows.forEach((r, i) => { r.rank = i + 1; });
    const byAtt = [...rows].sort((a, b) => b.att - a.att), byDef = [...rows].sort((a, b) => a.def - b.def);
    rows.forEach((r) => { r.rankAtt = byAtt.indexOf(r) + 1; r.rankDef = byDef.indexOf(r) + 1; });
    return rows;
  });
}

// ── simulação da temporada (em Web Worker) ───────────────────────────────────
let worker = null;
export function seasonInputs(fixed = {}) {
  const ids = teamIds();
  const T = buildTable(ids, finished());
  const rest = allMatches().filter((m) => m.status === 'pre' || m.status === 'in' || m.status === 'off');
  const fixtures = rest.map((m) => { const e = expected(m); return { id: m.id, h: m.h, a: m.a, lh: e.lh, la: e.la, live: m.status === 'in' ? liveState(m) : null }; });
  const tm = Object.values(S.state.teams).filter((t) => t.lg === 'A');
  const sigma = Math.sqrt(tm.reduce((s, t) => s + t.ch.g.Po + t.ch.g.Pd, 0) / Math.max(1, 2 * tm.length));
  return { hz: S.model.hazard, ids, table: T, fixtures, sigma, fixed };
}
export function runSeasonSim({ n = 20000, fixed = {}, onDone = null } = {}) {
  const inp = seasonInputs(fixed);
  const key = JSON.stringify([inp.fixtures.map((f) => [f.id, f.lh.toFixed(3), f.live && [f.live.step >> 2, f.live.hg, f.live.ag, f.live.rh, f.live.ra]]), Object.values(inp.table).map((t) => t.pts), fixed, n]);
  if (!onDone && (key === S.simKey || S.simRunning)) return;
  if (!onDone) { S.simKey = key; S.simRunning = true; }
  try {
    const w = onDone ? new Worker(new URL('./season-worker.js', import.meta.url), { type: 'module' }) : (worker ||= new Worker(new URL('./season-worker.js', import.meta.url), { type: 'module' }));
    const done = (res) => { if (onDone) { onDone(res); w.terminate(); } else { S.seasonSim = res; S.simRunning = false; S.version++; emit(); } };
    w.onmessage = (ev) => done(ev.data);
    w.onerror = (err) => { console.error('worker', err); if (!onDone) S.simRunning = false; };
    w.postMessage({ ...inp, n, seed: (Math.random() * 2 ** 31) | 0 });
  } catch (e) { console.error(e); S.simRunning = false; }
}

// ── detalhes do jogo (lances, estatísticas, escalações) ──────────────────────
const kind = (slug = '') => (/own-goal/.test(slug) ? 'own' : /goal|penalty---scored/.test(slug) ? 'goal' : /yellow/.test(slug) ? 'yellow' : /red/.test(slug) ? 'red' : /substitution/.test(slug) ? 'sub' : /kickoff|start-2nd/.test(slug) ? 'kickoff' : /halftime|end-/.test(slug) ? 'whistle' : /penalty/.test(slug) ? 'pen' : /corner/.test(slug) ? 'corner' : /shot|save|post|miss|attempt/.test(slug) ? 'shot' : /var|delay/.test(slug) ? 'var' : /offside/.test(slug) ? 'offside' : /foul|free/.test(slug) ? 'foul' : '');
export function parseSummary(json, m) {
  const H = team(m.h), A = team(m.a);
  const sideOf = (t) => { if (!t) return null; if (t.id) return String(t.id) === m.h ? 'h' : String(t.id) === m.a ? 'a' : null; const n = t.displayName; return n === H.full || n === H.name ? 'h' : n === A.full || n === A.name ? 'a' : null; };
  const commentary = (json.commentary || []).map((c) => ({ label: c.time?.displayValue || '', text: c.text, type: kind(c.play?.type?.type), side: sideOf(c.play?.team) })).reverse();
  const key = (json.keyEvents || []).filter((k) => k.text).map((k) => ({ label: k.clock?.displayValue || '', text: k.text, type: kind(k.type?.type), side: sideOf(k.team), short: k.shortText || k.type?.text })).reverse();
  const stats = {};
  for (const t of json.boxscore?.teams || []) {
    const side = sideOf(t.team); if (!side) continue;
    const g = (n) => { const s = (t.statistics || []).find((x) => x.name === n); const v = s ? parseFloat(s.displayValue) : NaN; return Number.isFinite(v) ? v : null; };
    stats[side] = { po: g('possessionPct'), sh: g('totalShots'), sot: g('shotsOnTarget'), co: g('wonCorners'), fo: g('foulsCommitted'), yc: g('yellowCards'), rc: g('redCards'), off: g('offsides'), sv: g('saves'), pass: g('totalPasses'), passOk: g('accuratePasses') };
  }
  const lineups = {};
  for (const r of json.rosters || []) {
    const side = sideOf(r.team); if (!side || !r.roster) continue;
    const mapP = (p) => ({ id: String(p.athlete?.id), n: p.athlete?.displayName, j: p.jersey || '', pos: p.position?.abbreviation || '', out: !!p.subbedOut, in: !!p.subbedIn });
    lineups[side] = { formation: r.formation || '', xi: r.roster.filter((p) => p.starter).map(mapP), bench: r.roster.filter((p) => !p.starter).map(mapP) };
  }
  return { commentary, key, stats: stats.h && stats.a && stats.h.sh != null ? stats : null, lineups: lineups.h?.xi?.length ? lineups : null, venue: json.gameInfo?.venue?.fullName, att: json.gameInfo?.attendance, at: Date.now() };
}
const inflight = new Map();
export function loadDetails(m, maxAge = 20000) {
  const cur = S.details.get(m.id);
  const ttl = m.status === 'in' ? maxAge : m.status === 'post' ? 3600000 : 300000;
  if (cur && Date.now() - cur.at < ttl) return;
  if (inflight.has(m.id)) return;
  inflight.set(m.id, api.summary(m.id).then((j) => { S.details.set(m.id, parseSummary(j, m)); emit(); }).catch(() => { if (!cur) S.details.set(m.id, { commentary: [], key: [], stats: null, lineups: null, at: Date.now(), failed: true }); }).finally(() => inflight.delete(m.id)));
}

// ── mercado (odds) — só para comparação; nunca entra no modelo ───────────────
/** O que o mercado dizia/diz sobre o jogo: probabilidades implícitas e odds decimais. */
export function market(m) {
  const o = S.odds.get(m.id);
  if (!o) return null;
  return { ...impliedProbs(o.dec), dec: o.dec, src: o.src, kind: o.kind };
}
const oddsTried = new Set();
let oddsQueue = Promise.resolve();
/** Busca as odds de fechamento de jogos já iniciados que ainda não têm (um por vez, sem pressa). */
export function loadOdds(list) {
  for (const m of list) {
    if (S.odds.has(m.id) || oddsTried.has(m.id) || m.status === 'pre' || m.status === 'off') continue;
    oddsTried.add(m.id);
    oddsQueue = oddsQueue.then(() => api.coreOdds(m.id)).then((j) => {
      for (const it of j?.items || []) { const o = parseOdds(it); if (o) { S.odds.set(m.id, { ...o, kind: 'fechamento' }); S.version++; emit(); break; } }
    }).catch(() => {});
  }
}
/** RPS de uma previsão 1X2 para um jogo encerrado (quanto menor, melhor). */
export function rpsOf(p, m) {
  const y = m.hg > m.ag ? [1, 0, 0] : m.hg === m.ag ? [0, 1, 0] : [0, 0, 1];
  return ((p.pH - y[0]) ** 2 + (p.pH + p.pD - y[0] - y[1]) ** 2) / 2;
}
export const outcomeProb = (p, m) => (m.hg > m.ag ? p.pH : m.hg === m.ag ? p.pD : p.pA);
