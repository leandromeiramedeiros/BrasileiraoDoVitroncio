// ─────────────────────────────────────────────────────────────────────────────
// REAJUSTE SEMANAL DO VITRÔNCIO
//   node tools/refit.mjs            reajuste completo (otimiza hiperparâmetros)
//   node tools/refit.mjs --quick    reaproveita os hiperparâmetros do modelo atual
//
// Etapas: (1) proxy de xG  (2) hiperparâmetros dos ratings  (3) passada
// "walk-forward" por toda a história  (4) camada de combinação e contexto do
// jogo  (5) modelo de intensidades minuto a minuto  (6) backtest  (7) jogadores
// (8) exporta site/data/model.json
// ─────────────────────────────────────────────────────────────────────────────
import { readFile, writeFile, readdir } from 'node:fs/promises';
import { loadHistory } from './lib/history.mjs';
import { nelderMead, ols, poissonGLM } from './lib/optim.mjs';
import * as R from '../site/js/engine/ratings.js';
import { scoreDistribution, calibrateLambdas, timeline, stepOf, diffIdx, menIdx } from '../site/js/engine/match-model.js';
import { applyContext } from '../site/js/engine/context.js';
import { bestAgainst, GAPS } from '../site/js/engine/against.js';
import { buildContext, loadCities } from './lib/context.mjs';
import { createLineupState, lineupFeatures, applyLineupMatch, teamSnapshot } from '../site/js/engine/lineup.js';
import { createCardState, applyCardMatch, serveBans, suspendedFor, absentFeatures } from '../site/js/engine/absent.js';
import { gunzipSync } from 'node:zlib';

const QUICK = process.argv.includes('--quick');
const OUT = new URL('../site/data/model.json', import.meta.url);
const t0 = Date.now();
const log = (...a) => console.log(`[${((Date.now() - t0) / 1000).toFixed(1)}s]`, ...a);

const { matches, teams, extra } = await loadHistory();
const done = matches.filter((m) => m.status === 'post');
const seasonNow = Math.max(...matches.filter((m) => m.lg === 'A').map((m) => m.season));
let prev = null;
try { prev = JSON.parse(await readFile(OUT, 'utf8')); } catch { /* primeiro reajuste */ }

const lf = [0]; for (let i = 1; i < 60; i++) lf[i] = lf[i - 1] + Math.log(i);
const pl = (k, l) => k * Math.log(l) - l - lf[Math.min(k, 59)];

// ── (1) proxy de gols esperados a partir de finalizações ─────────────────────
const xg = (() => {
  const X = [], y = [];
  for (const m of done) if (m.statsOk) for (const s of ['h', 'a']) { const st = m.stats[s]; X.push([1, st.sot, Math.max(0, st.sh - st.sot)]); y.push(s === 'h' ? m.hg : m.ag); }
  const b = ols(X, y);
  return { b0: b[0], b1: b[1], b2: b[2], n: X.length };
})();
log('xG proxy', xg);

// ── (2) hiperparâmetros ──────────────────────────────────────────────────────
const params = structuredClone(prev?.params || R.DEFAULT_PARAMS);
for (const c of R.CHANNELS) params[c] = { ...R.DEFAULT_PARAMS[c], ...params[c] };
const LOGK = new Set(['q', 'P0', 'sSeason', 'phi', 'kMu', 'kH', 'Pg0', 'qg']);
const LGT = new Set(['rho', 'w', 'rhoG', 'wRed']);
const enc = (p, keys) => keys.map((k) => (LOGK.has(k) ? Math.log(Math.max(p[k], 1e-7)) : LGT.has(k) ? Math.log(Math.min(0.9999, Math.max(1e-4, p[k])) / (1 - Math.min(0.9999, Math.max(1e-4, p[k])))) : p[k]));
const dec = (x, keys, base) => { const p = { ...base }; keys.forEach((k, i) => { p[k] = LOGK.has(k) ? Math.exp(x[i]) : LGT.has(k) ? 1 / (1 + Math.exp(-x[i])) : x[i]; }); return p; };

function channelLoss(c, pc, fromSeason) {
  const P = { ...params, [c]: pc };
  const st = R.createState(P);
  let llA = 0, nA = 0, llB = 0, nB = 0;
  for (const m of done) {
    R.prepareTeam(st, P, m.h, m.lg, m.season); R.prepareTeam(st, P, m.a, m.lg, m.season);
    if (m.season >= fromSeason) {
      let y;
      if (c === 'g') y = [m.hg, m.ag];
      else y = R.observe(m, P, null)[c];
      if (y) {
        const p = R.predictChannel(st, P, c, m.h, m.a, m.lg);
        const v = pl(Math.round(y[0]), p.lh) + pl(Math.round(y[1]), p.la);
        if (m.lg === 'A') { llA += v; nA++; } else { llB += v; nB++; }
      }
    }
    R.applyMatch(st, P, m, xg, [c]);
  }
  return -(llA / Math.max(1, nA)) - 0.3 * (llB / Math.max(1, nB));
}

if (!QUICK) {
  const gKeys = ['q', 'P0', 'rho', 'sSeason', 'dPromoO', 'dPromoD', 'dNew', 'w', 'phi', 'kMu', 'kH', 'Pg0', 'qg', 'rhoG', 'wRed'];
  let x = enc(params.g, gKeys);
  let best = Infinity;
  for (let r = 0; r < 3; r++) {
    const res = nelderMead((v) => channelLoss('g', dec(v, gKeys, params.g), 2012), x, { maxIter: 450, steps: x.map(() => (r === 0 ? 0.3 : 0.15)) });
    x = res.x; best = res.f;
    log(`ratings de gols: rodada ${r + 1}, perda ${best.toFixed(5)} (${res.evals} avaliações)`);
  }
  params.g = dec(x, gKeys, params.g);
  const oKeys = ['q', 'P0', 'rho', 'sSeason', 'phi'];
  for (const c of ['sh', 'sot', 'co', 'yc', 'fo']) {
    const from = c === 'yc' ? 2012 : 2018;
    const res = nelderMead((v) => channelLoss(c, dec(v, oKeys, params[c]), from), enc(params[c], oKeys), { maxIter: 160, steps: oKeys.map(() => 0.3) });
    params[c] = dec(res.x, oKeys, params[c]);
    log(`canal ${c}: perda ${res.f.toFixed(5)}`);
  }
} else log('modo rápido: hiperparâmetros reaproveitados');

// segundo rating de gols (xG por chute): mesma dinâmica do rating de gols; peso do placar e ruído
// fixados pela pesquisa de nov/2026 (escolhidos só com 2013–2021)
params.x = { ...params.g, w: 0.15, phi: 0.7 };

// ── (3) passada walk-forward ─────────────────────────────────────────────────
const state = R.createState(params);
const rows = [];
const ratingHistory = {};
const applied = [];
for (const m of done) {
  R.prepareTeam(state, params, m.h, m.lg, m.season); R.prepareTeam(state, params, m.a, m.lg, m.season);
  const pr = {}; for (const c of R.CHANNELS) pr[c] = R.predictChannel(state, params, c, m.h, m.a, m.lg);
  const L = {}; for (const c of ['g', 'sh', 'sot']) L[c] = { mu: state.lg[m.lg][c].mu, hfa: state.lg[m.lg][c].hfa };
  rows.push({ m, pr, L });
  R.applyMatch(state, params, m, xg);
  if (m.season === seasonNow) {
    applied.push(m.id);
    if (m.lg === 'A') for (const id of [m.h, m.a]) { const s = state.teams[id].ch.g; (ratingHistory[id] ||= []).push([m.date.slice(0, 10), +s.o.toFixed(3), +s.d.toFixed(3)]); }
  }
}
log(`walk-forward: ${rows.length} partidas`);

// ── (4) camada de combinação (ajustada só com o passado de cada temporada) ───
const feats = (r, side) => {
  const o = side === 'h' ? 'lh' : 'la', sign = side === 'h' ? 1 : -1;
  const rel = (c) => Math.log(r.pr[c][o]) - (r.L[c].mu + (sign * r.L[c].hfa) / 2);
  return [1, Math.log(r.pr.g[o]), rel('sot'), rel('sh'), Math.log(r.pr.x[o] / r.pr.g[o])];
};
function fitStack(sub) {
  const X = [], y = [];
  for (const r of sub) for (const s of ['h', 'a']) { X.push(feats(r, s)); y.push(s === 'h' ? r.m.hg : r.m.ag); }
  if (X.length < 3000) return null;
  const b = poissonGLM(X, y, null, { ridge: 1e-3 });
  return { a0: b[0], a1: b[1], a2: b[2], a3: b[3], a4: b[4] };
}
const stackBySeason = {};
const seasons = [...new Set(rows.map((r) => r.m.season))].sort();
for (const s of seasons) stackBySeason[s] = fitStack(rows.filter((r) => r.m.season >= 2015 && r.m.season < s));
const stack = fitStack(rows.filter((r) => r.m.season >= 2015));
for (const r of rows) {
  const st = stackBySeason[r.m.season];
  const f = (side) => { if (!st) return r.pr.g[side === 'h' ? 'lh' : 'la']; const x = feats(r, side); return Math.exp(st.a0 + st.a1 * x[1] + st.a2 * x[2] + st.a3 * x[3] + st.a4 * x[4]); };
  r.lh = f('h'); r.la = f('a');
}
log('combinação', stack);

// ── (4b) contexto do jogo (ver site/js/engine/context.js) ────────────────────
// Regressão de Poisson sobre os gols, com log λ da etapa anterior como offset.
// Para prever a temporada Y só entram jogos da Série A de 2009 até Y−1.
const cx = buildContext(matches, extra, await loadCities());
const CTX_KEYS = ['tot', 'hfa', 'dec', 'nc', 'closed', 'disp'];
const ctxX = (r, s) => { const f = cx.get(r.m.id); return [1, s, s * (f.dec[0] - f.dec[1]), s * (f.nc[0] - f.nc[1]), s * f.closed, s * f.disp]; };
function fitCtx(sub) {
  if (sub.length < 300) return null;
  const X = [], y = [], off = [];
  for (const r of sub) { X.push(ctxX(r, 1)); y.push(r.m.hg); off.push(Math.log(r.lh)); X.push(ctxX(r, -1)); y.push(r.m.ag); off.push(Math.log(r.la)); }
  // encolhimento: atributos padronizados com desvio a priori de 0,05; constantes quase livres
  const k = CTX_KEYS.length, scale = new Array(k).fill(1);
  for (let j = 2; j < k; j++) { let s1 = 0, s2 = 0; for (const x of X) { s1 += x[j]; s2 += x[j] * x[j]; } const sd = Math.sqrt(Math.max(0, s2 / X.length - (s1 / X.length) ** 2)); scale[j] = sd > 1e-12 ? 0.05 / sd : 0; }
  const b = poissonGLM(X.map((x) => x.map((v, j) => v * scale[j])), y, off, { ridge: 1 });
  return Object.fromEntries(CTX_KEYS.map((key, j) => [key, b[j] * scale[j]]));
}
const ctxTrain = (to) => rows.filter((r) => r.m.lg === 'A' && r.m.season >= 2009 && r.m.season < to);
const ctxBySeason = {};
for (const s of seasons) ctxBySeason[s] = fitCtx(ctxTrain(s));
const ctx = fitCtx(ctxTrain(Infinity));
for (const r of rows) {
  if (r.m.lg !== 'A') continue;
  const adj = applyContext(ctxBySeason[r.m.season], r.lh, r.la, cx.get(r.m.id));
  r.lh0 = r.lh; r.la0 = r.la; r.lh = adj.lh; r.la = adj.la;
}
log('contexto', ctx);

// ── (4c) escalação anunciada (ver site/js/engine/lineup.js) ──────────────────
// Atributos da escalação de cada jogo, calculados só com jogos anteriores; o
// coeficiente da temporada Y sai de uma regressão de Poisson nos jogos da Série A
// de 2009 a Y−1, com os gols esperados até aqui como offset. O backtest com
// escalação supõe a escalação conhecida (no site, cerca de 1 h antes do jogo).
const LU_KEYS = ['usage', 'ga', 'bench'];
const lineupRows = JSON.parse(gunzipSync(await readFile(new URL('../model/data/rosters.json.gz', import.meta.url))).toString('utf8'));
const luState = createLineupState();
const luFeat = new Map();
for (const m of lineupRows) {
  luFeat.set(m.id, [lineupFeatures(luState, m.h, m.home.xi, m.home.bench, m.date), lineupFeatures(luState, m.a, m.away.xi, m.away.bench, m.date)]);
  applyLineupMatch(luState, m);
}
const luDiff = (id) => { const f = luFeat.get(id); return LU_KEYS.map((k) => (f && f[0][k] != null && f[1][k] != null ? f[0][k] - f[1][k] : 0)); };
function fitLineup(sub) {
  if (sub.length < 300) return null;
  const X = [], y = [], off = [];
  for (const r of sub) { const d = luDiff(r.m.id); X.push(d); y.push(r.m.hg); off.push(Math.log(r.lh)); X.push(d.map((v) => -v)); y.push(r.m.ag); off.push(Math.log(r.la)); }
  const b = poissonGLM(X, y, off, { ridge: 1e-6 });
  return Object.fromEntries(LU_KEYS.map((k, j) => [k, b[j]]));
}
const luTrain = (to) => rows.filter((r) => r.m.lg === 'A' && r.m.season >= 2009 && r.m.season < to && luFeat.has(r.m.id));
const luBySeason = {};
for (const s of seasons) luBySeason[s] = fitLineup(luTrain(s));
const lineupCoef = fitLineup(luTrain(Infinity));
for (const r of rows) {
  r.lhNoLu = r.lh; r.laNoLu = r.la;
  const b = luBySeason[r.m.season]; if (r.m.lg !== 'A' || !b || !luFeat.has(r.m.id)) continue;
  const e = luDiff(r.m.id).reduce((a, x, j) => a + b[LU_KEYS[j]] * x, 0);
  r.lh *= Math.exp(e); r.la *= Math.exp(-e);
}
// foto de cada time para o navegador calcular a escalação quando a ESPN publicar
const luSnaps = {};
const nowIso = new Date().toISOString();
for (const id of new Set(matches.filter((m) => m.lg === 'A' && m.season === seasonNow).flatMap((m) => [m.h, m.a]))) { const sn = teamSnapshot(luState, id, nowIso); if (sn.n) luSnaps[id] = sn; }
log('escalação', lineupCoef, `${luFeat.size} jogos com escalação`);

// ── (4d) desfalques conhecidos na véspera (ver site/js/engine/absent.js) ─────
// Suspensos (cartões) e titulares habituais fora do último jogo. Entram só na previsão
// sem escalação (r.lhNoLu); coeficiente da temporada Y com Série A 2009..Y−1.
const AB_KEYS = ['susp_w', 'absP_w'];
const rosterById = new Map(lineupRows.map((x) => [x.id, x]));
const abState = createLineupState(), cards = createCardState(), abFeat = new Map(), abNext = new Map(), ahead = {}, pname = {};
for (const m of matches) {
  if (m.status !== 'post' && m.status !== 'pre') continue;
  for (const e of m.ev) if (e.p && e.n) pname[e.p] = e.n;
  const rr = rosterById.get(m.id);
  const f = [m.h, m.a].map((t) => absentFeatures(teamSnapshot(abState, t, m.date), abState.teams[t], cards, t, m.lg, m.season, m.date));
  if (m.status === 'post') {
    abFeat.set(m.id, f);
    if (rr) applyLineupMatch(abState, rr);
    applyCardMatch(cards, { lg: m.lg, season: m.season, h: m.h, a: m.a, ev: m.ev, min: rr ? { ...rr.home.min, ...rr.away.min } : null });
  } else {
    for (const t of [m.h, m.a]) ahead[t] = (ahead[t] || 0) + 1;
    // só o próximo jogo de cada time: depois dele a foto já não vale
    if (m.lg === 'A' && m.season === seasonNow) abNext.set(m.id, [m.h, m.a].map((t, j) => (ahead[t] === 1 ? { susp_w: f[j].susp_w, absP_w: f[j].absP_w, s: suspendedFor(cards, t, m.lg, m.season).map((p) => pname[p] || '').filter(Boolean) } : null)));
    serveBans(cards, m.h, m.lg, m.season); serveBans(cards, m.a, m.lg, m.season);
  }
}
const abDiff = (id) => { const f = abFeat.get(id); return AB_KEYS.map((k) => (f && f[0][k] != null && f[1][k] != null ? f[0][k] - f[1][k] : 0)); };
function fitAbsent(sub) {
  if (sub.length < 300) return null;
  const X = [], y = [], off = [];
  for (const r of sub) { const d = abDiff(r.m.id); X.push(d); y.push(r.m.hg); off.push(Math.log(r.lhNoLu)); X.push(d.map((v) => -v)); y.push(r.m.ag); off.push(Math.log(r.laNoLu)); }
  const b = poissonGLM(X, y, off, { ridge: 1e-6 });
  return Object.fromEntries(AB_KEYS.map((k, j) => [k, b[j]]));
}
const abTrain = (to) => rows.filter((r) => r.m.lg === 'A' && r.m.season >= 2009 && r.m.season < to && abFeat.has(r.m.id));
const abBySeason = {}; for (const s of seasons) abBySeason[s] = fitAbsent(abTrain(s));
const absentCoef = fitAbsent(abTrain(Infinity));
for (const r of rows) {
  const b = abBySeason[r.m.season]; if (r.m.lg !== 'A' || !b || !abFeat.has(r.m.id)) continue;
  const e = abDiff(r.m.id).reduce((a, x, j) => a + b[AB_KEYS[j]] * x, 0);
  r.lhNoLu *= Math.exp(e); r.laNoLu *= Math.exp(-e);
}
log('desfalques', absentCoef, `próximos jogos com dados: ${abNext.size}`);

// ── (5) intensidades minuto a minuto ─────────────────────────────────────────
const S1 = 3, S2 = 7;
const hzShape = { S1, S2 };
const steps = timeline(hzShape);
function fitHazard() {
  // células: [faixa][saldo][homens] -> {exposição ponderada por λ, gols}, e análogos
  const cell = () => ({ eG: 0, g: 0, eM: 0, r: 0, eY: 0, y: 0 });
  const H = Array.from({ length: 8 }, () => Array.from({ length: 5 }, () => Array.from({ length: 3 }, cell)));
  const A = Array.from({ length: 8 }, () => Array.from({ length: 5 }, () => Array.from({ length: 3 }, cell)));
  let nM = 0;
  for (const r of rows) {
    const m = r.m;
    if (m.season < 2015 || !m.evOk || !m.ev.length) continue;
    nM++;
    const byStep = new Map();
    for (const e of m.ev) { const k = stepOf(hzShape, e.m, e.x); (byStep.get(k) || byStep.set(k, []).get(k)).push(e); }
    let hg = 0, ag = 0, rh = 0, ra = 0;
    for (let t = 0; t < steps.length; t++) {
      const b = steps[t].bin;
      const ch = H[b][diffIdx(hg - ag)][menIdx(rh, ra)], ca = A[b][diffIdx(ag - hg)][menIdx(ra, rh)];
      ch.eG += r.lh / 90; ca.eG += r.la / 90; ch.eM++; ca.eM++; ch.eY += r.pr.yc.lh / 90; ca.eY += r.pr.yc.la / 90;
      for (const e of byStep.get(t) || []) {
        const c = e.s === 'h' ? ch : ca;
        if (e.t === 'G' || e.t === 'P' || e.t === 'O') { c.g++; e.s === 'h' ? hg++ : ag++; }
        else if (e.t === 'R') { c.r++; e.s === 'h' ? rh++ : ra++; }
        else if (e.t === 'Y') c.y++;
      }
    }
  }
  // GLMs
  const mk = (useMen, useLate, useAway) => {
    const X = [], yG = [], oG = [], yR = [], oR = [], yY = [], oY = [];
    for (const [side, T] of [['h', H], ['a', A]]) for (let b = 0; b < 8; b++) for (let d = 0; d < 5; d++) for (let mn = 0; mn < 3; mn++) {
      const c = T[b][d][mn]; if (c.eM < 1) continue;
      const x = new Array(8 + 4 + 2 + 1 + 1).fill(0);
      x[b] = 1; if (d !== 2) x[8 + (d > 2 ? d - 1 : d)] = 1;
      if (useMen && mn !== 1) x[12 + (mn === 2 ? 1 : 0)] = 1;
      if (useLate && b >= 6 && d < 2) x[14] = 1;
      if (useAway && side === 'a') x[15] = 1;
      X.push(x); yG.push(c.g); oG.push(Math.log(c.eG)); yR.push(c.r); oR.push(Math.log(c.eM)); yY.push(c.y); oY.push(Math.log(c.eY));
    }
    return { X, yG, oG, yR, oR, yY, oY };
  };
  const dG = mk(true, true, false);
  const bG = poissonGLM(dG.X, dG.yG, dG.oG, { ridge: 1e-4 });
  const dR = mk(false, false, true);
  const bR = poissonGLM(dR.X, dR.yR, dR.oR, { ridge: 1e-4 });
  const bY = poissonGLM(dR.X, dR.yY, dR.oY, { ridge: 1e-4 });
  const diff = (b) => [b[8], b[9], 0, b[10], b[11]];
  const round = (v) => +v.toFixed(4);
  return {
    S1, S2, nMatches: nM,
    goal: { bin: bG.slice(0, 8).map(round), diff: diff(bG).map(round), men: [round(bG[12]), 0, round(bG[13])], lateTrail: round(bG[14]), norm: 1 },
    red: { bin: bR.slice(0, 8).map(round), diff: diff(bR).map(round), away: round(bR[15]) },
    yel: { bin: bY.slice(0, 8).map(round), diff: diff(bY).map(round), away: round(bY[15]) },
  };
}
const hazard = fitHazard();
// normalização: o total esperado pela cadeia deve bater com λ nos jogos reais
{
  let sumL = 0, sumX = 0, n = 0;
  for (const r of rows) { if (r.m.lg !== 'A' || r.m.season < 2019) continue; if (n++ % 4) continue; const d = scoreDistribution(hazard, r.lh, r.la); sumL += r.lh + r.la; sumX += d.xgH + d.xgA; }
  hazard.goal.norm = +(sumL / sumX).toFixed(4);
}
log('intensidades', JSON.stringify(hazard));

// ── (6) backtest (Série A) ───────────────────────────────────────────────────
let odds = new Map();
const oddsDec = new Map();
const oddsAvg = new Map(); // odds médias de fechamento (para a aba Best Against)
try {
  const map = { 'Flamengo RJ': 'Flamengo', 'Sport Recife': 'Sport', 'Sao Paulo': 'São Paulo', Gremio: 'Grêmio', 'Botafogo RJ': 'Botafogo', Ceara: 'Ceará', Goias: 'Goiás', 'Atletico GO': 'Atlético-GO', 'America MG': 'América-MG', Avai: 'Avaí', Cuiaba: 'Cuiabá', Vitoria: 'Vitória', 'Chapecoense-SC': 'Chapecoense', Nautico: 'Náutico', Parana: 'Paraná', Criciuma: 'Criciúma', 'Atletico-MG': 'Atlético-MG' };
  const nameToId = {}; for (const [id, t] of Object.entries(teams)) { nameToId[t.name] = id; nameToId[t.full] = id; }
  const csv = (await readFile(new URL('../model/cache/BRA.csv', import.meta.url), 'utf8')).replace(/^﻿/, '').split(/\r?\n/).slice(1).filter(Boolean).map((l) => l.split(','));
  for (const r of csv) {
    const hi = nameToId[map[r[5]] || r[5]], ai = nameToId[map[r[6]] || r[6]]; if (!hi || !ai) continue;
    const o = [parseFloat(r[10]) || parseFloat(r[16]), parseFloat(r[11]) || parseFloat(r[17]), parseFloat(r[12]) || parseFloat(r[18])]; if (!(o[0] && o[1] && o[2])) continue;
    const inv = o.map((v) => 1 / v), s = inv[0] + inv[1] + inv[2];
    odds.set(`${r[2]}|${hi}|${ai}`, inv.map((v) => v / s));
    oddsDec.set(`${r[2]}|${hi}|${ai}`, o);
    const av = [16, 17, 18].map((j) => parseFloat(r[j])); if (av.every((v) => v > 1)) oddsAvg.set(`${r[2]}|${hi}|${ai}`, av);
  }
} catch { log('sem arquivo de referência de mercado (model/cache/BRA.csv) — benchmark ignorado'); }

const CALIB = 2; // iterações da calibração da cadeia (ver calibrateLambdas)
const cmpRows = [], gpA = {}; // (Best Against) jogos com odds médias de fechamento
const bt = { from: 2013, seasons: {}, calib: Array.from({ length: 10 }, () => ({ n: 0, p: 0, y: 0 })), totals: { pred: new Array(9).fill(0), obs: new Array(9).fill(0) }, scoreHit: { n: 0, top1: 0, top3: 0 } };
const agg = () => ({ n: 0, ll: 0, rps: 0, brier: 0, hit: 0, nM: 0, llMk: 0, rpsMk: 0, llM2: 0, rpsM2: 0, llBase: 0, rpsBase: 0, rpsNoLu: 0 });
const all = agg();
let baseCnt = [4.8, 2.6, 2.6]; // priori de frequências (mandante/empate/visitante), atualizada ao longo do tempo
for (const r of rows) {
  const m = r.m;
  if (m.lg !== 'A') continue;
  const k = m.hg > m.ag ? 0 : m.hg === m.ag ? 1 : 2;
  if (m.season >= bt.from) {
    const c = calibrateLambdas(hazard, r.lh, r.la, CALIB);
    const d = scoreDistribution(hazard, c.lh, c.la);
    const p = [d.pH, d.pD, d.pA];
    const y = [0, 0, 0]; y[k] = 1;
    const rps = (q) => ((q[0] - y[0]) ** 2 + (q[0] + q[1] - y[0] - y[1]) ** 2) / 2;
    const c0 = r.lhNoLu !== r.lh ? calibrateLambdas(hazard, r.lhNoLu, r.laNoLu, CALIB) : null, d0 = c0 ? scoreDistribution(hazard, c0.lh, c0.la) : d;
    const rpsNoLu = rps([d0.pH, d0.pD, d0.pA]);
    const av = oddsAvg.get(`${m.season}|${m.h}|${m.a}`);
    if (av) { const inv = av.map((v) => 1 / v), s = inv[0] + inv[1] + inv[2]; cmpRows.push({ id: m.id, season: m.season, early: (gpA[`${m.season}|${m.h}`] || 0) < 5 && (gpA[`${m.season}|${m.a}`] || 0) < 5 ? 1 : 0, p: [d0.pH, d0.pD, d0.pA], q: inv.map((v) => v / s), o: av, k }); }
    const bs = baseCnt.map((v) => v / (baseCnt[0] + baseCnt[1] + baseCnt[2]));
    const mk = odds.get(`${m.season}|${m.h}|${m.a}`);
    for (const a of [all, (bt.seasons[m.season] ||= agg())]) {
      a.n++; a.ll -= Math.log(p[k]); a.rps += rps(p); a.rpsNoLu += rpsNoLu; a.brier += p.reduce((s, v, i) => s + (v - y[i]) ** 2, 0);
      a.hit += p[k] === Math.max(...p) ? 1 : 0; a.llBase -= Math.log(bs[k]); a.rpsBase += rps(bs);
      if (mk) { a.nM++; a.llMk -= Math.log(mk[k]); a.rpsMk += rps(mk); a.llM2 -= Math.log(p[k]); a.rpsM2 += rps(p); }
    }
    for (let i = 0; i < 3; i++) { const c = bt.calib[Math.min(9, Math.floor(p[i] * 10))]; c.n++; c.p += p[i]; c.y += y[i]; }
    for (let t = 0; t <= 8; t++) { let pt = 0; for (let i = 0; i <= 10; i++) for (let j = 0; j <= 10; j++) if (Math.min(8, i + j) === t) pt += d.matrix[i][j]; bt.totals.pred[t] += pt; }
    bt.totals.obs[Math.min(8, m.hg + m.ag)]++;
    const flat = []; for (let i = 0; i <= 6; i++) for (let j = 0; j <= 6; j++) flat.push([d.matrix[i][j], i, j]); flat.sort((a, b) => b[0] - a[0]);
    bt.scoreHit.n++; if (flat[0][1] === m.hg && flat[0][2] === m.ag) bt.scoreHit.top1++; if (flat.slice(0, 3).some((f) => f[1] === m.hg && f[2] === m.ag)) bt.scoreHit.top3++;
  }
  baseCnt[k]++;
  for (const id of [m.h, m.a]) gpA[`${m.season}|${id}`] = (gpA[`${m.season}|${id}`] || 0) + 1;
}
const fin = (a) => ({ n: a.n, logloss: +(a.ll / a.n).toFixed(4), rps: +(a.rps / a.n).toFixed(4), rpsSemEscalacao: +(a.rpsNoLu / a.n).toFixed(4), brier: +(a.brier / a.n).toFixed(4), acerto: +(a.hit / a.n).toFixed(4), rpsBase: +(a.rpsBase / a.n).toFixed(4), loglossBase: +(a.llBase / a.n).toFixed(4), ...(a.nM ? { nRef: a.nM, rpsRef: +(a.rpsMk / a.nM).toFixed(4), loglossRef: +(a.llMk / a.nM).toFixed(4), rpsModelNoRef: +(a.rpsM2 / a.nM).toFixed(4) } : {}) });
const backtest = {
  desde: bt.from, geral: fin(all),
  temporadas: Object.fromEntries(Object.entries(bt.seasons).map(([s, a]) => [s, fin(a)])),
  calibracao: bt.calib.map((c) => (c.n ? { n: c.n, previsto: +(c.p / c.n).toFixed(4), observado: +(c.y / c.n).toFixed(4) } : null)),
  totalGols: { previsto: bt.totals.pred.map((v) => +(v / all.n).toFixed(4)), observado: bt.totals.obs.map((v) => +(v / all.n).toFixed(4)) },
  placarExato: { top1: +(bt.scoreHit.top1 / bt.scoreHit.n).toFixed(4), top3: +(bt.scoreHit.top3 / bt.scoreHit.n).toFixed(4) },
};
log('backtest geral', backtest.geral);

// ── (6b) Best Against: o que o histórico diz quando o Vitrôncio discorda do mercado ──
// Células de discordâncias parecidas (site/js/engine/against.js). Para a temporada Y, as
// células usam só temporadas anteriores. Palpite do histórico: o resultado mais acima do
// mercado, quando o ganho esperado com a probabilidade corrigida e a odd média é positivo.
// O Vitrôncio usado é o de dias antes do jogo (sem a escalação).
const agAdd = (cells, r) => { const b = bestAgainst(null, r.p, r.q, r.early); if (!b) return; const h = (cells[b.key] ||= { n: 0, ex: 0, y: 0, p: 0, q: 0, ret: 0 }); const won = r.k === b.i ? 1 : 0; h.n++; h.ex += won - r.q[b.i]; h.y += won; h.p += r.p[b.i]; h.q += r.q[b.i]; h.ret += won * r.o[b.i] - 1; };
const agSeasons = [...new Set(cmpRows.map((r) => r.season))].sort();
const agCellsBy = {}; { const acc = {}; for (const s of agSeasons) { agCellsBy[s] = structuredClone(acc); for (const r of cmpRows) if (r.season === s) agAdd(acc, r); } }
const agCells = {}; for (const r of cmpRows) agAdd(agCells, r);
const agCell = () => ({ n: 0, y: 0, p: 0, q: 0, ret: 0 });
const agGap = GAPS.map(agCell), agSeason = {}, agRecord = [];
for (const r of cmpRows) {
  const S = (agSeason[r.season] ||= { n: 0, bets: 0, wins: 0, ret: 0, rawBets: 0, rawWins: 0, rawRet: 0 }); S.n++;
  const raw = bestAgainst(null, r.p, r.q, r.early);
  if (raw) {
    const won = r.k === raw.i ? 1 : 0, b = agGap[+raw.key[0]];
    b.n++; b.y += won; b.p += r.p[raw.i]; b.q += r.q[raw.i]; b.ret += won * r.o[raw.i] - 1;
    if (r.p[raw.i] * r.o[raw.i] > 1) { S.rawBets++; S.rawWins += won; S.rawRet += won * r.o[raw.i] - 1; }
  }
  const res = bestAgainst(agCellsBy[r.season], r.p, r.q, r.early);
  if (!res || res.c[res.i] * r.o[res.i] <= 1) continue;
  const won = r.k === res.i ? 1 : 0;
  S.bets++; S.wins += won; S.ret += won * r.o[res.i] - 1;
  if (r.season === seasonNow) agRecord.push({ id: r.id, k: res.i, p: +r.p[res.i].toFixed(4), q: +r.q[res.i].toFixed(4), c: +res.c[res.i].toFixed(4), o: r.o[res.i], won });
}
// múltiplas: em cada rodada (semana), a combinação de k pernas com maior ganho esperado
// corrigido, que são os k palpites com maior (probabilidade corrigida × odd). Mesmas
// células por temporada (só o passado); sem a primeira temporada, que não tem histórico.
const agDate = new Map(matches.map((m) => [m.id, m.date]));
const agWeeks = new Map();
for (const r of cmpRows) {
  if (r.season === agSeasons[0]) continue;
  const res = bestAgainst(agCellsBy[r.season], r.p, r.q, r.early); if (!res) continue;
  const w = `${r.season}|${Math.floor((Date.parse(agDate.get(r.id)) - Date.parse('2000-01-03T00:00:00Z')) / 6048e5)}`;
  (agWeeks.get(w) || agWeeks.set(w, []).get(w)).push({ e: res.c[res.i] * r.o[res.i], o: r.o[res.i], won: r.k === res.i, season: r.season });
}
const agMult = {};
for (const legs of agWeeks.values()) {
  legs.sort((x, y) => y.e - x.e);
  for (let k = 1; k <= 5 && k <= legs.length; k++) {
    const top = legs.slice(0, k), win = top.every((l) => l.won), pay = win ? top.reduce((a, l) => a * l.o, 1) : 0;
    const A = (agMult[k] ||= { n: 0, hits: 0, ret: 0, odd: [], temporadas: {} }), S = (A.temporadas[top[0].season] ||= { n: 0, hits: 0, ret: 0 });
    for (const x of [A, S]) { x.n++; x.hits += win ? 1 : 0; x.ret += pay - 1; }
    A.odd.push(top.reduce((a, l) => a * l.o, 1));
  }
}
for (const A of Object.values(agMult)) { A.odd.sort((x, y) => x - y); A.oddMediana = +A.odd[A.odd.length >> 1].toFixed(1); delete A.odd; A.ret = +A.ret.toFixed(2); for (const s of Object.values(A.temporadas)) s.ret = +s.ret.toFixed(2); }
const agMargin = cmpRows.reduce((a, r) => a + (1 / r.o[0] + 1 / r.o[1] + 1 / r.o[2] - 1), 0) / Math.max(1, cmpRows.length);
const agFmt = (b) => ({ n: b.n, aconteceu: b.n ? +(b.y / b.n).toFixed(4) : null, modelo: b.n ? +(b.p / b.n).toFixed(4) : null, mercado: b.n ? +(b.q / b.n).toFixed(4) : null, retorno: b.n ? +(b.ret / b.n).toFixed(4) : null });
const agTot = Object.values(agSeason).reduce((t, x) => { for (const k in x) t[k] = (t[k] || 0) + x[k]; return t; }, {});
const r2 = (v) => +v.toFixed(2);
const against = cmpRows.length ? {
  cells: Object.fromEntries(Object.entries(agCells).map(([k, h]) => [k, { n: h.n, ex: +h.ex.toFixed(3), y: h.y, p: +h.p.toFixed(3), q: +h.q.toFixed(3), ret: +h.ret.toFixed(3) }])),
  porDiferenca: agGap.map(agFmt),
  temporadas: Object.fromEntries(Object.entries(agSeason).map(([s, v]) => [s, { ...v, ret: r2(v.ret), rawRet: r2(v.rawRet) }])),
  total: { ...agTot, ret: r2(agTot.ret), rawRet: r2(agTot.rawRet) },
  record: agRecord, jogos: cmpRows.length, fonte: 'odds médias de fechamento (football-data.co.uk)',
  multiplas: agMult, margem: +agMargin.toFixed(4),
} : null;
log('best against', against && { total: against.total, porDiferenca: against.porDiferenca });

// dispersão por canal (para distribuições de escanteios, cartões, finalizações)
const dispersion = {};
for (const c of ['sh', 'sot', 'co', 'yc', 'fo']) {
  let s = 0, n = 0;
  for (const r of rows) { if (r.m.lg !== 'A' || r.m.season < 2019) continue; const o = R.observe(r.m, params, null)[c]; if (!o) continue; s += (o[0] - r.pr[c].lh) ** 2 / r.pr[c].lh + (o[1] - r.pr[c].la) ** 2 / r.pr[c].la; n += 2; }
  dispersion[c] = +(s / Math.max(1, n)).toFixed(3);
}

// ── (7) jogadores (temporada atual, a partir dos summaries) ──────────────────
const players = {};
const posGroup = (ab = '') => { const a = ab.toUpperCase(); if (a === 'G' || a === 'GK') return 'G'; if (/^(CD|LB|RB|D|SW|LWB|RWB|CB)/.test(a) || a.includes('B')) return 'D'; if (a.includes('M')) return 'M'; if (/F|S|W/.test(a) && a !== 'SUB') return 'F'; return null; };
try {
  const dir = new URL('../model/cache/sum/', import.meta.url);
  const files = await readdir(dir);
  const order = new Map(done.filter((m) => m.lg === 'A' && m.season === seasonNow).map((m) => [m.id, m.date]));
  const list = files.map((f) => f.replace('.json', '')).filter((id) => order.has(id)).sort((a, b) => (order.get(a) < order.get(b) ? -1 : 1));
  const teamGames = {};
  for (const id of list) {
    const d = JSON.parse(await readFile(new URL(`${id}.json`, dir), 'utf8'));
    for (const r of d.rosters || []) {
      const tid = String(r.team?.id); if (!r.roster) continue;
      const gi = (teamGames[tid] = (teamGames[tid] || 0) + 1);
      const T = (players[tid] ||= {});
      for (const pe of r.roster) {
        const a = pe.athlete; if (!a) continue;
        const P = (T[a.id] ||= { id: String(a.id), n: a.displayName, sn: a.shortName || a.displayName, j: pe.jersey, pos: {}, apps: 0, st: 0, min: 0, g: 0, pk: 0, a: 0, sh: 0, sot: 0, y: 0, r: 0, last: 0, recentSt: [] });
        const stat = (n) => pe.stats?.find((s) => s.name === n)?.value || 0;
        const subPlay = (pe.plays || []).find((x) => x.substitution);
        const subMin = subPlay ? parseInt(subPlay.clock?.displayValue) || 60 : null;
        let min = 0;
        if (pe.starter) min = pe.subbedOut ? Math.min(90, subMin ?? 70) : 90;
        else if (pe.subbedIn) min = Math.max(1, 90 - Math.min(89, subMin ?? 70));
        if (min > 0) { P.apps++; P.last = gi; if (pe.jersey) P.j = pe.jersey; }
        if (pe.starter) { P.st++; const g = posGroup(pe.position?.abbreviation); if (g) P.pos[g] = (P.pos[g] || 0) + 1; }
        P.recentSt.push([gi, pe.starter ? 1 : 0, min]);
        P.min += min; P.g += stat('totalGoals'); P.a += stat('goalAssists'); P.sh += stat('totalShots'); P.sot += stat('shotsOnTarget'); P.y += stat('yellowCards'); P.r += stat('redCards');
      }
    }
  }
  // pênaltis convertidos (dos eventos do scoreboard)
  for (const m of done) if (m.lg === 'A' && m.season === seasonNow) for (const e of m.ev) if (e.t === 'P' && e.p) { const tid = e.s === 'h' ? m.h : m.a; if (players[tid]?.[e.p]) players[tid][e.p].pk++; }
  for (const tid of Object.keys(players)) {
    const G = teamGames[tid];
    players[tid] = Object.values(players[tid]).filter((p) => p.min > 0).map((p) => {
      const pos = Object.entries(p.pos).sort((a, b) => b[1] - a[1])[0]?.[0] || 'M';
      const rec = p.recentSt.filter((x) => x[0] > G - 6);
      return { id: p.id, n: p.n, sn: p.sn, j: p.j || '', pos, apps: p.apps, st: p.st, min: p.min, g: p.g, pk: p.pk, a: p.a, sh: p.sh, sot: p.sot, y: p.y, r: p.r, rs: rec.reduce((s, x) => s + x[1], 0), rm: rec.reduce((s, x) => s + x[2], 0), last: G - p.last };
    }).sort((a, b) => b.min - a.min);
  }
  log(`jogadores: ${Object.values(players).reduce((s, v) => s + v.length, 0)} em ${Object.keys(players).length} times`);
} catch (e) { log('jogadores indisponíveis:', e.message); }

// ── (8) exportação ───────────────────────────────────────────────────────────
const keep = Object.fromEntries(Object.entries(state.teams).filter(([, t]) => t.season >= seasonNow - 1).map(([id, t]) => [id, { lg: t.lg, season: t.season, n: t.n, ch: Object.fromEntries(Object.entries(t.ch).map(([c, s]) => [c, Object.fromEntries(Object.entries(s).map(([k, v]) => [k, +v.toFixed(5)]))])) }]));
const lgOut = {}; for (const L of ['A', 'B']) { lgOut[L] = {}; for (const c of R.CHANNELS) lgOut[L][c] = { mu: +state.lg[L][c].mu.toFixed(5), hfa: +state.lg[L][c].hfa.toFixed(5), n: state.lg[L][c].n }; }
const teamMeta = Object.fromEntries(Object.keys(keep).map((id) => [id, teams[id]]));
const preLam = new Map(rows.filter((r) => r.m.lg === 'A' && r.m.season === seasonNow).map((r) => { const c = calibrateLambdas(hazard, r.lh, r.la, CALIB); return [r.m.id, [+c.lh.toFixed(4), +c.la.toFixed(4)]]; }));
// calendário da temporada; jogos já aplicados levam o λ pré-jogo (para gráficos de jogos passados);
// jogos por vir levam o contexto conhecido de antemão: [continental em seguida (m, v), mando fora da sede]
const cxOut = (id) => { const f = cx.get(id); return f && (f.nc[0] || f.nc[1] || f.disp) ? [f.nc[0], f.nc[1], f.disp] : null; };
const fixtures = matches.filter((m) => m.lg === 'A' && m.season === seasonNow).map((m) => ({ id: m.id, date: m.date, status: m.status, h: m.h, a: m.a, hg: m.hg, ag: m.ag, venue: m.venue, city: m.city, ...(oddsDec.has(`${m.season}|${m.h}|${m.a}`) && m.status === 'post' ? { mkt: oddsDec.get(`${m.season}|${m.h}|${m.a}`) } : {}), ...(m.status !== 'post' && cxOut(m.id) ? { cx: cxOut(m.id) } : {}), ...(m.status !== 'post' && abNext.has(m.id) ? { ab: abNext.get(m.id) } : {}), ...(preLam.has(m.id) ? { lam: preLam.get(m.id), ev: m.ev.filter((e) => e.t !== 'Y').map((e) => [e.m, e.x, e.t, e.s, e.n]) } : {}) }));
const lastA = done.filter((m) => m.lg === 'A').at(-1);
const round5 = (o) => Object.fromEntries(Object.entries(o).map(([k, v]) => [k, typeof v === 'number' ? +v.toPrecision(5) : v]));
const model = {
  version: 3,
  nome: 'Vitrôncio v3.3',
  geradoEm: new Date().toISOString(),
  dadosAte: lastA?.date,
  temporada: seasonNow,
  base: { partidas: done.length, serieA: done.filter((m) => m.lg === 'A').length, serieB: done.filter((m) => m.lg === 'B').length, desde: 2006, eventos: done.reduce((s, m) => s + m.ev.length, 0) },
  params: Object.fromEntries(Object.entries(params).map(([c, p]) => [c, round5(p)])),
  xg: round5(xg),
  stack: stack && round5(stack),
  ctx: ctx && { ...round5(ctx), ...(lineupCoef ? { lineup: round5(lineupCoef) } : {}), ...(absentCoef ? { absent: round5(absentCoef) } : {}) },
  lineupSnaps: luSnaps,
  calib: CALIB,
  hazard,
  dispersion,
  state: { lg: lgOut, teams: keep, applied: state.applied, lastDate: state.lastDate },
  applied,
  teams: teamMeta,
  fixtures,
  ratingHistory,
  players,
  backtest,
  against,
};
await writeFile(OUT, JSON.stringify(model));
log(`model.json gravado (${((await readFile(OUT)).length / 1024).toFixed(0)} KB)`);
