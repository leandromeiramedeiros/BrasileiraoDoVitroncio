import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import * as R from '../site/js/engine/ratings.js';
import { scoreDistribution, nextGoal, summarize, timeline, stepOf, liveState, lambdasFromProbs } from '../site/js/engine/match-model.js';
import { simulateSeason, buildTable } from '../site/js/engine/season.js';
import { simulateMatch } from '../site/js/engine/sim-match.js';
import { probableLineup, scorerProbabilities, genericLineup } from '../site/js/engine/players.js';
import { makeRng } from '../site/js/engine/rng.js';
import { parseClock } from '../site/js/data/espn-parse.js';

const model = JSON.parse(readFileSync(new URL('../site/data/model.json', import.meta.url), 'utf8'));
const hz = model.hazard;
const near = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg || ''} esperado ~${b}, veio ${a}`);

test('matriz de placar soma 1 e reproduz os gols esperados', () => {
  for (const [lh, la] of [[1.5, 1.0], [2.4, 0.6], [0.8, 1.7], [1.2, 1.2]]) {
    const d = scoreDistribution(hz, lh, la);
    const s = d.matrix.flat().reduce((a, b) => a + b, 0);
    near(s, 1, 1e-9, 'soma');
    near(d.pH + d.pD + d.pA, 1, 1e-9, '1X2');
    near(d.xgH, lh, lh * 0.06, 'gols mandante'); near(d.xgA, la, la * 0.06, 'gols visitante');
  }
});

test('time mais forte é favorito e mando de campo pesa', () => {
  const a = scoreDistribution(hz, 2.0, 0.8), b = scoreDistribution(hz, 0.8, 2.0);
  assert.ok(a.pH > 0.6 && b.pA > 0.6);
  near(a.pH, b.pA, 0.02, 'simetria');
});

test('ao vivo: placar e tempo restante mudam as chances de forma coerente', () => {
  const pre = scoreDistribution(hz, 1.4, 1.1);
  const lead60 = scoreDistribution(hz, 1.4, 1.1, liveState(hz, { minute: 60, hg: 1, ag: 0 }));
  const lead88 = scoreDistribution(hz, 1.4, 1.1, liveState(hz, { minute: 88, hg: 1, ag: 0 }));
  const red = scoreDistribution(hz, 1.4, 1.1, liveState(hz, { minute: 60, hg: 1, ag: 0, rh: 1 }));
  assert.ok(lead60.pH > pre.pH && lead88.pH > lead60.pH && lead88.pH > 0.85);
  assert.ok(red.pH < lead60.pH, 'vermelho derruba a chance de quem lidera');
  const end = scoreDistribution(hz, 1.4, 1.1, liveState(hz, { minute: 90, hg: 2, ag: 2, finished: true }));
  near(end.pD, 1, 1e-12, 'jogo encerrado'); near(end.matrix[2][2], 1, 1e-12);
  // placar já construído nunca diminui
  let below = 0; for (let i = 0; i <= 10; i++) for (let j = 0; j <= 10; j++) if (i < 1) below += lead60.matrix[i][j];
  near(below, 0, 1e-12, 'massa abaixo do placar atual');
});

test('próximo gol e resumos são probabilidades válidas', () => {
  const ng = nextGoal(hz, 1.6, 1.0);
  near(ng.pH + ng.pA + ng.pNone, 1, 1e-9);
  const d = scoreDistribution(hz, 1.6, 1.0), s = summarize(d);
  near(ng.pNone, d.matrix[0][0], 1e-6, '0x0 = nenhum gol');
  assert.ok(s.over05 > s.over15 && s.over15 > s.over25 && s.over25 > s.over35);
  near(s.totals.reduce((a, b) => a + b, 0), 1, 1e-9);
});

test('linha do tempo e relógio', () => {
  const tl = timeline(hz);
  assert.equal(tl.length, 90 + hz.S1 + hz.S2);
  assert.equal(stepOf(hz, 1), 0); assert.equal(stepOf(hz, 45), 44); assert.equal(stepOf(hz, 45, 2), 46);
  assert.equal(stepOf(hz, 46), hz.S1 + 45); assert.equal(stepOf(hz, 90, 4), 90 + hz.S1 + 3);
  assert.deepEqual(parseClock("45'+3'"), { min: 45, add: 3 }); assert.deepEqual(parseClock("67'"), { min: 67, add: 0 });
});

test('ratings: previsão simétrica e atualização na direção certa', () => {
  const st = structuredClone(model.state);
  const ids = Object.keys(model.state.teams).filter((id) => model.state.teams[id].lg === 'A' && model.state.teams[id].season === model.temporada);
  assert.equal(ids.length, 20);
  const [h, a] = ids;
  const before = R.expectedMatch(model, st, h, a, 'A', model.temporada);
  assert.ok(before.lh > 0.2 && before.lh < 4 && before.la > 0.2 && before.la < 4);
  R.applyMatch(st, model.params, { id: 'x', lg: 'A', season: model.temporada, date: '2099-01-01', h, a, hg: 5, ag: 0, evOk: false, ev: [], stats: null, statsOk: false }, model.xg);
  const after = R.expectedMatch(model, st, h, a, 'A', model.temporada);
  assert.ok(after.lh > before.lh && after.la < before.la, 'goleada melhora o mandante');
});

test('simulação de temporada: probabilidades fecham', () => {
  const ids = Object.keys(model.state.teams).filter((id) => model.state.teams[id].lg === 'A' && model.state.teams[id].season === model.temporada);
  const fin = model.fixtures.filter((m) => m.status === 'post'), rest = model.fixtures.filter((m) => m.status !== 'post');
  const table = buildTable(ids, fin);
  const fx = rest.map((m) => { const e = R.expectedMatch(model, model.state, m.h, m.a, 'A', model.temporada); return { id: m.id, h: m.h, a: m.a, lh: e.lh, la: e.la }; });
  const res = simulateSeason(hz, ids, table, fx, { n: 3000, seed: 7 });
  const sum = (k) => Object.values(res.teams).reduce((s, t) => s + t[k], 0);
  near(sum('titulo'), 1, 1e-9); near(sum('rebaixamento'), 4, 1e-9); near(sum('libertadores'), 4, 1e-9);
  for (const t of Object.values(res.teams)) { near(t.pos.reduce((a, b) => a + b, 0), 1, 1e-9); assert.ok(t.pts >= table[t.id].pts); }
});

test('jogo simulado: placar bate com os eventos e a média bate com o modelo', () => {
  const rng = makeRng(42);
  const cfg = { home: { name: 'Casa', lineup: genericLineup('Casa') }, away: { name: 'Fora', lineup: genericLineup('Fora') }, lh: 1.5, la: 1.0 };
  let gh = 0, ga = 0, wins = 0; const N = 4000;
  for (let i = 0; i < N; i++) {
    const r = simulateMatch(model, cfg, rng);
    const eh = r.events.filter((e) => e.type === 'goal' && e.side === 'h').length, ea = r.events.filter((e) => e.type === 'goal' && e.side === 'a').length;
    assert.equal(eh, r.score[0]); assert.equal(ea, r.score[1]);
    gh += r.score[0]; ga += r.score[1]; if (r.score[0] > r.score[1]) wins++;
  }
  const d = scoreDistribution(hz, 1.5, 1.0);
  near(gh / N, d.xgH, 0.08, 'média gols mandante'); near(ga / N, d.xgA, 0.07, 'média gols visitante'); near(wins / N, d.pH, 0.03, 'vitórias mandante');
});

test('jogadores: escalação provável e artilheiros', () => {
  const tid = Object.keys(model.players)[0];
  const lu = probableLineup(model.players[tid]);
  assert.equal(lu.xi.length, 11); assert.equal(lu.xi.filter((p) => p.pos === 'G').length, 1);
  const sp = scorerProbabilities(lu, 1.5);
  assert.ok(sp[0].p > 0.1 && sp[0].p < 0.7);
  const exp = sp.reduce((s, x) => s + x.xg, 0);
  assert.ok(exp > 1.2 && exp <= 1.5);
});

test('caminho inverso: probabilidades-alvo viram gols esperados coerentes', () => {
  for (const [lh, la] of [[1.5, 1.0], [2.3, 0.7], [0.9, 1.6], [1.1, 1.1]]) {
    const d = scoreDistribution(hz, lh, la);
    const r = lambdasFromProbs(hz, d.pH, d.pA);
    near(r.lh, lh, 0.01, 'lh'); near(r.la, la, 0.01, 'la');
    const t = lambdasFromProbs(hz, d.pH, d.pA, { total: lh + la });
    near(t.lh, lh, 0.01, 'lh com total fixo'); near(t.la, la, 0.01, 'la com total fixo');
  }
  // alvo "estranho" (empate alto demais) ainda devolve algo utilizável
  const w = lambdasFromProbs(hz, 0.30, 0.25);
  assert.ok(w.lh > 0.05 && w.la > 0.05 && Number.isFinite(w.lh + w.la));
});
