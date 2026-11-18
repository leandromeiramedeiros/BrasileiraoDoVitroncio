import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import * as R from '../site/js/engine/ratings.js';
import { scoreDistribution, nextGoal, summarize, timeline, stepOf, liveState, lambdasFromProbs, calibrateLambdas } from '../site/js/engine/match-model.js';
import { simulateSeason, buildTable } from '../site/js/engine/season.js';
import { simulateMatch } from '../site/js/engine/sim-match.js';
import { probableLineup, scorerProbabilities, genericLineup } from '../site/js/engine/players.js';
import { makeRng } from '../site/js/engine/rng.js';
import { parseClock } from '../site/js/data/espn-parse.js';
import { applyContext, decided } from '../site/js/engine/context.js';

const model = JSON.parse(readFileSync(new URL('../site/data/model.json', import.meta.url), 'utf8'));
const hz = model.hazard;
const near = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg || ''} esperado ~${b}, veio ${a}`);

test('matriz de placar soma 1 e reproduz os gols esperados', () => {
  for (const [lh, la] of [[1.5, 1.0], [2.4, 0.6], [0.8, 1.7], [1.2, 1.2]]) {
    const d = scoreDistribution(hz, lh, la);
    const s = d.matrix.flat().reduce((a, b) => a + b, 0);
    near(s, 1, 1e-9, 'soma');
    near(d.pH + d.pD + d.pA, 1, 1e-9, '1X2');
    // a cadeia dá um pouco mais de gols a quem está atrás no placar: em jogo muito desigual o azarão
    // fica ~6% acima do λ e o favorito ~5% abaixo (medido: 6,0% no v2, 6,3% no v3). Tolerância de 8%.
    near(d.xgH, lh, lh * 0.08, 'gols mandante'); near(d.xgA, la, la * 0.08, 'gols visitante');
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

test('contexto: situação definida na tabela e ajuste dos gols esperados', () => {
  // 20 times, 38 jogos: lanterna com 10 pts e 2 jogos por fazer não alcança o 16º (20 pts)
  const rows = Array.from({ length: 20 }, (_, i) => ({ id: String(i), pts: 80 - 3 * i, played: 36 }));
  rows[19].pts = 10; rows[0].pts = 90;
  const d = decided(rows, 38);
  assert.equal(d.get('19'), 'rebaixado');
  assert.equal(d.get('0'), 'campeao');
  assert.equal(d.has('10'), false);
  // no meio do campeonato ninguém está definido
  assert.equal(decided(rows.map((r) => ({ ...r, played: 20 })), 38).size, 0);
  const ctx = { tot: 0, hfa: 0, dec: -0.2, nc: -0.08, closed: -0.07, disp: -0.12 };
  const base = applyContext(ctx, 1.4, 1.0, null);
  assert.ok(Math.abs(base.lh - 1.4) < 1e-12 && Math.abs(base.la - 1.0) < 1e-12);
  const rel = applyContext(ctx, 1.4, 1.0, { dec: [1, 0] });
  assert.ok(rel.lh < 1.4 && rel.la > 1.0, 'mandante já rebaixado rende menos');
  const rot = applyContext(ctx, 1.4, 1.0, { nc: [0, 1] });
  assert.ok(rot.lh > 1.4 && rot.la < 1.0, 'visitante que vai poupar rende menos');
  assert.deepEqual(applyContext(null, 1.4, 1.0, { dec: [1, 0] }), { lh: 1.4, la: 1.0 });
  if (model.ctx) for (const k of ['dec', 'nc', 'disp']) assert.ok(model.ctx[k] < 0 && model.ctx[k] > -0.6, `coeficiente ${k} plausível`);
});

test('calibração: a cadeia reproduz os gols esperados de cada lado', () => {
  for (const [lh, la] of [[1.5, 1.0], [2.4, 0.6], [0.7, 2.6], [1.2, 1.2]]) {
    const c = calibrateLambdas(hz, lh, la);
    const d = scoreDistribution(hz, c.lh, c.la);
    near(d.xgH, lh, lh * 0.002, 'gols mandante'); near(d.xgA, la, la * 0.002, 'gols visitante');
  }
  const [f] = model.fixtures.filter((x) => x.status === 'pre');
  if (f && model.calib) { const e = R.expectedMatch(model, model.state, f.h, f.a, 'A', model.temporada); const d = scoreDistribution(hz, e.lh, e.la); near(d.xgH, e.xh, e.xh * 0.002, 'jogo real'); }
});

test('escalação: foto do time e efeito nos gols esperados', async () => {
  const { createLineupState, applyLineupMatch, teamSnapshot, snapshotFeatures, lineupFeatures } = await import('../site/js/engine/lineup.js');
  const st = createLineupState();
  const xi = Array.from({ length: 11 }, (_, i) => `p${i}`), alt = [...xi.slice(0, 6), 'q1', 'q2', 'q3', 'q4', 'q5'];
  const min = Object.fromEntries(xi.map((p) => [p, 90]));
  for (let k = 0; k < 6; k++) applyLineupMatch(st, { date: `2026-05-${String(10 + k).padStart(2, '0')}T20:00Z`, season: 2026, h: 'T', a: 'U', home: { xi, bench: ['q1'], min, ga: { p9: 1 } }, away: { xi, bench: [], min, ga: {} } });
  const snap = JSON.parse(JSON.stringify(teamSnapshot(st, 'T', '2026-05-20T20:00Z')));
  const full = snapshotFeatures(snap, xi, ['q1']), half = snapshotFeatures(snap, alt, []);
  assert.ok(Math.abs(full.usage - 1) < 1e-3 && half.usage < 0.6, 'time habitual completo = 1');
  assert.deepEqual(full, lineupFeatures(st, 'T', xi, ['q1'], '2026-05-20T20:00Z'), 'foto no navegador = cálculo direto');
  if (model.ctx?.lineup) {
    const a = applyContext(model.ctx, 1.4, 1.0, { lu: [half, full] }), b = applyContext(model.ctx, 1.4, 1.0, { lu: [full, full] });
    assert.ok(a.lh < b.lh && a.la > b.la, 'mandante desfalcado rende menos');
    const c = applyContext(model.ctx, 1.4, 1.0, { lu: [{ usage: null }, full] }), z = applyContext(model.ctx, 1.4, 1.0, null);
    assert.ok(Math.abs(c.lh - z.lh) < 1e-12, 'atributo nulo de um lado não entra');
  }
});

test('Best Against: probabilidade corrigida pelo histórico', async () => {
  const { bestAgainst, cellKey } = await import('../site/js/engine/against.js');
  const p = [0.49, 0.25, 0.26], q = [0.80, 0.12, 0.08];
  assert.equal(bestAgainst({}, [0.5, 0.3, 0.2], [0.48, 0.3, 0.22]), null, 'diferença menor que 5 pontos');
  const none = bestAgainst({}, p, q);
  assert.equal(none.i, 2); assert.ok(Math.abs(none.c[2] - q[2]) < 1e-12, 'sem histórico, fica com o mercado');
  const key = cellKey(none.gap, q[2], false);
  const good = bestAgainst({ [key]: { n: 100, ex: 10 } }, p, q);
  assert.ok(Math.abs(good.c[2] - (q[2] + 10 / 200)) < 1e-12, 'histórico favorável sobe, com desconto');
  assert.ok(Math.abs(good.c.reduce((a, b) => a + b, 0) - 1) < 1e-12, 'soma 1');
  if (model.against) assert.ok(model.against.total.bets > 50 && Object.keys(model.against.cells).length > 4);
});

test('desfalques: suspensão por cartões', async () => {
  const { createCardState, applyCardMatch, suspendedFor, serveBans } = await import('../site/js/engine/absent.js');
  const c = createCardState(), base = { lg: 'A', season: 2026, h: 'T', a: 'U' };
  for (let k = 0; k < 2; k++) applyCardMatch(c, { ...base, ev: [{ t: 'Y', s: 'h', p: 'p1', m: 30 }] });
  assert.equal(suspendedFor(c, 'T', 'A', 2026).length, 0, 'dois amarelos: ainda joga');
  applyCardMatch(c, { ...base, ev: [{ t: 'Y', s: 'h', p: 'p1', m: 30 }, { t: 'R', s: 'a', p: 'p2', m: 80 }] });
  assert.deepEqual(suspendedFor(c, 'T', 'A', 2026), ['p1'], 'terceiro amarelo suspende');
  assert.deepEqual(suspendedFor(c, 'U', 'A', 2026), ['p2'], 'vermelho direto suspende');
  serveBans(c, 'T', 'A', 2026);
  assert.equal(suspendedFor(c, 'T', 'A', 2026).length, 0, 'cumpriu no jogo seguinte');
  if (model.ctx?.absent) assert.ok(model.ctx.absent.susp_w < 0 && model.ctx.absent.absP_w < 0, 'desfalque reduz os gols esperados');
});

test('mercados: pernas do mesmo jogo e contagens', async () => {
  const { jointProb, selProb } = await import('../site/js/engine/markets.js');
  const d = scoreDistribution(hz, 1.5, 1.0);
  const ctx = { matrix: d.matrix, pRed: 0.22, yc: 4.6, co: 9.8, disp: model.dispersion, cal: model.against?.calMercados };
  near(selProb(ctx, 'r:H') + selProb(ctx, 'r:D') + selProb(ctx, 'r:A'), 1, 1e-9, '1X2 soma 1');
  near(selProb(ctx, 'g:o2.5') + selProb(ctx, 'g:u2.5'), 1, 1e-9, 'mais + menos = 1');
  near(jointProb(ctx, ['r:H', 'g:u1.5']), d.matrix[1][0], 1e-12, 'vitória com menos de 1,5 gol = só 1 a 0');
  assert.ok(jointProb(ctx, ['r:H', 'bt:n']) !== selProb(ctx, 'r:H') * selProb(ctx, 'bt:n'), 'mesmo jogo: não é o produto');
  near(selProb(ctx, 'c:o4.5') + selProb(ctx, 'c:u4.5'), 1, 1e-6, 'cartões: mais + menos = 1');
  near(jointProb(ctx, ['c:o3.5', 'c:u5.5']), selProb(ctx, 'c:o3.5') - selProb(ctx, 'c:o5.5'), 1e-9, 'faixa de cartões');
  assert.equal(jointProb(ctx, ['v:s', 'v:n']), 0);
  near(jointProb(ctx, ['r:H', 'k:o9.5']), selProb(ctx, 'r:H') * selProb(ctx, 'k:o9.5'), 1e-12, 'escanteios independentes do placar');
});

test('radar: leitura das prop bets e candidatos', async () => {
  const { parseProps, radarPicks } = await import('../site/js/engine/radar.js');
  const it = (name, o, extra = {}) => ({ type: { name }, current: { over: { decimal: o }, ...(extra.L != null ? { target: { value: extra.L } } : {}) }, ...(extra.t ? { team: { $ref: `x/teams/${extra.t}?y` } } : {}) });
  // escanteios: primeiro as linhas de "mais" (odds sobem com a linha), depois as de "menos"
  const items = [it('Total Corners', 1.74, { L: 10.5 }), it('Total Corners', 1.26, { L: 8.5 }), it('Total Corners', 2.0, { L: 10.5 }), it('Total Corners', 3.5, { L: 8.5 }),
    it('Both Teams To Score', 1.55), it('Both Teams To Score', 2.35), it('Double Chance', 1.76, { t: '10' }), it('Double Chance', 1.29, { t: '20' }), it('Double Chance', 1.29)];
  const o = Object.fromEntries(parseProps(items, '10', '20').map((x) => [x.sel, x]));
  assert.equal(o['k:o10.5'].o, 1.74); assert.equal(o['k:u8.5'].o, 3.5); assert.equal(o['bt:s'].o, 1.55); assert.equal(o['dc:1X'].o, 1.76);
  near(o['k:o10.5'].q + o['k:u10.5'].q, 1, 1e-9, 'par sem margem');
  near(o['dc:1X'].q + o['dc:X2'].q + o['dc:12'].q, 2, 1e-9, 'dupla chance soma 2');
  const picks = radarPicks([{ sel: 'a', o: 5, q: 0.19, p: 0.27 }, { sel: 'b', o: 2, q: 0.5, p: 0.52 }]);
  assert.deepEqual(picks.map((x) => x.sel), ['a'], 'só a discordância grande entra');
  assert.ok(picks[0].contra, 'odd ≥ 3 marcada');
});
