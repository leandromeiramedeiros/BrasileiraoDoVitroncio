// Parity + results for lineup.mjs.
//   node check.mjs            -> features_js.csv, parity with the Python features, walk-forward shift on top of ctx_core
//                                (lambda_lineup_js.csv, preds_lineup_js.csv), production coefficients (lineup_coefs.json)
//   node check.mjs features   -> only features_js.csv + parity
// Shift:  log λ_home += Σ b_j·(x_j,home − x_j,away),  log λ_away −= the same.  A feature that is null for either team contributes 0.
// Coefficients for season Y: Poisson regression with offset on Série A seasons 2009..Y−1 (offset = ctx_core expected goals
// where they exist, i.e. 2013+, v1 expected goals before that). No scores are computed here (use shared/score.py).
import { readFileSync, writeFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { execFileSync } from 'node:child_process';
import { createLineupState, lineupFeatures, applyLineupMatch, teamSnapshot, snapshotFeatures } from './lineup.mjs';

const HERE = new URL('.', import.meta.url).pathname;
const SHARED = '/home/user/bdv-lab/shared/';
const CTX = '/home/user/bdv-lab/context/';
export const FEATURES = ['usage', 'ga', 'bench'];                // chosen on DEV (see select_dev.py): levels, exp dropped
const ALL = ['usage', 'ga', 'exp', 'bench', 'c_usage', 'c_ga', 'c_exp', 'c_bench'];
const EXP = ['usage', 'ga', 'exp', 'bench'];                     // e_* columns: expected values when the lineup is unknown
const FIRST_FIT = 2009;

const csv = (path) => {
  const [head, ...rows] = readFileSync(path, 'utf8').trim().split(/\r?\n/);
  const h = head.split(',');
  return rows.map((l) => { const r = l.split(','); const o = {}; h.forEach((k, i) => (o[k] = r[i])); return o; });
};
const num = (v) => (v === '' || v == null ? null : +v);

// ── 1. run the module over every finished match, chronologically ──────────────────────────────────────────────
const matches = JSON.parse(gunzipSync(readFileSync(HERE + 'rosters.json.gz')).toString('utf8'));
const state = createLineupState();
const feat = new Map();   // id -> { h: {...}, a: {...} }
let snapBytes = 0, snapN = 0, mismatches = 0;
for (const m of matches) {
  const f = {};
  for (const [side, tid, t] of [['h', m.h, m.home], ['a', m.a, m.away]]) {
    f[side] = lineupFeatures(state, tid, t.xi, t.bench, m.date);
    // the browser route: snapshot -> JSON -> features must give the same numbers
    const js = JSON.stringify(teamSnapshot(state, tid, m.date));
    const g = snapshotFeatures(JSON.parse(js), t.xi, t.bench);
    if (ALL.some((k) => f[side][k] !== g[k])) mismatches++;
    const e = snapshotFeatures(JSON.parse(js), null);            // lineup not announced: the team's expected features
    for (const k of EXP) f[side]['e_' + k] = e[k];
    if (m.season >= 2025) { snapBytes += js.length; snapN++; }
  }
  feat.set(m.id, f);
  applyLineupMatch(state, m);
}
const OUTC = [...ALL, ...EXP.map((k) => 'e_' + k)];
const out = ['id,side,' + OUTC.join(',')];
for (const [id, f] of feat) for (const s of ['h', 'a']) out.push([id, s, ...OUTC.map((k) => (f[s][k] == null ? '' : f[s][k]))].join(','));
writeFileSync(HERE + 'features_js.csv', out.join('\n') + '\n');
console.log(`matches ${matches.length} | features_js.csv rows ${out.length - 1} | snapshot route mismatches ${mismatches} | mean snapshot size ${(snapBytes / snapN / 1024).toFixed(2)} KB (2025-26)`);
console.log(`state size: ${(JSON.stringify(state).length / 1e6).toFixed(2)} MB (teams ${Object.keys(state.teams).length}, players ${Object.keys(state.career).length})`);

// ── 2. parity with the Python features ────────────────────────────────────────────────────────────────────────
{
  const [head, ...rows] = readFileSync('/home/user/bdv-lab/players/work/side_features.csv', 'utf8').trim().split(/\r?\n/);
  const h = head.split(','); const ix = (k) => h.indexOf(k);
  const cols = { id: ix('id'), side: ix('side'), lg: ix('lg'), season: ix('season'), ns: ix('n_season'), usage: ix('k_usage'), ga: ix('k_gashare'), exp: ix('k_exp'), bench: ix('k_bench_usage'), eu: ix('u_e_usage'), eg: ix('u_e_gashare'), ee: ix('u_exp') };
  const pairs = { usage: [], ga: [], exp: [], bench: [], c_usage: [], c_ga: [], c_exp: [] };
  for (const l of rows) {
    const r = l.split(','); const f = feat.get(r[cols.id]); if (!f) continue;
    if (r[cols.lg] !== 'A' || +r[cols.season] < 2013) continue;
    const j = f[r[cols.side]]; const inSeason = +r[cols.ns] >= 3;
    const py = { usage: inSeason ? num(r[cols.usage]) : null, ga: inSeason ? num(r[cols.ga]) : null, exp: num(r[cols.exp]), bench: num(r[cols.bench]) };
    py.c_usage = py.usage == null || num(r[cols.eu]) == null ? null : py.usage - num(r[cols.eu]);
    py.c_ga = py.ga == null || num(r[cols.eg]) == null ? null : py.ga - num(r[cols.eg]);
    py.c_exp = py.exp == null || num(r[cols.ee]) == null ? null : py.exp - num(r[cols.ee]);
    for (const k of Object.keys(pairs)) if (j[k] != null && py[k] != null) pairs[k].push([j[k], py[k]]);
  }
  const corr = (p) => { const n = p.length; let sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0; for (const [x, y] of p) { sx += x; sy += y; sxx += x * x; syy += y * y; sxy += x * y; } return (n * sxy - sx * sy) / Math.sqrt((n * sxx - sx * sx) * (n * syy - sy * sy)); };
  console.log('parity with the Python features (Série A 2013+, Pearson r, n):');
  for (const [k, p] of Object.entries(pairs)) console.log(`  ${k.padEnd(8)} r = ${corr(p).toFixed(4)}  n = ${p.length}  ${k.startsWith('c_') ? '(Python: level minus expected-XI column, suspended players removed)' : ''}`);
}
if (process.argv[2] === 'features') process.exit(0);

// ── 3. Poisson regression with offset (Newton) ────────────────────────────────────────────────────────────────
function solve(A, b) {   // Gauss-Jordan, returns [x, inverse]
  const n = b.length; const M = A.map((r, i) => [...r, ...Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)), b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    const d = M[c][c]; for (let j = 0; j < 2 * n + 1; j++) M[c][j] /= d;
    for (let r = 0; r < n; r++) if (r !== c) { const f = M[r][c]; if (f) for (let j = 0; j < 2 * n + 1; j++) M[r][j] -= f * M[c][j]; }
  }
  return [M.map((r) => r[2 * n]), M.map((r) => r.slice(n, 2 * n))];
}
function fitPoisson(rows, nf) {
  let b = new Array(nf).fill(0), cov = null;
  for (let it = 0; it < 25; it++) {
    const g = new Array(nf).fill(0), H = Array.from({ length: nf }, () => new Array(nf).fill(0));
    for (const r of rows) {
      let e = 0; for (let j = 0; j < nf; j++) e += b[j] * r.d[j];
      const mh = r.lh * Math.exp(e), ma = r.la * Math.exp(-e);
      const res = r.hg - mh - (r.ag - ma), w = mh + ma;
      for (let j = 0; j < nf; j++) { g[j] += res * r.d[j]; for (let k = 0; k < nf; k++) H[j][k] += w * r.d[j] * r.d[k]; }
    }
    const [step, inv] = solve(H, g); cov = inv;
    b = b.map((v, j) => v + step[j]);
    if (Math.max(...step.map(Math.abs)) < 1e-9) break;
  }
  return { b, se: cov.map((r, j) => Math.sqrt(r[j])) };
}

// ── 4. data: results, offsets ─────────────────────────────────────────────────────────────────────────────────
const lamCtx = new Map(csv(CTX + 'coef/lambda_ctx_core.csv').map((r) => [r.id, [+r.lh, +r.la]]));
const lamV1 = new Map(csv(SHARED + 'baseline_features.csv').map((r) => [r.id, [+r.lh, +r.la]]));
const all = JSON.parse(readFileSync(SHARED + 'matches.json', 'utf8')).filter((m) => m.lg === 'A');
const diff = (id, pre = '') => {
  const f = feat.get(String(id));
  return FEATURES.map((k) => (f && f.h[pre + k] != null && f.a[pre + k] != null ? f.h[pre + k] - f.a[pre + k] : 0));   // needs both teams
};
const rowsA = all.filter((m) => m.status === 'post' && m.season >= FIRST_FIT && feat.has(String(m.id)))
  .map((m) => { const l = lamCtx.get(String(m.id)) || lamV1.get(String(m.id)); return { season: m.season, lh: l[0], la: l[1], hg: m.hg, ag: m.ag, d: diff(m.id) }; });

// ── 5. walk-forward shift on top of ctx_core ──────────────────────────────────────────────────────────────────
// two outputs: (1) announced lineup; (2) "pre": lineup not announced, every feature replaced by the team's expected value
const lamOut = ['id,lh,la'], lamPre = ['id,lh,la'], shiftOut = ['id,season,shift_lineup,shift_expected'], wf = {};
for (const m of all) {
  const l = lamCtx.get(String(m.id)); if (!l) continue;
  if (!wf[m.season]) wf[m.season] = fitPoisson(rowsA.filter((r) => r.season < m.season), FEATURES.length);
  const dot = (d) => d.reduce((a, x, j) => a + wf[m.season].b[j] * x, 0);
  const eP = m.status === 'post' ? dot(diff(m.id, 'e_')) : 0;      // unplayed fixtures are not in rosters.json.gz -> 0 here
  const e = m.status === 'post' ? dot(diff(m.id)) : eP;
  lamOut.push([m.id, l[0] * Math.exp(e), l[1] * Math.exp(-e)].join(','));
  lamPre.push([m.id, l[0] * Math.exp(eP), l[1] * Math.exp(-eP)].join(','));
  shiftOut.push([m.id, m.season, e.toFixed(5), eP.toFixed(5)].join(','));
}
writeFileSync(HERE + 'lambda_lineup_js.csv', lamOut.join('\n') + '\n');
writeFileSync(HERE + 'lambda_lineup_js_pre.csv', lamPre.join('\n') + '\n');
writeFileSync(HERE + 'shift_lineup_js.csv', shiftOut.join('\n') + '\n');
execFileSync('node', [CTX + 'code/probs.mjs', HERE + 'lambda_lineup_js.csv', HERE + 'preds_lineup_js.csv']);
execFileSync('node', [CTX + 'code/probs.mjs', HERE + 'lambda_lineup_js_pre.csv', HERE + 'preds_lineup_js_pre.csv']);
console.log(`walk-forward: wrote lambda_lineup_js[_pre].csv, shift_lineup_js.csv, preds_lineup_js[_pre].csv (${lamOut.length - 1} matches)`);
console.log('walk-forward coefficients by season:');
for (const y of Object.keys(wf).sort()) console.log(`  ${y}: ` + FEATURES.map((k, j) => `${k} ${wf[y].b[j].toFixed(3)}`).join('  '));

// ── 6. production coefficients (all finished Série A matches 2009–2026) ───────────────────────────────────────
const prod = fitPoisson(rowsA, FEATURES.length);
console.log(`production coefficients (Série A ${FIRST_FIT}-2026, n = ${rowsA.length} matches, per unit of home-minus-away feature):`);
FEATURES.forEach((k, j) => console.log(`  ${k.padEnd(8)} b = ${prod.b[j].toFixed(4)}  se = ${prod.se[j].toFixed(4)}  z = ${(prod.b[j] / prod.se[j]).toFixed(2)}`));
const sd = FEATURES.map((_, j) => { const v = rowsA.map((r) => r.d[j]); const mu = v.reduce((a, x) => a + x, 0) / v.length; return Math.sqrt(v.reduce((a, x) => a + (x - mu) ** 2, 0) / v.length); });
console.log('  sd of the home-minus-away feature: ' + FEATURES.map((k, j) => `${k} ${sd[j].toFixed(3)}`).join('  '));
writeFileSync(HERE + 'lineup_coefs.json', JSON.stringify({ features: FEATURES, type: 'diff', b: prod.b.map((v) => +v.toFixed(5)), se: prod.se.map((v) => +v.toFixed(5)), fit: `Serie A ${FIRST_FIT}-2026`, n: rowsA.length,
  rule: 'log lambda_home += sum_j b_j*(x_j_home - x_j_away); log lambda_away -= the same; a feature that is null for either team contributes 0; lineup not announced: use snapshotFeatures(snapshot, null)' }, null, 1) + '\n');

// ── 7. example of the browser flow: one snapshot per team for its next Série A fixture ───────────────────────────
{
  const canon = (t, lg) => (t === '5266' ? '3395' : t === '6080' ? '3448' : t === '6274' && lg === 'A' ? '4929' : t === '7385' && lg === 'B' ? '3448' : t);
  const snaps = {}, fixtures = [];
  for (const m of all.filter((x) => x.status === 'pre')) {
    const row = { id: String(m.id), date: m.date, h: canon(String(m.h), 'A'), a: canon(String(m.a), 'A') };
    for (const t of [row.h, row.a]) if (!snaps[t]) snaps[t] = teamSnapshot(state, t, m.date);     // the team's NEXT fixture
    const eh = snapshotFeatures(snaps[row.h], null), ea = snapshotFeatures(snaps[row.a], null);
    row.shift_expected = +FEATURES.reduce((s, k, j) => s + (eh[k] != null && ea[k] != null ? prod.b[j] * (eh[k] - ea[k]) : 0), 0).toFixed(5);
    fixtures.push(row);
  }
  const js = JSON.stringify({ generated: new Date().toISOString().slice(0, 10), coefs: { features: FEATURES, b: prod.b.map((v) => +v.toFixed(5)) }, fixtures, snapshots: snaps });
  writeFileSync(HERE + 'snapshots_next.json', js + '\n');
  console.log(`snapshots_next.json: ${Object.keys(snaps).length} team snapshots, ${fixtures.length} fixtures, ${(js.length / 1024).toFixed(1)} KB`);
}
