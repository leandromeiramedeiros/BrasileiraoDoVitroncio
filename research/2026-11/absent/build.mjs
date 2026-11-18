// Builds the missing-player features with absent.mjs on top of lineup.mjs, chronologically over every Série A/B match.
//   node build.mjs   → features_absent.csv (finished matches), future_absent.csv (unplayed fixtures), validation printout
// Card history: shared/matches.json ev (Y / R events with ESPN athlete ids). Lineup state: players/js/rosters.json.gz.
import { readFileSync, writeFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { createLineupState, applyLineupMatch, teamSnapshot } from '../players/js/lineup.mjs';
import { createCardState, applyCardMatch, absentFeatures, serveBans, suspendedFor } from './absent.mjs';

const HERE = new URL('.', import.meta.url).pathname;
const canon = (t, lg) => { t = String(t); return t === '5266' ? '3395' : t === '6080' ? '3448' : t === '6274' && lg === 'A' ? '4929' : t === '7385' && lg === 'B' ? '3448' : t; };
const rosters = new Map(JSON.parse(gunzipSync(readFileSync(HERE + '../players/js/rosters.json.gz')).toString('utf8')).map((r) => [r.id, r]));
const matches = JSON.parse(readFileSync(HERE + '../shared/matches.json', 'utf8'));

const lstate = createLineupState();
const cards = createCardState(), cardsAlt = createCardState();   // alt: the yellow of a Y+R pair counts towards accumulation
const COLS = ['susp_w', 'susp_g', 'susp_n', 'abs1_w', 'abs1_g', 'abs2_w', 'absP_w', 'abs1_n'];
const out = ['id,lg,season,' + ['h', 'a'].flatMap((s) => COLS.map((c) => `${s}_${c}`)).join(',')];
const fut = ['id,date,lg,season,h,a,h_ahead,a_ahead,' + ['h', 'a'].flatMap((s) => COLS.map((c) => `${s}_${c}`)).join(',')];
const V = {};                            // reason -> [n, appeared in XI, appeared on bench, regulars]
const VA = { n: 0, app: 0, nm: 0, appm: 0 };  // bans booked by only one of the two rules (alt-only / main-only)
const ahead = {};
const fmt = (v) => (v == null ? '' : v);
for (const m of matches) {
  if (m.status !== 'post' && m.status !== 'pre') continue;
  const r = rosters.get(String(m.id));
  const h = r ? r.h : canon(m.h, m.lg), a = r ? r.a : canon(m.a, m.lg);
  const F = {};
  for (const [s, t, side] of [['h', h, r && r.home], ['a', a, r && r.away]]) {
    const snap = teamSnapshot(lstate, t, m.date);
    F[s] = absentFeatures(snap, lstate.teams[t], cards, t, m.lg, m.season, m.date);
    if (side && m.status === 'post') {               // validation: a suspended player should not be in this matchday squad
      const xi = new Set(side.xi), bench = new Set(side.bench);
      for (const pid of suspendedFor(cards, t, m.lg, m.season)) {
        const k = cards.pend[pid].why + (m.season >= 2013 ? ' 2013+' : ' <2013'); const v = V[k] || (V[k] = [0, 0, 0, 0]);
        v[0]++; if (xi.has(pid)) v[1]++; else if (bench.has(pid) || (side.min && side.min[pid])) v[2]++;
        if (snap.p[pid] && snap.p[pid][0] >= 0.03) v[3]++;
      }
      const main = new Set(suspendedFor(cards, t, m.lg, m.season));
      const alt = new Set(suspendedFor(cardsAlt, t, m.lg, m.season));
      for (const pid of alt) if (!main.has(pid)) { VA.n++; if (xi.has(pid) || bench.has(pid)) VA.app++; }
      for (const pid of main) if (!alt.has(pid)) { VA.nm++; if (xi.has(pid) || bench.has(pid)) VA.appm++; }
    }
  }
  if (m.status === 'post') {
    out.push([m.id, m.lg, m.season, ...['h', 'a'].flatMap((s) => COLS.map((c) => fmt(F[s][c])))].join(','));
    if (r) applyLineupMatch(lstate, r);
    const min = r ? { ...r.home.min, ...r.away.min } : null;
    applyCardMatch(cards, { lg: m.lg, season: m.season, h, a, ev: m.ev, min });
    applyCardMatch(cardsAlt, { lg: m.lg, season: m.season, h, a, ev: m.ev, min }, { yrYellowCounts: true });
  } else {
    // unplayed fixture: features from everything known now; then roll the team forward (its bans are served there)
    ahead[h] = (ahead[h] || 0) + 1; ahead[a] = (ahead[a] || 0) + 1;
    fut.push([m.id, m.date, m.lg, m.season, h, a, ahead[h], ahead[a], ...['h', 'a'].flatMap((s) => COLS.map((c) => fmt(F[s][c])))].join(','));
    serveBans(cards, h, m.lg, m.season); serveBans(cards, a, m.lg, m.season);
  }
}
writeFileSync(HERE + 'features_absent.csv', out.join('\n') + '\n');
writeFileSync(HERE + 'future_absent.csv', fut.join('\n') + '\n');
console.log(`features_absent.csv ${out.length - 1} matches | future_absent.csv ${fut.length - 1} fixtures`);
console.log('suspension-rule validation (matches with a lineup): reason  n  inXI  onBench  rate_listed  regulars');
let T = [0, 0, 0, 0];
for (const [k, v] of Object.entries(V).sort()) { console.log(`  ${k.padEnd(12)} ${v[0]}  ${v[1]}  ${v[2]}  ${((v[1] + v[2]) / v[0] * 100).toFixed(2)}%  ${v[3]}`); T = T.map((x, i) => x + v[i]); }
console.log(`  ALL    ${T[0]}  ${T[1]}  ${T[2]}  ${((T[1] + T[2]) / T[0] * 100).toFixed(2)}%  ${T[3]}`);
console.log(`Y+R pair: bans booked only if its yellow counts ${VA.n} (listed in squad ${VA.app}, ${(VA.app / Math.max(1, VA.n) * 100).toFixed(1)}%) | only if it does not count ${VA.nm} (listed ${VA.appm}, ${(VA.appm / Math.max(1, VA.nm) * 100).toFixed(1)}%)`);
