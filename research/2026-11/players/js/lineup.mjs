// ─────────────────────────────────────────────────────────────────────────────
// Lineup features for the Vitrôncio match model — pure ES module, no Node APIs,
// no dependencies (runs unchanged in the weekly refit and in the browser).
//
// For each team we keep its last 20 league matches (who started, who was on the
// bench, minutes, goals+assists) and every player's career minutes. From that:
//
//   weight w_i   decayed minute share of player i over the team's last 10 league
//                matches (0.85 per match back, matches older than 400 days ignored),
//                divided by the sum of the 11 largest shares  → the "usual XI" sums to 1
//   share  g_i   player i's share of the team's goals+assists over its last 20
//                matches (0.95 per match back)
//   career c_i   log(1 + career full matches) of player i (all matches fed to the state)
//
// Features of an announced lineup (xi = 11 starters, bench = substitutes):
//   usage  = Σ_xi w_i        how much of the usual XI starts                      ← used by the production shift
//   ga     = Σ_xi g_i        how much of the team's recent goals+assists starts   ← used by the production shift
//   bench  = Σ_bench w_i     usual players available on the bench                 ← used by the production shift
//   exp    = mean_xi c_i     experience of the XI (diagnostic: adds nothing on DEV, not used; only reason for state.career)
// plus the same four CENTRED on what the team usually fields (c_usage, c_ga, c_exp, c_bench: feature minus its expectation
// when every player starts / sits on the bench as often as he did in the last 10 matches). The centred versions are
// diagnostics for the site ("this XI is weaker than usual"); the production shift uses the LEVELS (centred lost
// 0.0004 RPS on DEV).
//
// Production shift (coefficients in lineup_coefs.json), for every feature that is non-null for BOTH teams:
//   log λ_home += b·(x_home − x_away),   log λ_away −= b·(x_home − x_away)
// Lineup not announced yet: snapshotFeatures(snapshot, null) returns the team's EXPECTED usage / ga / bench.
//
// usage and ga are null until the team has played 3 matches of the current season (before that the window still
// describes last year's squad) or when its last match is more than 75 days old; bench is null without a bench list.
// ─────────────────────────────────────────────────────────────────────────────

export const K_USE = 10, D_USE = 0.85;   // window / decay of the "usual XI"
export const K_GA = 20, D_GA = 0.95;     // window / decay of goal+assist shares
export const MAX_DAYS = 400;             // older team matches are ignored
export const RECENT_DAYS = 75, MIN_SEASON = 3;   // season-start gate (see teamSnapshot)

const r4 = (x) => Math.round(x * 1e4) / 1e4;
const dayOf = (iso) => Math.floor(Date.parse(iso) / 864e5);
const NONE = Object.freeze({ usage: null, ga: null, exp: null, bench: null, c_usage: null, c_ga: null, c_exp: null, c_bench: null });

export function createLineupState() {
  return { teams: {}, career: {} };   // plain JSON: teams[id] = last 20 matches, career[playerId] = full-match equivalents
}

/** Everything needed to turn an announced lineup of `teamId` into features, as a small JSON object. */
export function teamSnapshot(state, teamId, isoDate) {
  const day = dayOf(isoDate);
  const hist = state.teams[teamId] || [];
  const recs = [];
  for (let i = hist.length - 1; i >= 0 && recs.length < K_GA; i--) {
    if (day - hist[i].day > MAX_DAYS) break;
    recs.push(hist[i]);
  }
  // season-start gate: the team's latest match is recent and it has played at least MIN_SEASON matches in that season
  let nSeason = 0;
  for (const r of recs) if (r.season === recs[0].season) nSeason++;
  const inSeason = recs.length > 0 && day - recs[0].day <= RECENT_DAYS && nSeason >= MIN_SEASON;
  const u = {}, s = {}, b = {}, g = {};
  let W = 0, G = 0;
  recs.forEach((r, j) => {
    if (j < K_USE) {
      const w = D_USE ** j; W += w;
      for (const id in r.min) u[id] = (u[id] || 0) + (w * r.min[id]) / 90;
      for (const id of r.xi) s[id] = (s[id] || 0) + w;
      for (const id of r.bench) b[id] = (b[id] || 0) + w;
    }
    const w2 = D_GA ** j;
    for (const id in r.ga) { g[id] = (g[id] || 0) + w2 * r.ga[id]; G += w2 * r.ga[id]; }
  });
  const top11 = Object.values(u).sort((x, y) => y - x).slice(0, 11).reduce((a, x) => a + x, 0);
  const p = {};
  let S = 0, eU = 0, eG = 0, eC = 0, eB = 0;
  for (const id of new Set([...Object.keys(u), ...Object.keys(s), ...Object.keys(b), ...Object.keys(g)])) {
    const w = top11 > 0 ? r4((u[id] || 0) / top11) : 0;
    const gs = G > 0 ? r4((g[id] || 0) / G) : 0;
    const c = r4(Math.log1p(state.career[id] || 0));
    p[id] = [w, gs, c];
    const st = W > 0 ? (s[id] || 0) / W : 0, bs = W > 0 ? (b[id] || 0) / W : 0;
    S += st; eU += st * w; eG += st * gs; eC += st * c; eB += bs * w;
  }
  return {
    team: String(teamId), date: isoDate, n: recs.length, ok: inSeason, hasGa: G > 0,
    p,                                             // playerId -> [w, g, c]
    avg: S > 0 ? { usage: r4((11 * eU) / S), ga: r4((11 * eG) / S), exp: r4(eC / S), bench: r4(eB) } : null,
  };
}

/**
 * Features of an announced lineup from a snapshot only (browser side). `bench` may be empty / omitted.
 * With xi = null (lineup not announced yet) it returns the team's EXPECTED features (what it usually fields):
 * usage/ga/exp/bench = the snapshot averages, centred features = 0, plus expected: true.
 */
export function snapshotFeatures(snap, xi, bench = []) {
  if (!snap || !snap.n) return { ...NONE };
  if (!xi || !xi.length) {
    const a = snap.avg; if (!a) return { ...NONE };
    return { usage: snap.ok ? a.usage : null, ga: snap.ok && snap.hasGa ? a.ga : null, exp: a.exp, bench: a.bench,
      c_usage: snap.ok ? 0 : null, c_ga: snap.ok && snap.hasGa ? 0 : null, c_exp: snap.ok ? 0 : null, c_bench: snap.ok ? 0 : null, expected: true };
  }
  let us = 0, ga = 0, ex = 0, be = 0;
  for (const id of xi) { const q = snap.p[id]; if (q) { us += q[0]; ga += q[1]; ex += q[2]; } }
  const k = 11 / xi.length;                         // tolerate a lineup with 10 listed starters
  us *= k; ga *= k; ex /= xi.length;
  for (const id of bench || []) { const q = snap.p[id]; if (q) be += q[0]; }
  const hasBench = !!(bench && bench.length), ok = snap.ok && !!snap.avg, a = snap.avg || {};
  return {
    usage: snap.ok ? r4(us) : null,
    ga: snap.ok && snap.hasGa ? r4(ga) : null,
    exp: r4(ex),
    bench: hasBench ? r4(be) : null,
    c_usage: ok ? r4(us - a.usage) : null,
    c_ga: ok && snap.hasGa ? r4(ga - a.ga) : null,
    c_exp: ok ? r4(ex - a.exp) : null,
    c_bench: ok && hasBench ? r4(be - a.bench) : null,
  };
}

/** PRE-match features of `teamId` for the lineup (xi, bench) on `isoDate`. Same numbers as the snapshot route. */
export function lineupFeatures(state, teamId, xi, bench, isoDate) {
  return snapshotFeatures(teamSnapshot(state, teamId, isoDate), xi, bench);
}

/** Feed a finished match: match = { id, date, lg, season, h, a, home: { xi, bench, min, ga }, away: { ... } }. */
export function applyLineupMatch(state, match) {
  const day = dayOf(match.date);
  for (const [tid, side] of [[match.h, match.home], [match.a, match.away]]) {
    if (!side || !side.xi || side.xi.length < 10) continue;
    const hist = state.teams[tid] || (state.teams[tid] = []);
    hist.push({ day, season: match.season ?? +String(match.date).slice(0, 4), xi: side.xi.slice(), bench: (side.bench || []).slice(), min: { ...(side.min || {}) }, ga: { ...(side.ga || {}) } });
    if (hist.length > K_GA) hist.shift();
    for (const id in side.min || {}) state.career[id] = (state.career[id] || 0) + side.min[id] / 90;
  }
  return state;
}
